"""
Simplified FBA Integration for Web App
Uses U-Net + FBA without external dependencies
"""

import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Add FBA-model to path
FBA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../code/fba'))
FBA_MATTING_PATH = os.path.join(FBA_PATH, 'FBA_Matting')
sys.path.insert(0, FBA_PATH)
sys.path.insert(0, FBA_MATTING_PATH)  # Add FBA_Matting to path for internal imports


class SimpleFBAPipeline:
    
    def __init__(self, unet_model_path, fba_checkpoint_path, img_size=256):
        self.img_size = img_size
        
        # Load U-Net
        print("Loading U-Net segmentation model...")
        self.unet_model = tf.keras.models.load_model(unet_model_path, compile=False)
        print("U-Net loaded")
        
        # Load FBA
        try:
            import torch
            from FBA_Matting.networks.models import build_model
            
            print("Loading FBA model...")
            self.fba_model = build_model(fba_checkpoint_path)
            self.fba_model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.fba_model = self.fba_model.to(self.device)
            print("FBA loaded")
        except Exception as e:
            print(f"FBA loading failed: {e}")
            self.fba_model = None
    
    def predict_unet_alpha(self, image_np):
        """Get alpha from U-Net"""
        # Resize to model size
        h, w = image_np.shape[:2]
        image_resized = tf.image.resize(np.expand_dims(image_np, 0), (self.img_size, self.img_size))
        
        # Predict
        prediction = self.unet_model.predict(image_resized, verbose=0)
        
        # Invert (model outputs bg=1)
        # alpha_low_res = 1.0 - prediction
        # Update: It seems the model might be outputting fg=1 or the pipeline expects it differently.
        # Based on user feedback, the previous result was inverted.
        alpha_low_res = 1.0 - prediction
        
        # Resize back
        alpha_high_res = tf.image.resize(alpha_low_res, (h, w))
        
        return alpha_high_res.numpy()[0, :, :, 0]
    
    def create_trimap(self, alpha, erosion=5, dilation=5):
        """Create trimap from alpha matte"""
        from scipy.ndimage import binary_erosion, binary_dilation
        
        binary_mask = (alpha > 0.5).astype(np.uint8)
        
        # Erode for foreground
        fg_mask = binary_erosion(binary_mask, iterations=erosion)
        
        # Dilate for unknown region
        dilated_mask = binary_dilation(binary_mask, iterations=dilation)
        
        # Create trimap
        trimap = np.full(alpha.shape, 128, dtype=np.uint8)
        trimap[dilated_mask == 0] = 0  # Background
        trimap[fg_mask == 1] = 255  # Foreground
        
        return trimap
    
    def refine_with_fba(self, image_np, alpha_unet):
        """Refine alpha using FBA"""
        if self.fba_model is None:
            return alpha_unet
        
        try:
            import torch
            import cv2
            
            h_orig, w_orig = image_np.shape[:2]
            
            # Pad to multiple of 8 to avoid model size mismatch
            pad_h = (8 - h_orig % 8) % 8
            pad_w = (8 - w_orig % 8) % 8
            
            if pad_h > 0 or pad_w > 0:
                image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                alpha_unet = np.pad(alpha_unet, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            h, w = image_np.shape[:2]
            
            # Create trimap
            trimap = self.create_trimap(alpha_unet)
            
            # Prepare FBA inputs
            # 1. Two-channel trimap (FG, BG)
            two_chan_trimap = np.zeros((h, w, 2), dtype=np.float32)
            two_chan_trimap[:, :, 0] = (trimap == 255).astype(np.float32)
            two_chan_trimap[:, :, 1] = (trimap == 0).astype(np.float32)
            
            # 2. 6-channel distance features
            # Recreate logic from FBAMattingRefiner.py
            fg_mask = (trimap == 255).astype(np.uint8)
            bg_mask = (trimap == 0).astype(np.uint8)
            unk_mask = (trimap == 128).astype(np.uint8) # Or 1 - fg - bg
            
            # Distance transforms (distance TO the region, so invert mask)
            fg_input = (fg_mask == 0).astype(np.uint8)
            bg_input = (bg_mask == 0).astype(np.uint8)
            
            dist_fg = cv2.distanceTransform(fg_input, cv2.DIST_L2, 5)
            dist_bg = cv2.distanceTransform(bg_input, cv2.DIST_L2, 5)
            
            # Stack 6 channels: dist_fg, dist_bg, 1-dist_fg, 1-dist_bg, unk, fg
            trimap_transformed_np = np.stack(
                [
                    dist_fg,
                    dist_bg,
                    1.0 - dist_fg,
                    1.0 - dist_bg,
                    unk_mask.astype(np.float32),
                    fg_mask.astype(np.float32),
                ],
                axis=0,
            )
            
            # Convert to torch
            image_t = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            two_chan_t = torch.from_numpy(two_chan_trimap.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            trimap_t = torch.from_numpy(trimap_transformed_np).unsqueeze(0).float().to(self.device)
            
            # 3. Normalize image
            # ImageNet mean/std
            IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            image_n_t = (image_t - IMG_MEAN) / IMG_STD
            
            # Predict
            with torch.no_grad():
                output = self.fba_model(image_t, two_chan_t, image_n_t, trimap_t)
            
            # Extract alpha
            alpha_fba = output[0, 0].cpu().numpy()
            
            # Crop back to original size
            if pad_h > 0 or pad_w > 0:
                alpha_fba = alpha_fba[:h_orig, :w_orig]
            
            return alpha_fba
        except Exception as e:
            print(f"FBA refinement failed: {e}")
            return alpha_unet[:h_orig, :w_orig] if 'h_orig' in locals() else alpha_unet
    
    def predict(self, image_pil):
        """Full pipeline prediction"""
        # Convert to numpy
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        
        # Get U-Net alpha
        alpha_unet = self.predict_unet_alpha(image_np)
        
        # Refine with FBA
        alpha_final = self.refine_with_fba(image_np, alpha_unet)
        
        return alpha_final


def load_fba_pipeline(unet_model_path, fba_checkpoint_path, device='cpu'):
    """Load FBA pipeline"""
    return SimpleFBAPipeline(unet_model_path, fba_checkpoint_path)


def predict_fba_pipeline(pipeline, image_pil):
    """Predict using FBA pipeline"""
    return pipeline.predict(image_pil)
