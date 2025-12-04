import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'modnet')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'FBA')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../webapp')))

from modnet_architecture import MODNet
from fba_model import load_fba_pipeline, predict_fba_pipeline

# Configuration
IMG_SIZE = 256 # FBA prefers 256+
NUM_SAMPLES = None # Evaluate on 100 samples for speed, or set to None for full test set
MODNET_PATH = '../models/modnet/modnet_pet_matting.keras'
UNET_PATH = '../models/fba/unet/pet_unet_improved_final.keras'
FBA_CKPT_PATH = '../models/fba/fba_pet_final.pth'

def load_modnet():
    print("Loading MODNet...")
    try:
        model = tf.keras.models.load_model(
            MODNET_PATH,
            custom_objects={'MODNet': MODNet},
            compile=False
        )
        print("MODNet loaded.")
        return model
    except Exception as e:
        print(f"Failed to load MODNet: {e}")
        return None

def load_fba():
    print("Loading FBA Pipeline...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline = load_fba_pipeline(UNET_PATH, FBA_CKPT_PATH, device=device)
        print("FBA Pipeline loaded.")
        return pipeline
    except Exception as e:
        print(f"Failed to load FBA Pipeline: {e}")
        return None

def calculate_metrics(y_true, y_pred):
    """Calculate IoU, Dice, MAE"""
    # Ensure binary mask for IoU/Dice (threshold 0.5)
    pred_mask = y_pred > 0.5
    true_mask = y_true > 0.5
    
    # Intersection and Union
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    # IoU
    iou = intersection / (union + 1e-7)
    
    # Dice
    dice = 2 * intersection / (pred_mask.sum() + true_mask.sum() + 1e-7)
    
    # MAE (Mean Absolute Error) on continuous values
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {'iou': iou, 'dice': dice, 'mae': mae}

def evaluate():
    print(f"Loading Oxford-IIIT Pet dataset (Test split)...")
    ds, info = tfds.load('oxford_iiit_pet:3.2.0', split='test', with_info=True)
    
    # Load models
    modnet = load_modnet()
    fba = load_fba()
    
    modnet_metrics = {'iou': [], 'dice': [], 'mae': []}
    fba_metrics = {'iou': [], 'dice': [], 'mae': []}
    
    print(f"Evaluating on {NUM_SAMPLES if NUM_SAMPLES else 'all'} samples...")
    
    count = 0
    for data in tqdm(ds):
        if NUM_SAMPLES and count >= NUM_SAMPLES:
            break
            
        image = data['image'].numpy()
        mask = data['segmentation_mask'].numpy()
        
        # Preprocess Ground Truth
        # Mask: 1=Foreground, 2=Background, 3=Unknown (Trimap)
        # We want binary alpha: 1=Pet, 0=Background
        # Oxford Pet: 1=Pet, 2=Background, 3=Border
        # So we want mask == 1
        gt_alpha = (mask == 1).astype(np.float32)
        if gt_alpha.ndim == 3:
            gt_alpha = gt_alpha[:, :, 0]
            
        # Convert to PIL for resizing
        img_pil = Image.fromarray(image)
        gt_pil = Image.fromarray((gt_alpha * 255).astype(np.uint8))
        
        # --- MODNet Evaluation ---
        if modnet:
            # MODNet expects 128x128
            img_modnet = img_pil.resize((128, 128), Image.BILINEAR)
            img_np = np.array(img_modnet).astype(np.float32) / 255.0
            img_tensor = np.expand_dims(img_np, 0)
            
            pred = modnet.predict(img_tensor, verbose=0)[0, :, :, 0]
            pred_alpha = 1.0 - pred
            
            # Resize GT to 128x128 for comparison
            gt_modnet = gt_pil.resize((128, 128), Image.NEAREST)
            gt_alpha_modnet = np.array(gt_modnet).astype(np.float32) / 255.0
            
            metrics = calculate_metrics(gt_alpha_modnet, pred_alpha)
            for k, v in metrics.items():
                modnet_metrics[k].append(v)
                
        # --- FBA Evaluation ---
        if fba:
            # FBA prefers 256x256 (or larger)
            img_fba = img_pil.resize((256, 256), Image.BILINEAR)
            
            # predict_fba_pipeline returns numpy array in [0, 1] range
            pred_alpha_np = predict_fba_pipeline(fba, img_fba)
            
            # Resize GT to 256x256 for comparison
            gt_fba = gt_pil.resize((256, 256), Image.NEAREST)
            gt_alpha_fba = np.array(gt_fba).astype(np.float32) / 255.0
            
            metrics = calculate_metrics(gt_alpha_fba, pred_alpha_np)
            for k, v in metrics.items():
                fba_metrics[k].append(v)
        
        count += 1
        
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    
    if modnet:
        print("MODNet:")
        print(f"  IoU:  {np.mean(modnet_metrics['iou']):.4f}")
        print(f"  Dice: {np.mean(modnet_metrics['dice']):.4f}")
        print(f"  MAE:  {np.mean(modnet_metrics['mae']):.4f}")
        
    if fba:
        print("\nFBA (U-Net + Refinement):")
        print(f"  IoU:  {np.mean(fba_metrics['iou']):.4f}")
        print(f"  Dice: {np.mean(fba_metrics['dice']):.4f}")
        print(f"  MAE:  {np.mean(fba_metrics['mae']):.4f}")

if __name__ == "__main__":
    evaluate()
