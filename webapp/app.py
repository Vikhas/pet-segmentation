import os
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, request, send_file, jsonify, render_template
from PIL import Image
import io

# Add code directory to path for imports (relative to webapp/)
# Add code directory to path for imports (relative to webapp/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../code/modnet')))
from modnet_architecture import MODNet

# Import PyTorch for FBA model
import torch
from fba_model import load_fba_pipeline

app = Flask(__name__)

# Configuration
MODEL_PATH = '../models/modnet/modnet_pet_matting.keras'
UNET_MODEL_PATH = '../models/fba/unet/pet_unet_improved_final.keras'  # U-Net for FBA pipeline
FBA_MODEL_PATH = '../models/fba/fba_pet_final.pth'
IMG_SIZE = 128

# Global model variables
model = None
fba_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_modnet_model():
    """Load the trained MODNet model"""
    global model
    if model is None:
        print("Loading MODNet model...")
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'MODNet': MODNet},
                compile=False
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

def load_fba_model_func():
    """Load the FBA pipeline (U-Net + FBA refinement)"""
    global fba_model
    if fba_model is None:
        print("Loading FBA pipeline (U-Net + FBA refinement)...")
        try:
            from fba_model import load_fba_pipeline
            
            # Check if U-Net model exists
            if not os.path.exists(UNET_MODEL_PATH):
                print(f"Warning: U-Net model not found at {UNET_MODEL_PATH}")
                print("   FBA comparison will be unavailable")
                return
            
            fba_model = load_fba_pipeline(UNET_MODEL_PATH, FBA_MODEL_PATH, device=device)
            print("FBA Pipeline loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load FBA pipeline: {e}")
            print("   Comparison feature will be unavailable")
            fba_model = None

def predict_alpha(image):
    """
    Predict alpha matte for an image using MODNet
    Args:
        image: PIL Image
    Returns:
        alpha_mask: PIL Image (L mode)
    """
    # Ensure model is loaded
    global model
    if model is None:
        load_modnet_model()
        
    # Resize for model
    img_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array = np.array(img_resized) / 255.0
    img_tensor = tf.expand_dims(img_array, 0)
    
    # Predict
    alpha_pred = model.predict(img_tensor, verbose=0)
    
    # Invert alpha (model trained with bg=1, fg=0)
    alpha = 1.0 - alpha_pred[0, :, :, 0]
    
    # Resize back to original size
    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
    alpha_mask = alpha_img.resize(image.size, Image.BILINEAR)
    
    return alpha_mask

def predict_alpha_fba(image):
    """
    Predict alpha matte using FBA pipeline (U-Net + FBA refinement)
    Args:
        image: PIL Image
    Returns:
        alpha_mask: PIL Image (L mode)
    """
    global fba_model
    if fba_model is None:
        load_fba_model_func()
    
    if fba_model is None:
        raise Exception("FBA pipeline not available")
    
    # Use the FBA pipeline prediction function
    from fba_model import predict_fba_pipeline
    alpha = predict_fba_pipeline(fba_model, image)
    
    # Convert to PIL Image
    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
    
    return alpha_img

def auto_crop(image):
    """Crop image to content bounding box"""
    bbox = image.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

@app.route('/')
def index():
    return render_template('index.html')

import base64

@app.route('/api/cutout', methods=['POST'])
def generate_cutout():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Get selected model (default to modnet)
    selected_model = request.form.get('model', 'modnet')
        
    try:
        # Open image
        img = Image.open(file.stream).convert('RGB')
        
        # Predict alpha based on selected model
        if selected_model == 'fba':
            alpha_mask = predict_alpha_fba(img)
            benchmark_metrics = {
                'benchmark_iou': 0.7500,
                'benchmark_dice': 0.8462,
                'benchmark_mae': 0.0869
            }
        else:  # default to modnet
            alpha_mask = predict_alpha(img)
            benchmark_metrics = {
                'benchmark_iou': 0.7037,
                'benchmark_dice': 0.8174,
                'benchmark_mae': 0.0899
            }
        
        # Calculate dynamic metrics (Confidence)
        # Convert to numpy for calculation
        alpha_np = np.array(alpha_mask) / 255.0
        
        # Confidence: How distinct is the separation? (Mean distance from 0.5)
        # 0.5 is uncertain, 0 or 1 is confident. 
        # Formula: 2 * |0.5 - p| -> ranges from 0 to 1
        confidence_map = 2 * np.abs(0.5 - alpha_np)
        avg_confidence = np.mean(confidence_map)
        
        # Create RGBA
        rgba = img.copy()
        rgba.putalpha(alpha_mask)
        
        # Auto crop
        cutout = auto_crop(rgba)
        
        # Save to buffer
        img_io = io.BytesIO()
        cutout.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Encode Cutout to Base64
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        # Encode Alpha Matte to Base64 for visualization
        alpha_io = io.BytesIO()
        alpha_mask.save(alpha_io, 'PNG')
        alpha_io.seek(0)
        alpha_base64 = base64.b64encode(alpha_io.getvalue()).decode('utf-8')
        
        return jsonify({
            'image': f'data:image/png;base64,{img_base64}',
            'alpha_matte': f'data:image/png;base64,{alpha_base64}',
            'metrics': {
                'confidence': float(avg_confidence),
                **benchmark_metrics
            }
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """Compare MODNet and FBA model outputs"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Open image
        img = Image.open(file.stream).convert('RGB')
        
        # MODNet Prediction
        alpha_modnet = predict_alpha(img)
        alpha_modnet_np = np.array(alpha_modnet) / 255.0
        
        # Calculate MODNet metrics
        confidence_modnet = np.mean(2 * np.abs(0.5 - alpha_modnet_np))
        
        # Create MODNet cutout
        rgba_modnet = img.copy()
        rgba_modnet.putalpha(alpha_modnet)
        cutout_modnet = auto_crop(rgba_modnet)
        
        # Encode MODNet results
        modnet_io = io.BytesIO()
        cutout_modnet.save(modnet_io, 'PNG')
        modnet_io.seek(0)
        modnet_base64 = base64.b64encode(modnet_io.getvalue()).decode('utf-8')
        
        modnet_alpha_io = io.BytesIO()
        alpha_modnet.save(modnet_alpha_io, 'PNG')
        modnet_alpha_io.seek(0)
        modnet_alpha_base64 = base64.b64encode(modnet_alpha_io.getvalue()).decode('utf-8')
        
        # FBA Prediction (if available)
        fba_available = fba_model is not None
        if not fba_available:
            load_fba_model_func()
            fba_available = fba_model is not None
        
        if fba_available:
            try:
                alpha_fba = predict_alpha_fba(img)
                alpha_fba_np = np.array(alpha_fba) / 255.0
                
                # Calculate FBA metrics
                confidence_fba = np.mean(2 * np.abs(0.5 - alpha_fba_np))
                
                # Create FBA cutout
                rgba_fba = img.copy()
                rgba_fba.putalpha(alpha_fba)
                cutout_fba = auto_crop(rgba_fba)
                
                # Encode FBA results
                fba_io = io.BytesIO()
                cutout_fba.save(fba_io, 'PNG')
                fba_io.seek(0)
                fba_base64 = base64.b64encode(fba_io.getvalue()).decode('utf-8')
                
                fba_alpha_io = io.BytesIO()
                alpha_fba.save(fba_alpha_io, 'PNG')
                fba_alpha_io.seek(0)
                fba_alpha_base64 = base64.b64encode(fba_alpha_io.getvalue()).decode('utf-8')
                
                fba_result = {
                    'image': f'data:image/png;base64,{fba_base64}',
                    'alpha_matte': f'data:image/png;base64,{fba_alpha_base64}',
                    'metrics': {
                        'confidence': float(confidence_fba),
                        'benchmark_iou': 0.7654,
                        'benchmark_dice': 0.8561,
                        'benchmark_mae': 0.0774
                    }
                }
            except Exception as e:
                print(f"FBA prediction error: {e}")
                fba_result = None
        else:
            fba_result = None
        
        return jsonify({
            'modnet': {
                'image': f'data:image/png;base64,{modnet_base64}',
                'alpha_matte': f'data:image/png;base64,{modnet_alpha_base64}',
                'metrics': {
                    'confidence': float(confidence_modnet),
                    'benchmark_iou': 0.7237,
                    'benchmark_dice': 0.8290,
                    'benchmark_mae': 0.0807
                }
            },
            'fba': fba_result
        })
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/composite', methods=['POST'])
def generate_composite():
    if 'foreground' not in request.files or 'background' not in request.files:
        return jsonify({'error': 'Both foreground and background images required'}), 400
        
    fg_file = request.files['foreground']
    bg_file = request.files['background']
    
    try:
        # Open images
        fg_img = Image.open(fg_file.stream).convert('RGB')
        bg_img = Image.open(bg_file.stream).convert('RGB') # Keep RGB for background
        
        # Get cutout of foreground
        alpha_mask = predict_alpha(fg_img)
        fg_rgba = fg_img.copy()
        fg_rgba.putalpha(alpha_mask)
        fg_cutout = auto_crop(fg_rgba)
        
        # Resize foreground to fit nicely in background
        # Logic: Fit foreground into 80% of background height or width, whichever is smaller
        bg_w, bg_h = bg_img.size
        fg_w, fg_h = fg_cutout.size
        
        scale_h = (bg_h * 0.8) / fg_h
        scale_w = (bg_w * 0.8) / fg_w
        scale = min(scale_h, scale_w, 1.0) # Don't upscale if it's already small enough
        
        new_w = int(fg_w * scale)
        new_h = int(fg_h * scale)
        
        fg_resized = fg_cutout.resize((new_w, new_h), Image.LANCZOS)
        
        # Center foreground on background
        x_offset = (bg_w - new_w) // 2
        y_offset = (bg_h - new_h) // 2
        
        # Composite
        final_img = bg_img.copy()
        final_img.paste(fg_resized, (x_offset, y_offset), fg_resized)
        
        # Save to buffer
        img_io = io.BytesIO()
        final_img.save(img_io, 'PNG') # Return PNG to preserve quality, though JPG is fine too
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        print(f"Error creating composite: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_modnet_model()
    load_fba_model_func()
    print("Starting Flask server on http://0.0.0.0:4000")
    app.run(host='0.0.0.0', port=4000, debug=True)
