"""
Simple script to generate transparent pet cutouts using trained MODNet
Removes all background, keeping only the pet with transparency
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys


import sys
sys.path.insert(0, os.path.dirname(__file__))
from modnet_architecture import MODNet

def generate_pet_cutout(image_path, model_path='../models/modnet_pet_matting.keras', output_path=None):
    """
    Generate transparent pet cutout from image
    
    Args:
        image_path: Path to input pet image
        model_path: Path to trained MODNet model
        output_path: Path to save cutout (optional, auto-generated if None)
    
    Returns:
        Path to saved cutout
    """
    print(f"🐾 Processing: {image_path}")
    
    # Load model
    print("Loading MODNet model...")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'MODNet': MODNet},
            compile=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load and preprocess image
    print(" Loading image...")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    print(f"   Original size: {original_size[0]}x{original_size[1]}")
    
    # Resize for model (128x128)
    img_resized = img.resize((128, 128), Image.BILINEAR)
    img_array = np.array(img_resized) / 255.0
    img_tensor = np.expand_dims(img_array, 0).astype(np.float32)
    
    # Predict alpha matte
    print("🎯 Predicting alpha matte...")
    alpha_pred = model.predict(img_tensor, verbose=0)
    
    # Postprocess alpha to original size
    # NOTE: Model was trained with inverted masks (bg=1, fg=0), so we invert here
    alpha = 1.0 - alpha_pred[0, :, :, 0]
    
    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
    alpha_resized = alpha_img.resize(original_size, Image.BILINEAR)
    alpha_array = np.array(alpha_resized)
    
    # Create transparent cutout (RGBA)
    print(" Creating transparent cutout...")
    img_array = np.array(img)
    rgba = np.dstack([img_array, alpha_array])
    cutout = Image.fromarray(rgba, 'RGBA')
    
    # Auto-crop to pet content
    print(" Auto-cropping to pet content...")
    # Get bounding box of non-transparent pixels
    bbox = cutout.getbbox()
    if bbox:
        cutout = cutout.crop(bbox)
        print(f"   Cropped from {original_size} to {cutout.size}")
    else:
        print("   Warning: Warning: No pet detected (image is fully transparent)")
    
    # Save
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_cutout.png"
    
    cutout.save(output_path)
    print(f"Saved transparent cutout to: {output_path}")
    
    # Print stats
    # Check center pixel for transparency sanity check
    center_alpha = alpha_array[original_size[1]//2, original_size[0]//2]
    print(f"Center pixel alpha: {center_alpha:.2f} (Should be high for pet)")
    
    return output_path


def batch_generate_cutouts(input_dir, output_dir='cutouts', model_path='../models/modnet_pet_matting.keras'):
    """
    Generate cutouts for all images in a directory
    
    Args:
        input_dir: Directory with input images
        output_dir: Directory to save cutouts
        model_path: Path to trained MODNet model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n{'='*70}")
    print(f"Processing {len(image_files)} images from {input_dir}")
    print(f"{'='*70}\n")
    
    # Load model once
    print("Loading MODNet model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    for i, filename in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {filename}")
        
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_cutout.png")
        
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            original_size = img.size
            
            # Resize for model
            img_resized = img.resize((128, 128), Image.BILINEAR)
            img_array = np.array(img_resized) / 255.0
            img_tensor = np.expand_dims(img_array, 0).astype(np.float32)
            
            # Predict
            alpha_pred = model.predict(img_tensor, verbose=0)
            
            # Postprocess
            alpha = alpha_pred[0, :, :, 0]
            alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
            alpha_resized = alpha_img.resize(original_size, Image.BILINEAR)
            alpha_array = np.array(alpha_resized)
            
            # Create cutout
            img_array = np.array(img)
            rgba = np.dstack([img_array, alpha_array])
            cutout = Image.fromarray(rgba, 'RGBA')
            cutout.save(output_path)
            
            print(f"   Saved to {output_path}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"All cutouts saved to {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate transparent pet cutouts using MODNet')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-m', '--model', default='../models/modnet_pet_matting.keras',
                       help='Path to trained MODNet model')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Single image
        generate_pet_cutout(args.input, args.model, args.output)
    elif os.path.isdir(args.input):
        # Directory
        output_dir = args.output or 'cutouts'
        batch_generate_cutouts(args.input, output_dir, args.model)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)
