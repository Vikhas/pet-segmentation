"""
Demo script to generate sample pet cutouts using the trained MODNet model
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
import sys

# Import custom classes
sys.path.insert(0, os.path.dirname(__file__))
from modnet_architecture import MODNet

def generate_demo_cutouts():
    """Generate demo cutouts from test dataset"""
    
    print("=" * 70)
    print("Generating Sample Pet Cutouts with MODNet")
    print("=" * 70)
    
    # Load model
    print("\nLoading MODNet model...")
    model = tf.keras.models.load_model(
        '../models/modnet_pet_matting.keras',
        custom_objects={'MODNet': MODNet},
        compile=False
    )
    print("Model loaded")
    
    # Load test images
    print("\n Loading test images...")
    _, test_ds = tfds.load('oxford_iiit_pet:3.2.0', split=['train', 'test'])
    
    # Create output directory
    output_dir = '../outputs/sample_cutouts'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate cutouts for 10 samples
    print(f"\n Generating cutouts...")
    
    for i, data in enumerate(test_ds.take(10)):
        image = data['image'].numpy()
        
        # Resize for model
        img_resized = tf.image.resize(image, (128, 128)) / 255.0
        img_tensor = tf.expand_dims(img_resized, 0)
        
        # Predict alpha
        alpha_pred = model.predict(img_tensor, verbose=0)
        
        # Invert alpha (model trained with bg=1, fg=0)
        alpha_inverted = 1.0 - alpha_pred[0]
        
        # Resize alpha to original size
        original_h, original_w = image.shape[:2]
        alpha_resized = tf.image.resize(
            alpha_inverted,
            (original_h, original_w),
            method='bilinear'
        ).numpy()
        
        # Create RGBA cutout
        rgba = np.dstack([image, (alpha_resized[:, :, 0] * 255).astype(np.uint8)])
        cutout = Image.fromarray(rgba, 'RGBA')
        
        # Auto-crop
        bbox = cutout.getbbox()
        if bbox:
            cutout = cutout.crop(bbox)
        
        # Save
        cutout_path = os.path.join(output_dir, f'pet_cutout_{i+1:02d}.png')
        cutout.save(cutout_path)
        
        print(f"  [{i+1}/10] Saved {cutout_path}")
    
    print(f"\nAll cutouts saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    generate_demo_cutouts()
