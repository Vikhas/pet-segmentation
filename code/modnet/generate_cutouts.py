"""
Script to generate high-quality transparent cutouts (RGBA PNGs).
Combines the original image with the alpha matte to create a transparent background.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from pipeline import PetMattingPipeline
from matting_model import MattingModel, matting_loss
from trimap_generation import generate_trimap_adaptive

# Constants
IMG_SIZE = 128
CUTOUTS_DIR = 'final_cutouts'

# Ensure output directory exists
os.makedirs(CUTOUTS_DIR, exist_ok=True)

def save_transparent_png(image, alpha, save_path):
    """
    Save image with alpha channel as PNG.
    image: (H, W, 3) float32 [0, 1]
    alpha: (H, W, 1) float32 [0, 1]
    """
    # Combine image and alpha
    rgba = np.concatenate([image, alpha], axis=-1)
    
    # Convert to uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    
    # Save using PIL
    img = Image.fromarray(rgba_uint8, 'RGBA')
    img.save(save_path)
    print(f"Saved transparent cutout to {save_path}")

def main():
    print("="*60)
    print("GENERATING TRANSPARENT CUTOUTS")
    print("="*60)
    
    # 1. Load Data
    print("\nLoading Oxford-IIIT Pet dataset...")
    test_ds = tfds.load('oxford_iiit_pet:3.2.0', split='test', with_info=False)
    
    def preprocess(data):
        image = data['image']
        mask = data['segmentation_mask']
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method='nearest')
        mask = tf.where(mask > 1, 1.0, 0.0)
        return image, mask
    
    test = test_ds.map(preprocess)
    
    # 2. Initialize Pipeline
    print("Initializing pipeline...")
    # Model paths (relative to project root)
    segmentation_model_path = '../models/fba/unet/pet_unet_improved_final.keras'
    pipeline = PetMattingPipeline(segmentation_model_path, img_size=IMG_SIZE)
    
    # Matting model path
    matting_model_path = '../models/modnet/modnet_pet_matting.keras'
    pipeline.matting_model = tf.keras.models.load_model(
        matting_model_path,
        custom_objects={'MattingModel': MattingModel, 'matting_loss': matting_loss}
    )
    print("Models loaded successfully")
    
    # 3. Generate Cutouts
    print("\nGenerating cutouts for 10 samples...")
    
    for i, (image, mask) in enumerate(test.take(10)):
        image_np = np.expand_dims(image.numpy(), 0)
        
        # Run pipeline
        # 1. Segmentation
        seg_mask = pipeline.generate_segmentation_masks(image_np)
        
        # 2. Trimap
        trimap = generate_trimap_adaptive(seg_mask[0, :, :, 0], uncertainty_region=10)
        trimap = np.expand_dims(trimap, -1)
        trimap_batch = np.expand_dims(trimap, 0)
        
        # 3. Matting
        matting_input = tf.concat([image_np, trimap_batch/255.0], axis=-1)
        alpha_matte = pipeline.matting_model(matting_input, training=False)
        alpha_matte_np = alpha_matte.numpy()[0]
        
        # 4. Save Transparent PNG
        save_path = os.path.join(CUTOUTS_DIR, f'cutout_{i+1}.png')
        save_transparent_png(image.numpy(), alpha_matte_np, save_path)
        
        # Also save original for comparison
        orig_path = os.path.join(CUTOUTS_DIR, f'original_{i+1}.jpg')
        tf.keras.preprocessing.image.save_img(orig_path, image.numpy())

    print("\n" + "="*60)
    print(f"DONE! Check the '{CUTOUTS_DIR}' folder for results.")
    print("="*60)

if __name__ == "__main__":
    main()
