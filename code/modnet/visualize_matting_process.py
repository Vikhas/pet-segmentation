"""
Create a step-by-step visualization showing how the alpha matte combines with the original image
to produce the final transparent cutout.
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
OUTPUT_DIR = 'matting_process_demo'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_process_visualization(image, alpha_matte, save_path):
    """
    Create a visualization showing the matting process step by step.
    
    Args:
        image: Original image (H, W, 3) in [0, 1]
        alpha_matte: Alpha matte (H, W, 1) in [0, 1]
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Step 1: Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Step 1: Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Step 2: Alpha Matte (Transparency Map)
    axes[0, 1].imshow(alpha_matte[:, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Step 2: Alpha Matte\n(White = Opaque, Black = Transparent)', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Step 3: Alpha Matte as Heatmap
    im = axes[0, 2].imshow(alpha_matte[:, :, 0], cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('Step 3: Alpha Matte (Heatmap)\nShowing Soft Edges', 
                         fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Step 4: Multiplication Process (RGB channels)
    multiplied = image * alpha_matte
    axes[1, 0].imshow(multiplied)
    axes[1, 0].set_title('Step 4: Image × Alpha Matte\n(Foreground Extraction)', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Step 5: Final Transparent Cutout (on checkerboard)
    # Create checkerboard pattern
    checker_size = 16
    h, w = image.shape[:2]
    checkerboard = np.zeros((h, w, 3))
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                checkerboard[i:i+checker_size, j:j+checker_size] = 0.8
            else:
                checkerboard[i:i+checker_size, j:j+checker_size] = 0.6
    
    # Composite on checkerboard
    composite = checkerboard * (1 - alpha_matte) + multiplied
    axes[1, 1].imshow(composite)
    axes[1, 1].set_title('Step 5: Final Transparent Cutout\n(On Checkerboard Pattern)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Step 6: Side-by-side comparison
    comparison = np.concatenate([image, composite], axis=1)
    axes[1, 2].imshow(comparison)
    axes[1, 2].set_title('Step 6: Before → After\n(Original | Transparent Cutout)', 
                         fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Pet Matting Process: From Image to Transparent Cutout', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved process visualization to {save_path}")

def main():
    print("="*60)
    print("CREATING MATTING PROCESS DEMONSTRATION")
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
    
    # 3. Generate Process Visualizations
    print("\nGenerating process demonstrations for 5 samples...")
    
    for i, (image, mask) in enumerate(test.take(5)):
        image_np = image.numpy()
        image_batch = np.expand_dims(image_np, 0)
        
        # Run pipeline
        seg_mask = pipeline.generate_segmentation_masks(image_batch)
        trimap = generate_trimap_adaptive(seg_mask[0, :, :, 0], uncertainty_region=10)
        trimap = np.expand_dims(trimap, -1)
        trimap_batch = np.expand_dims(trimap, 0)
        
        matting_input = tf.concat([image_batch, trimap_batch/255.0], axis=-1)
        alpha_matte = pipeline.matting_model(matting_input, training=False)
        alpha_matte_np = alpha_matte.numpy()[0]
        
        # Create visualization
        save_path = os.path.join(OUTPUT_DIR, f'matting_process_{i+1}.png')
        create_process_visualization(image_np, alpha_matte_np, save_path)
        
        # Also save the actual transparent PNG
        rgba = np.concatenate([image_np, alpha_matte_np], axis=-1)
        rgba_uint8 = (rgba * 255).astype(np.uint8)
        img = Image.fromarray(rgba_uint8, 'RGBA')
        png_path = os.path.join(OUTPUT_DIR, f'transparent_cutout_{i+1}.png')
        img.save(png_path)
        print(f"Saved transparent PNG to {png_path}")
    
    print("\n" + "="*60)
    print(f"DONE! Check the '{OUTPUT_DIR}' folder for results.")
    print("="*60)
    print("\nThe visualizations show:")
    print("  1. Original image")
    print("  2. Alpha matte (transparency map)")
    print("  3. Alpha matte as heatmap (showing soft edges)")
    print("  4. Image × Alpha (foreground extraction)")
    print("  5. Final transparent cutout (on checkerboard)")
    print("  6. Before/After comparison")

if __name__ == "__main__":
    main()
