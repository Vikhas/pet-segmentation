"""
Model comparison script
Compares MODNet vs simple matting model on pet images
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Import custom classes for model loading
import sys
sys.path.insert(0, os.path.dirname(__file__))
from matting_model import MattingModel, matting_loss
from modnet_architecture import MODNet

# Configuration
IMG_SIZE = 128
NUM_SAMPLES = 10


def load_models():
    """Load both models for comparison"""
    print("Loading models...")
    
    # Load simple matting model
    simple_model_path = '../models/pet_matting_model_improved.keras'
    if os.path.exists(simple_model_path):
        try:
            simple_model = tf.keras.models.load_model(
                simple_model_path,
                custom_objects={'MattingModel': MattingModel, 'matting_loss': matting_loss},
                compile=False
            )
            print(f"Loaded simple model from {simple_model_path}")
        except Exception as e:
            print(f"Warning:  Could not load simple model: {e}")
            simple_model = None
    else:
        print(f"Warning:  Simple model not found at {simple_model_path}")
        simple_model = None
    
    # Load MODNet model
    modnet_path = '../models/modnet_pet_matting.keras'
    if os.path.exists(modnet_path):
        try:
            modnet = tf.keras.models.load_model(
                modnet_path,
                custom_objects={'MODNet': MODNet},
                compile=False
            )
            print(f"Loaded MODNet from {modnet_path}")
        except Exception as e:
            print(f"Warning:  Could not load MODNet: {e}")
            modnet = None
    else:
        print(f"Warning:  MODNet not found at {modnet_path}")
        modnet = None
    
    return simple_model, modnet




def prepare_test_data(num_samples=10):
    """Prepare test images and masks"""
    print(f"\nPreparing {num_samples} test samples...")
    
    _, test_ds = tfds.load(
        'oxford_iiit_pet:3.2.0',
        split=['train', 'test'],
        with_info=False
    )
    
    images = []
    masks = []
    
    for data in test_ds.take(num_samples):
        image = data['image']
        mask = data['segmentation_mask']
        
        # Resize
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method='nearest')
        
        # Normalize
        image = image / 255.0
        mask = tf.where(mask > 1, 1.0, 0.0)
        mask = tf.expand_dims(mask, -1) if len(mask.shape) == 2 else mask
        
        images.append(image.numpy())
        masks.append(mask.numpy())
    
    return np.array(images), np.array(masks)


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # IoU
    intersection = np.sum((y_true > 0.5) & (y_pred > 0.5))
    union = np.sum((y_true > 0.5) | (y_pred > 0.5))
    iou = intersection / (union + 1e-7)
    
    # Dice
    dice = 2 * intersection / (np.sum(y_true > 0.5) + np.sum(y_pred > 0.5) + 1e-7)
    
    return {'mae': mae, 'iou': iou, 'dice': dice}


def compare_models(simple_model, modnet, images, masks, save_dir='../outputs/model_comparison'):
    """Compare both models visually and quantitatively"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    
    # Get predictions
    if simple_model is not None:
        print("\nGenerating simple model predictions...")
        # Simple model expects concatenated input (image + trimap)
        # For fair comparison, we'll use ground truth masks as trimaps
        simple_inputs = np.concatenate([images, masks], axis=-1)
        simple_preds = simple_model.predict(simple_inputs, verbose=0)
    else:
        simple_preds = np.zeros_like(masks)
    
    if modnet is not None:
        print("Generating MODNet predictions...")
        modnet_preds = modnet.predict(images, verbose=0)
    else:
        modnet_preds = np.zeros_like(masks)
    
    # Calculate metrics
    print("\n" + "-" * 70)
    print("Quantitative Metrics")
    print("-" * 70)
    
    simple_metrics = {'mae': [], 'iou': [], 'dice': []}
    modnet_metrics = {'mae': [], 'iou': [], 'dice': []}
    
    for i in range(len(images)):
        if simple_model is not None:
            s_metrics = calculate_metrics(masks[i], simple_preds[i])
            for k in simple_metrics:
                simple_metrics[k].append(s_metrics[k])
        
        if modnet is not None:
            m_metrics = calculate_metrics(masks[i], modnet_preds[i])
            for k in modnet_metrics:
                modnet_metrics[k].append(m_metrics[k])
    
    # Print average metrics
    if simple_model is not None:
        print("\nSimple Model:")
        print(f"  MAE:  {np.mean(simple_metrics['mae']):.4f} ± {np.std(simple_metrics['mae']):.4f}")
        print(f"  IoU:  {np.mean(simple_metrics['iou']):.4f} ± {np.std(simple_metrics['iou']):.4f}")
        print(f"  Dice: {np.mean(simple_metrics['dice']):.4f} ± {np.std(simple_metrics['dice']):.4f}")
    
    if modnet is not None:
        print("\nMODNet:")
        print(f"  MAE:  {np.mean(modnet_metrics['mae']):.4f} ± {np.std(modnet_metrics['mae']):.4f}")
        print(f"  IoU:  {np.mean(modnet_metrics['iou']):.4f} ± {np.std(modnet_metrics['iou']):.4f}")
        print(f"  Dice: {np.mean(modnet_metrics['dice']):.4f} ± {np.std(modnet_metrics['dice']):.4f}")
    
    # Visual comparison
    print("\n" + "-" * 70)
    print("Generating visual comparisons...")
    print("-" * 70)
    
    num_display = min(8, len(images))
    fig, axes = plt.subplots(num_display, 5, figsize=(20, 4 * num_display))
    
    for i in range(num_display):
        # Input image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Input Image', fontsize=10)
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(masks[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        
        # Simple model prediction
        if simple_model is not None:
            axes[i, 2].imshow(simple_preds[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            s_mae = calculate_metrics(masks[i], simple_preds[i])['mae']
            axes[i, 2].set_title(f'Simple (MAE: {s_mae:.3f})', fontsize=10)
        else:
            axes[i, 2].text(0.5, 0.5, 'Not Available', ha='center', va='center')
            axes[i, 2].set_title('Simple Model', fontsize=10)
        axes[i, 2].axis('off')
        
        # MODNet prediction
        if modnet is not None:
            axes[i, 3].imshow(modnet_preds[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            m_mae = calculate_metrics(masks[i], modnet_preds[i])['mae']
            axes[i, 3].set_title(f'MODNet (MAE: {m_mae:.3f})', fontsize=10)
        else:
            axes[i, 3].text(0.5, 0.5, 'Not Available', ha='center', va='center')
            axes[i, 3].set_title('MODNet', fontsize=10)
        axes[i, 3].axis('off')
        
        # Difference map (if both models available)
        if simple_model is not None and modnet is not None:
            diff = np.abs(simple_preds[i, :, :, 0] - modnet_preds[i, :, :, 0])
            axes[i, 4].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[i, 4].set_title(f'Difference (max: {diff.max():.3f})', fontsize=10)
        else:
            axes[i, 4].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[i, 4].set_title('Difference', fontsize=10)
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {comparison_path}")
    plt.close()
    
    # Create cutout comparison
    print("\nGenerating cutout comparisons...")
    fig, axes = plt.subplots(num_display, 3, figsize=(12, 4 * num_display))
    
    for i in range(num_display):
        # Original
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Original', fontsize=10)
        axes[i, 0].axis('off')
        
        # Simple model cutout
        if simple_model is not None:
            simple_cutout = np.concatenate([
                images[i],
                simple_preds[i]
            ], axis=-1)
            axes[i, 1].imshow(simple_cutout)
            axes[i, 1].set_title('Simple Model Cutout', fontsize=10)
        else:
            axes[i, 1].text(0.5, 0.5, 'Not Available', ha='center', va='center')
            axes[i, 1].set_title('Simple Model', fontsize=10)
        axes[i, 1].axis('off')
        
        # MODNet cutout
        if modnet is not None:
            modnet_cutout = np.concatenate([
                images[i],
                modnet_preds[i]
            ], axis=-1)
            axes[i, 2].imshow(modnet_cutout)
            axes[i, 2].set_title('MODNet Cutout', fontsize=10)
        else:
            axes[i, 2].text(0.5, 0.5, 'Not Available', ha='center', va='center')
            axes[i, 2].set_title('MODNet', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    cutout_path = os.path.join(save_dir, 'cutout_comparison.png')
    plt.savefig(cutout_path, dpi=150, bbox_inches='tight')
    print(f"Saved cutout comparison to {cutout_path}")
    plt.close()


def main():
    print("=" * 70)
    print(" " * 20 + "MODNet vs Simple Model Comparison")
    print("=" * 70)
    
    # Load models
    simple_model, modnet = load_models()
    
    if simple_model is None and modnet is None:
        print("\nNo models found to compare!")
        print("   Please train at least one model first.")
        return
    
    # Prepare test data
    images, masks = prepare_test_data(num_samples=NUM_SAMPLES)
    
    # Compare models
    compare_models(simple_model, modnet, images, masks)
    
    print("\n" + "=" * 70)
    print("Comparison completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
