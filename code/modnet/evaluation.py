"""
Evaluation script for Pet Segmentation and Matting Pipeline.
Calculates quantitative metrics (IoU, Dice, MAE) and generates qualitative results.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from pipeline import PetMattingPipeline
from matting_model import MattingModel, matting_loss
from trimap_generation import generate_trimap_adaptive

# Constants
IMG_SIZE = 128
EVAL_SAMPLES = 200  # Number of samples to evaluate
RESULTS_DIR = 'evaluation_results'

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_iou(y_true, y_pred, threshold=0.5):
    """Compute Intersection over Union"""
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = (y_true > threshold).astype(np.float32)
    
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection
    
    if union == 0:
        return 1.0
    return intersection / union

def compute_dice(y_true, y_pred, threshold=0.5):
    """Compute Dice Coefficient"""
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = (y_true > threshold).astype(np.float32)
    
    intersection = np.sum(y_true_bin * y_pred_bin)
    return (2. * intersection) / (np.sum(y_true_bin) + np.sum(y_pred_bin) + 1e-7)

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def main():
    print("="*60)
    print("PET PIPELINE EVALUATION")
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
    
    # 3. Run Evaluation
    print(f"\nEvaluating on {EVAL_SAMPLES} samples...")
    
    ious = []
    dices = []
    maes = []
    
    for i, (image, mask) in enumerate(test.take(EVAL_SAMPLES)):
        image_np = np.expand_dims(image.numpy(), 0)
        mask_np = mask.numpy()
        
        # Run pipeline manually to ensure correct trimap generation
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
        
        # Calculate metrics
        iou = compute_iou(mask_np, alpha_matte_np)
        dice = compute_dice(mask_np, alpha_matte_np)
        mae = compute_mae(mask_np, alpha_matte_np)
        
        ious.append(iou)
        dices.append(dice)
        maes.append(mae)
        
        # Save qualitative results for first 5 images
        if i < 5:
            save_path = os.path.join(RESULTS_DIR, f'eval_result_{i+1}.png')
            pipeline.visualize_results(image_np, alpha_matte.numpy(), trimap_batch, seg_mask, save_path)
            
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{EVAL_SAMPLES} samples...")
            
    # 4. Report Results
    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    mean_mae = np.mean(maes)
    
    print("\n" + "="*60)
    print("QUANTITATIVE RESULTS")
    print("="*60)
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Mean MAE:        {mean_mae:.4f}")
    print("="*60)
    
    # Save metrics to file
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        f.write("PET PIPELINE EVALUATION METRICS\n")
        f.write("===============================\n")
        f.write(f"Samples evaluated: {EVAL_SAMPLES}\n")
        f.write(f"Mean IoU:        {mean_iou:.4f}\n")
        f.write(f"Mean Dice Score: {mean_dice:.4f}\n")
        f.write(f"Mean MAE:        {mean_mae:.4f}\n")
        
    print(f"\nEvaluation complete! Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
