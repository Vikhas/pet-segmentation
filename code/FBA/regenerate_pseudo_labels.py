import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from tqdm import tqdm
from FBAMattingRefiner import FBAMattingRefiner

# --- CONFIGURATION ---
UNET_MODEL_PATH = "../models/fba/unet/pet_unet_improved_best.keras"
FBA_CKPT_PATH = "../models/fba/fba_pet_final.pth"
OUTPUT_DIR = "../code/FBA/pseudo_labels"
IMG_SIZE = 256  # Match U-Net training size

# --- SETUP ---
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "alphas"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "trimaps"), exist_ok=True)

# Select Device for FBA
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"🚀 FBA Refiner using device: {device}")

# Load Models
print(f"Loading Improved U-Net from {UNET_MODEL_PATH}...")
unet_model = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)

print(f"Loading FBA Refiner from {FBA_CKPT_PATH}...")
fba_refiner = FBAMattingRefiner(FBA_CKPT_PATH, device=device)

# Load Dataset
print("Loading Oxford-IIIT Pet dataset (TRAIN split)...")
ds = tfds.load('oxford_iiit_pet:3.2.0', split='train', shuffle_files=False)

def preprocess_for_unet(image):
    # Resize to 256x256 and normalize
    img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    return img

def generate_trimap(alpha):
    # Generate trimap from alpha matte
    # Alpha > 0.95 -> FG (255)
    # Alpha < 0.05 -> BG (0)
    # Else -> Unknown (128)
    trimap = np.full_like(alpha, 128, dtype=np.uint8)
    trimap[alpha > 0.95] = 255
    trimap[alpha < 0.05] = 0
    return trimap

# --- GENERATION LOOP ---
print("Generating Pseudo-Labels...")
count = 0

for sample in tqdm(ds):
    try:
        # Get original image
        image_tf = sample['image']
        original_h, original_w = image_tf.shape[:2]
        file_id = sample['file_name'].numpy().decode('utf-8').split('.')[0]
        
        # 1. U-Net Prediction
        input_img = preprocess_for_unet(image_tf)
        input_batch = tf.expand_dims(input_img, 0) # (1, 256, 256, 3)
        pred_mask = unet_model.predict(input_batch, verbose=0) # (1, 256, 256, 1)
        
        # Resize mask back to original size
        pred_mask_resized = tf.image.resize(pred_mask, (original_h, original_w)).numpy().squeeze()
        
        # 2. FBA Refinement
        # Convert image to numpy (H, W, 3)
        image_np = image_tf.numpy()
        
        # Refine
        refined_alpha, _ = fba_refiner.refine(image_np, pred_mask_resized)
        
        # 3. Save Results
        # Save Image
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", f"{file_id}.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        
        # Save Alpha (0-255)
        alpha_uint8 = (refined_alpha.squeeze() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "alphas", f"{file_id}.png"), alpha_uint8)
        
        # Save Trimap
        trimap = generate_trimap(refined_alpha.squeeze())
        cv2.imwrite(os.path.join(OUTPUT_DIR, "trimaps", f"{file_id}.png"), trimap)
        
        count += 1
        
    except Exception as e:
        print(f"Error processing {file_id}: {e}")
        continue

print(f"✅ Successfully generated {count} pseudo-labels in {OUTPUT_DIR}")
