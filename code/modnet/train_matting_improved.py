"""
Improved training script for Matting Model with visualization and monitoring.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from matting_model import create_matting_model, prepare_matting_data, matting_loss
from trimap_generation import generate_trimap_adaptive

# Constants
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
CHECKPOINT_DIR = 'checkpoints'
VIS_DIR = 'training_vis'

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

class MattingVisCallback(tf.keras.callbacks.Callback):
    """Callback to visualize predictions during training"""
    
    def __init__(self, val_data, save_dir, interval=1):
        super().__init__()
        self.val_data = val_data  # (images, trimaps, masks)
        self.save_dir = save_dir
        self.interval = interval
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            images, trimaps, masks = self.val_data
            
            # Select a few samples
            indices = np.random.choice(len(images), 3, replace=False)
            sample_images = images[indices]
            sample_trimaps = trimaps[indices]
            sample_masks = masks[indices]
            
            # Prepare input
            # trimaps are already normalized in prepare_matting_data if we used that, 
            # but here we might need to check. 
            # Let's assume input to model expects normalized trimaps.
            
            # Construct model input
            model_input = tf.concat([sample_images, sample_trimaps], axis=-1)
            
            # Predict
            preds = self.model.predict(model_input, verbose=0)
            
            # Visualize
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            
            for i in range(3):
                # Image
                axes[i, 0].imshow(sample_images[i])
                axes[i, 0].set_title(f'Image {i}')
                axes[i, 0].axis('off')
                
                # Trimap
                axes[i, 1].imshow(sample_trimaps[i, :, :, 0], cmap='gray')
                axes[i, 1].set_title('Trimap')
                axes[i, 1].axis('off')
                
                # GT Mask
                axes[i, 2].imshow(sample_masks[i, :, :, 0], cmap='gray')
                axes[i, 2].set_title('GT Mask')
                axes[i, 2].axis('off')
                
                # Pred Mask
                axes[i, 3].imshow(preds[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
                axes[i, 3].set_title(f'Pred (Mean: {preds[i].mean():.4f})')
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f'epoch_{epoch+1}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"\nSaved visualization to {save_path}")

def load_and_preprocess_data():
    print("Loading dataset...")
    train_ds, test_ds = tfds.load('oxford_iiit_pet:3.2.0', split=['train', 'test'], with_info=False)
    
    def preprocess(data):
        image = data['image']
        mask = data['segmentation_mask']
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method='nearest')
        mask = tf.where(mask > 1, 1.0, 0.0)
        return image, mask
    
    train = train_ds.map(preprocess)
    test = test_ds.map(preprocess)
    
    return train, test

def generate_training_data(ds, unet_model=None, num_samples=1000):
    """
    Generate (image, trimap, alpha) tuples for training.
    If unet_model is provided, use it to generate trimaps.
    Otherwise, use GT masks to generate trimaps (simulating perfect segmentation).
    """
    print(f"Generating {num_samples} training samples...")
    
    images_list = []
    trimaps_list = []
    masks_list = []
    
    for image, mask in ds.take(num_samples):
        image_np = image.numpy()
        mask_np = mask.numpy()
        
        # Generate trimap
        # Option A: Use U-Net (more realistic for inference)
        if unet_model:
            img_batch = np.expand_dims(image_np, 0)
            pred_mask = unet_model.predict(img_batch, verbose=0)
            binary_mask = (pred_mask > 0.5).astype(np.float32)
            
            # Generate trimap from predicted mask
            trimap = generate_trimap_adaptive(binary_mask[0, :, :, 0], uncertainty_region=10)
            trimap = np.expand_dims(trimap, -1) # (128, 128, 1)
            
        # Option B: Use GT mask (better for initial training stability)
        else:
            # Generate trimap from GT mask
            # mask_np is (128, 128, 1), squeeze to (128, 128)
            trimap = generate_trimap_adaptive(mask_np.squeeze(), uncertainty_region=10)
            trimap = np.expand_dims(trimap, -1)
            
        images_list.append(image_np)
        trimaps_list.append(trimap / 255.0) # Normalize trimap to [0, 1]
        masks_list.append(mask_np)
        
    return np.array(images_list), np.array(trimaps_list), np.array(masks_list)

def main():
    print("="*50)
    print("IMPROVED MATTING MODEL TRAINING")
    print("="*50)
    
    # 1. Load Data
    train_ds, test_ds = load_and_preprocess_data()
    
    # 2. Load U-Net (Optional, for realistic trimaps)
    # For now, let's use GT masks for trimaps to ensure the matting model 
    # learns the task of "refining trimaps" without noise from U-Net errors first.
    # This is a curriculum learning approach: first learn to matte from good trimaps.
    print("Using Ground Truth masks to generate trimaps for training stability.")
    unet_model = None 
    # unet_model = tf.keras.models.load_model('pet_unet_final_10.keras')
    
    # 3. Prepare Data
    print("Preparing training data...")
    X_train_img, X_train_tri, y_train = generate_training_data(train_ds, unet_model, num_samples=800)
    
    print("Preparing validation data...")
    X_val_img, X_val_tri, y_val = generate_training_data(test_ds, unet_model, num_samples=100)
    
    # Verify data stats
    print("\nData Statistics:")
    print(f"Train Images: {X_train_img.shape}, Range: [{X_train_img.min():.3f}, {X_train_img.max():.3f}]")
    print(f"Train Trimaps: {X_train_tri.shape}, Range: [{X_train_tri.min():.3f}, {X_train_tri.max():.3f}]")
    print(f"Train Masks: {y_train.shape}, Range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Mask Mean: {y_train.mean():.4f} (Should be > 0)")
    
    # Prepare inputs
    X_train = tf.concat([X_train_img, X_train_tri], axis=-1)
    X_val = tf.concat([X_val_img, X_val_tri], axis=-1)
    
    # 4. Create Model
    model = create_matting_model(img_size=IMG_SIZE)
    
    # 5. Train
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    callbacks = [
        MattingVisCallback((X_val_img, X_val_tri, y_val), VIS_DIR, interval=1),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, 'matting_best.keras'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # 6. Save Final Model
    model.save('pet_matting_model_improved.keras')
    print("\nModel saved to pet_matting_model_improved.keras")
    
    # 7. Plot History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE')
    plt.legend()
    
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")

if __name__ == "__main__":
    main()
