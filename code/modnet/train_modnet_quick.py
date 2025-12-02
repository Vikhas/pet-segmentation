"""
Quick training script for MODNet - Reduced epochs for faster completion
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from modnet_architecture import create_modnet
from modnet_data_utils import prepare_modnet_training_data, create_training_dataset

# Reduced configuration for faster training
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 20  # Reduced from 40
LEARNING_RATE = 1e-4
NUM_TRAIN_SAMPLES = 1500  # Reduced from 2000
NUM_TEST_SAMPLES = 150    # Reduced from 200

# Directories
CHECKPOINT_DIR = '../models/modnet_checkpoints'
VIS_DIR = '../outputs/modnet_training_vis'
LOG_DIR = '../outputs/modnet_logs'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class QuickVisCallback(tf.keras.callbacks.Callback):
    """Lightweight visualization callback"""
    
    def __init__(self, val_images, val_masks, save_dir):
        super().__init__()
        self.val_images = val_images[:4]
        self.val_masks = val_masks[:4]
        self.save_dir = save_dir
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 != 0:  # Every 5 epochs
            return
        
        preds = self.model.modnet(self.val_images, training=False)
        
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        for i in range(4):
            axes[i, 0].imshow(self.val_images[i])
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(self.val_masks[i, :, :, 0], cmap='gray')
            axes[i, 1].set_title('GT')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(preds[i, :, :, 0], cmap='gray')
            axes[i, 2].set_title(f'Pred (μ={preds[i].numpy().mean():.3f})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch+1:03d}.png'), dpi=100)
        plt.close()
        print(f"\nSaved visualization")


def main():
    print("=" * 70)
    print("MODNet Quick Training (20 epochs)")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    train_ds, test_ds = tfds.load('oxford_iiit_pet:3.2.0', split=['train', 'test'])
    
    # Prepare data
    print(f"\nPreparing data...")
    X_train, y_train, X_test, y_test = prepare_modnet_training_data(
        train_ds, test_ds,
        img_size=IMG_SIZE,
        num_train=NUM_TRAIN_SAMPLES,
        num_test=NUM_TEST_SAMPLES
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_training_dataset(X_train, y_train, BATCH_SIZE, True, True)
    val_dataset = create_training_dataset(X_test, y_test, BATCH_SIZE, False, False)
    
    # Create model
    print("\n Creating model...")
    model = create_modnet((IMG_SIZE, IMG_SIZE, 3), LEARNING_RATE)
    dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    _ = model.modnet(dummy, training=False)
    print(f"Model: {model.modnet.count_params():,} parameters")
    
    # Callbacks
    callbacks = [
        QuickVisCallback(X_test[:10], y_test[:10], VIS_DIR),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, 'modnet_best.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    print("=" * 70)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    model.modnet.save('../models/modnet_pet_matting.keras')
    print(f"\nModel saved!")
    
    # Plot history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Val')
    plt.title('MAE')
    plt.legend()
    
    plt.savefig(os.path.join(VIS_DIR, 'training_history.png'), dpi=150)
    print("History saved!")
    
    # Final eval
    print("\nFinal Evaluation:")
    final_loss, final_mae = model.evaluate(val_dataset, verbose=0)
    print(f"Val Loss: {final_loss:.4f}")
    print(f"Val MAE: {final_mae:.4f}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    main()
