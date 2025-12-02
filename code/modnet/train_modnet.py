"""
Training script for MODNet on pet dataset
Fine-tunes MODNet for high-quality pet fur/hair matting
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from modnet_architecture import create_modnet, MODNet
from modnet_data_utils import prepare_modnet_training_data, create_training_dataset

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_TRAIN_SAMPLES = 2000
NUM_TEST_SAMPLES = 200

# Directories
CHECKPOINT_DIR = '../models/modnet_checkpoints'
VIS_DIR = '../outputs/modnet_training_vis'
LOG_DIR = '../outputs/modnet_logs'

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class MODNetVisualizationCallback(tf.keras.callbacks.Callback):
    """Callback to visualize predictions during training"""
    
    def __init__(self, val_images, val_masks, save_dir, interval=2):
        super().__init__()
        self.val_images = val_images
        self.val_masks = val_masks
        self.save_dir = save_dir
        self.interval = interval
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return
        
        # Select random samples
        indices = np.random.choice(len(self.val_images), 4, replace=False)
        sample_images = self.val_images[indices]
        sample_masks = self.val_masks[indices]
        
        # Get predictions (inference mode)
        preds = self.model.modnet(sample_images, training=False)
        
        # Visualize
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        for i in range(4):
            # Original image
            axes[i, 0].imshow(sample_images[i])
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(sample_masks[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            pred_alpha = preds[i, :, :, 0].numpy()
            axes[i, 2].imshow(pred_alpha, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Prediction (μ={pred_alpha.mean():.3f})')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Epoch {epoch + 1} - Val Loss: {logs.get("val_loss", 0):.4f}, Val MAE: {logs.get("val_mae", 0):.4f}')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'epoch_{epoch+1:03d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved visualization to {save_path}")


class DetailedLoggingCallback(tf.keras.callbacks.Callback):
    """Callback for detailed logging"""
    
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {self.start_time}\n")
            f.write("=" * 60 + "\n")
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = datetime.now() - self.start_time
        
        log_msg = (
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {logs.get('loss', 0):.4f} | "
            f"MAE: {logs.get('mae', 0):.4f} | "
            f"Val Loss: {logs.get('val_loss', 0):.4f} | "
            f"Val MAE: {logs.get('val_mae', 0):.4f} | "
            f"Time: {elapsed}\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg)
        
        print(log_msg.strip())


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training history to {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print(" " * 20 + "MODNet Training for Pet Matting")
    print("=" * 70)
    
    # 1. Load dataset
    print("\nLoading Oxford-IIIT Pet Dataset...")
    train_ds, test_ds = tfds.load(
        'oxford_iiit_pet:3.2.0',
        split=['train', 'test'],
        with_info=False
    )
    print("Dataset loaded")
    
    # 2. Prepare data
    print(f"\nPreparing data (Train: {NUM_TRAIN_SAMPLES}, Test: {NUM_TEST_SAMPLES})...")
    X_train, y_train, X_test, y_test = prepare_modnet_training_data(
        train_ds, test_ds,
        img_size=IMG_SIZE,
        num_train=NUM_TRAIN_SAMPLES,
        num_test=NUM_TEST_SAMPLES
    )
    
    # 3. Create datasets with augmentation
    print("\nCreating training dataset with augmentation...")
    train_dataset = create_training_dataset(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True
    )
    
    val_dataset = create_training_dataset(
        X_test, y_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    # 4. Create model
    print("\n Creating MODNet model...")
    model = create_modnet(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        learning_rate=LEARNING_RATE
    )
    
    # Build model to count parameters
    dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
    _ = model.modnet(dummy_input, training=False)
    
    print(f"Model created with {model.modnet.count_params():,} parameters")
    
    # 5. Setup callbacks
    callbacks = [
        MODNetVisualizationCallback(
            X_test[:20], y_test[:20],
            VIS_DIR,
            interval=2
        ),
        DetailedLoggingCallback(
            os.path.join(LOG_DIR, 'training_log.txt')
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, 'modnet_best.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1
        )
    ]
    
    # 6. Train
    print("\n" + "=" * 70)
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Training samples: {NUM_TRAIN_SAMPLES}")
    print(f"   Validation samples: {NUM_TEST_SAMPLES}")
    print("=" * 70 + "\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Save final model
    final_model_path = '../models/modnet_pet_matting.keras'
    model.modnet.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # 8. Plot training history
    history_plot_path = os.path.join(VIS_DIR, 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    # 9. Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    final_loss, final_mae = model.evaluate(val_dataset, verbose=0)
    print(f"Final Validation Loss: {final_loss:.4f}")
    print(f"Final Validation MAE: {final_mae:.4f}")
    
    # 10. Generate final predictions
    print("\nGenerating final prediction samples...")
    sample_indices = np.random.choice(len(X_test), 8, replace=False)
    sample_images = X_test[sample_indices]
    sample_masks = y_test[sample_indices]
    
    predictions = model.modnet(sample_images, training=False)
    
    fig, axes = plt.subplots(8, 3, figsize=(12, 32))
    for i in range(8):
        axes[i, 0].imshow(sample_images[i])
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sample_masks[i, :, :, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(predictions[i, :, :, 0], cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    final_preds_path = os.path.join(VIS_DIR, 'final_predictions.png')
    plt.savefig(final_preds_path, dpi=150, bbox_inches='tight')
    print(f"Saved final predictions to {final_preds_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print(f"\n📁 Outputs saved to:")
    print(f"   Model: {final_model_path}")
    print(f"   Visualizations: {VIS_DIR}")
    print(f"   Logs: {LOG_DIR}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Warning:  GPU configuration error: {e}")
    
    main()
