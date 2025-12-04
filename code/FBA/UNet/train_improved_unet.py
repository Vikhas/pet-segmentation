"""
Improved U-Net Training Script for Pet Segmentation
- Higher resolution (256x256)
- Data augmentation
- Combined Dice + BCE loss
- Better architecture with BatchNorm
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

print("✅ TensorFlow version:", tf.__version__)

# --- CONFIGURATION ---
IMG_SIZE = 256  # Increased from 128
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4

# --- DATA AUGMENTATION ---
def augment(image, mask):
    """Apply random augmentations to image and mask"""
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Random brightness (image only)
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast (image only)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def preprocess(data, augment_data=False):
    """Preprocess image and mask"""
    image = tf.cast(data['image'], tf.float32) / 255.0
    mask = tf.cast(data['segmentation_mask'], tf.float32)
    
    # Resize
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method='nearest')
    
    # Convert labels: 1=background, 2=pet, 3=border → binary mask
    mask = tf.where(mask > 1, 1.0, 0.0)
    
    # Apply augmentation if training
    if augment_data:
        image, mask = augment(image, mask)
    
    return image, mask

# --- LOAD DATASET ---
print("Loading Oxford-IIIT Pet dataset...")
train_ds, test_ds = tfds.load('oxford_iiit_pet:3.2.0', split=['train', 'test'], with_info=False)

train = train_ds.map(lambda x: preprocess(x, augment_data=True)).shuffle(512).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test = test_ds.map(lambda x: preprocess(x, augment_data=False)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"✅ Dataset loaded: {IMG_SIZE}x{IMG_SIZE}, Batch size: {BATCH_SIZE}")

# --- IMPROVED U-NET MODEL ---
def improved_unet(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    """U-Net with BatchNormalization and Dropout"""
    inputs = tf.keras.Input(input_size)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    p1 = tf.keras.layers.MaxPooling2D(2)(c1)
    p1 = tf.keras.layers.Dropout(0.1)(p1)

    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    p2 = tf.keras.layers.MaxPooling2D(2)(c2)
    p2 = tf.keras.layers.Dropout(0.1)(p2)

    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    p3 = tf.keras.layers.MaxPooling2D(2)(c3)
    p3 = tf.keras.layers.Dropout(0.2)(p3)

    c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    p4 = tf.keras.layers.MaxPooling2D(2)(c4)
    p4 = tf.keras.layers.Dropout(0.2)(p4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Dropout(0.3)(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c9)
    
    return tf.keras.Model(inputs, outputs)

# --- CUSTOM LOSS: DICE + BCE ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss for better boundary detection"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    """Combine Dice Loss + Binary Crossentropy"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# --- BUILD AND COMPILE MODEL ---
print("Building improved U-Net model...")
model = improved_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=combined_loss,
    metrics=['accuracy', dice_coefficient]
)
model.summary()

# --- CALLBACKS ---
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='pet_unet_improved_best.keras',
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        mode='max',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# --- TRAINING ---
print(f"\n🚀 Starting training for {EPOCHS} epochs...")
history = model.fit(
    train,
    validation_data=test,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- SAVE FINAL MODEL ---
model.save('pet_unet_improved_final.keras')
print("✅ Model saved as pet_unet_improved_final.keras")

# --- PLOT TRAINING HISTORY ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['dice_coefficient'], label='Train Dice')
plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("✅ Training history saved as training_history.png")
plt.show()

# --- VISUALIZE PREDICTIONS ---
def show_predictions(dataset, num=3):
    for imgs, masks in dataset.take(1):
        preds = model.predict(imgs)
        for i in range(min(num, len(imgs))):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(imgs[i])
            plt.title("Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(masks[i].numpy().squeeze(), cmap='gray')
            plt.title("True Mask")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(preds[i].squeeze() > 0.5, cmap='gray')
            plt.title("Pred Mask")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

print("\n📊 Showing predictions on test set...")
show_predictions(test, num=5)

print("\n✅ Training complete!")
print(f"   Best model: pet_unet_improved_best.keras")
print(f"   Final model: pet_unet_improved_final.keras")
