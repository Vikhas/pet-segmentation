"""
Data utilities for MODNet training
Includes data augmentation, trimap generation, and dataset preparation
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import cv2


def generate_trimap_for_training(mask: np.ndarray, 
                                  kernel_size: int = 10,
                                  iterations: int = 1) -> np.ndarray:
    """
    Generate trimap from binary mask for training
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        kernel_size: Size of dilation/erosion kernel
        iterations: Number of dilation/erosion iterations
        
    Returns:
        Trimap with values: 0 (background), 128 (unknown), 255 (foreground)
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilate to get outer boundary
    dilated = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    
    # Erode to get inner boundary
    eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)
    
    # Create trimap
    trimap = np.zeros_like(mask_uint8)
    trimap[dilated == 255] = 128  # Unknown region
    trimap[eroded == 255] = 255   # Foreground
    
    return trimap


def augment_image_and_mask(image: tf.Tensor, 
                           mask: tf.Tensor,
                           training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply data augmentation to image and mask
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W, 1)
        training: Whether to apply augmentation
        
    Returns:
        Augmented (image, mask) tuple
    """
    if not training:
        return image, mask
    
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Random hue (slight variation)
    image = tf.image.random_hue(image, max_delta=0.1)
    
    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask


def create_training_dataset(images: np.ndarray,
                            masks: np.ndarray,
                            batch_size: int = 16,
                            shuffle: bool = True,
                            augment: bool = True) -> tf.data.Dataset:
    """
    Create optimized TensorFlow dataset for training
    
    Args:
        images: Array of images (N, H, W, 3)
        masks: Array of masks (N, H, W, 1)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    if augment:
        dataset = dataset.map(
            lambda x, y: augment_image_and_mask(x, y, training=True),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def prepare_modnet_training_data(train_ds, 
                                 test_ds,
                                 img_size: int = 128,
                                 num_train: int = 2000,
                                 num_test: int = 200) -> Tuple:
    """
    Prepare training and validation data for MODNet
    
    Args:
        train_ds: Training dataset from tfds
        test_ds: Test dataset from tfds
        img_size: Image size
        num_train: Number of training samples
        num_test: Number of test samples
        
    Returns:
        (X_train, y_train, X_test, y_test) numpy arrays
    """
    def preprocess(data):
        image = data['image']
        mask = data['segmentation_mask']
        
        # Resize
        image = tf.image.resize(image, (img_size, img_size))
        mask = tf.image.resize(mask, (img_size, img_size), method='nearest')
        
        # Normalize image
        image = image / 255.0
        
        # Convert mask to binary (pet vs background)
        mask = tf.where(mask > 1, 1.0, 0.0)
        mask = tf.expand_dims(mask, -1) if len(mask.shape) == 2 else mask
        
        return image, mask
    
    print(f"Preparing {num_train} training samples...")
    train_images = []
    train_masks = []
    
    for data in train_ds.take(num_train):
        img, mask = preprocess(data)
        train_images.append(img.numpy())
        train_masks.append(mask.numpy())
    
    print(f"Preparing {num_test} test samples...")
    test_images = []
    test_masks = []
    
    for data in test_ds.take(num_test):
        img, mask = preprocess(data)
        test_images.append(img.numpy())
        test_masks.append(mask.numpy())
    
    X_train = np.array(train_images)
    y_train = np.array(train_masks)
    X_test = np.array(test_images)
    y_test = np.array(test_masks)
    
    print(f"\nData prepared:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")
    print(f"  Image range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  Mask range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  Mask mean: {y_train.mean():.4f}")
    
    return X_train, y_train, X_test, y_test


def visualize_augmentation(image: np.ndarray, 
                           mask: np.ndarray,
                           num_samples: int = 5):
    """
    Visualize data augmentation effects
    
    Args:
        image: Single image (H, W, 3)
        mask: Single mask (H, W, 1)
        num_samples: Number of augmented samples to generate
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Apply augmentation
        img_tensor = tf.constant(image)
        mask_tensor = tf.constant(mask)
        aug_img, aug_mask = augment_image_and_mask(img_tensor, mask_tensor, training=True)
        
        # Plot original
        if i == 0:
            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.squeeze(), cmap='gray')
            axes[i, 1].set_title('Original Mask')
            axes[i, 1].axis('off')
        else:
            axes[i, 0].imshow(aug_img.numpy())
            axes[i, 0].set_title(f'Augmented {i}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(aug_mask.numpy().squeeze(), cmap='gray')
            axes[i, 1].set_title(f'Augmented Mask {i}')
            axes[i, 1].axis('off')
        
        # Generate and show trimap
        trimap = generate_trimap_for_training(mask.squeeze())
        axes[i, 2].imshow(trimap, cmap='gray')
        axes[i, 2].set_title('Trimap')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("Saved augmentation examples to augmentation_examples.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("MODNet Data Utils Test")
    print("=" * 60)
    
    # Test trimap generation
    print("\n1. Testing trimap generation...")
    test_mask = np.zeros((128, 128))
    test_mask[40:90, 40:90] = 1.0
    
    trimap = generate_trimap_for_training(test_mask, kernel_size=10)
    print(f"   Trimap shape: {trimap.shape}")
    print(f"   Trimap unique values: {np.unique(trimap)}")
    
    # Test augmentation
    print("\n2. Testing augmentation...")
    test_image = np.random.rand(128, 128, 3).astype(np.float32)
    test_mask_3d = np.expand_dims(test_mask, -1).astype(np.float32)
    
    aug_img, aug_mask = augment_image_and_mask(
        tf.constant(test_image),
        tf.constant(test_mask_3d),
        training=True
    )
    print(f"   Augmented image shape: {aug_img.shape}")
    print(f"   Augmented mask shape: {aug_mask.shape}")
    
    # Test dataset creation
    print("\n3. Testing dataset creation...")
    dummy_images = np.random.rand(100, 128, 128, 3).astype(np.float32)
    dummy_masks = np.random.randint(0, 2, (100, 128, 128, 1)).astype(np.float32)
    
    dataset = create_training_dataset(
        dummy_images, 
        dummy_masks,
        batch_size=8,
        shuffle=True,
        augment=True
    )
    
    for batch_images, batch_masks in dataset.take(1):
        print(f"   Batch images shape: {batch_images.shape}")
        print(f"   Batch masks shape: {batch_masks.shape}")
    
    print("\n" + "=" * 60)
    print("Data utils test completed successfully!")
    print("=" * 60)
