"""
Complete Pet Segmentation and Matting Pipeline
Implements the two-stage approach: Segmentation (U-Net) → Matting (MODNet-inspired)
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from trimap_generation import generate_trimap_adaptive, batch_generate_trimaps
from matting_model import create_matting_model, prepare_matting_data
import os

IMG_SIZE = 128

class PetMattingPipeline:
    """Complete pipeline for pet segmentation and matting"""
    
    def __init__(self, segmentation_model_path, img_size=128):
        self.img_size = img_size
        
        # Load pre-trained U-Net segmentation model
        print("Loading U-Net segmentation model...")
        self.segmentation_model = tf.keras.models.load_model(segmentation_model_path)
        print("Segmentation model loaded")
        
        # Create matting model
        print("Creating matting model...")
        self.matting_model = create_matting_model(img_size=img_size)
        print("Matting model created")
        
    def preprocess_image(self, image, mask=None):
        """Preprocess image and mask"""
        image = tf.image.resize(image, (self.img_size, self.img_size)) / 255.0
        if mask is not None:
            mask = tf.image.resize(mask, (self.img_size, self.img_size), method='nearest')
            mask = tf.where(mask > 1, 1.0, 0.0)
            return image, mask
        return image
    
    def generate_segmentation_masks(self, images):
        """Generate binary masks using U-Net"""
        predictions = self.segmentation_model.predict(images)
        binary_masks = (predictions > 0.5).astype(np.float32)
        return binary_masks
    
    def generate_trimaps(self, masks, uncertainty_region=5):
        """Generate trimaps from binary masks"""
        trimaps = batch_generate_trimaps(
            masks.squeeze(),
            method='adaptive',
            uncertainty_region=uncertainty_region
        )
        return np.expand_dims(trimaps, axis=-1)
    
    def train_matting_model(self, train_ds, val_ds, epochs=20, batch_size=16):
        """
        Train the matting model.
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("\n" + "="*50)
        print("TRAINING MATTING MODEL")
        print("="*50)
        
        # Prepare training data
        train_images, train_masks = [], []
        for image, mask in train_ds.take(500):  # Use subset for training
            img, msk = self.preprocess_image(image, mask)
            train_images.append(img.numpy())
            train_masks.append(msk.numpy())
        
        train_images = np.array(train_images)
        train_masks = np.array(train_masks)
        
        # Generate segmentation masks using U-Net
        print("Generating segmentation masks...")
        seg_masks = self.generate_segmentation_masks(train_images)
        
        # Generate trimaps
        print("Generating trimaps...")
        trimaps = self.generate_trimaps(seg_masks)
        
        # Prepare matting data
        X_train, y_train = prepare_matting_data(train_images, trimaps, train_masks)
        
        # Prepare validation data
        val_images, val_masks = [], []
        for image, mask in val_ds.take(100):
            img, msk = self.preprocess_image(image, mask)
            val_images.append(img.numpy())
            val_masks.append(msk.numpy())
        
        val_images = np.array(val_images)
        val_masks = np.array(val_masks)
        
        val_seg_masks = self.generate_segmentation_masks(val_images)
        val_trimaps = self.generate_trimaps(val_seg_masks)
        X_val, y_val = prepare_matting_data(val_images, val_trimaps, val_masks)
        
        # Train model
        print(f"\nTraining on {len(X_train)} samples...")
        history = self.matting_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
            ]
        )
        
        return history
    
    def predict_alpha_matte(self, image):
        """
        Complete pipeline: image → segmentation → trimap → alpha matte
        
        Args:
            image: Input RGB image
            
        Returns:
            alpha_matte: Predicted alpha matte
            trimap: Generated trimap
            seg_mask: Segmentation mask
        """
        # Preprocess
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        # Step 1: Segmentation
        seg_mask = self.generate_segmentation_masks(image)
        
        # Step 2: Trimap generation
        trimap = self.generate_trimaps(seg_mask)
        
        # Ensure trimap has batch dimension
        if len(trimap.shape) == 3:
            trimap = np.expand_dims(trimap, 0)
            
        # Also ensure it has the channel dimension if missing (sometimes happens with squeeze)
        if len(trimap.shape) == 3:
            trimap = np.expand_dims(trimap, -1)
            
        # Step 3: Matting
        matting_input = tf.concat([image, trimap/255.0], axis=-1)
        alpha_matte = self.matting_model(matting_input, training=False)
        
        return alpha_matte.numpy(), trimap, seg_mask
    
    def save_matting_model(self, path):
        """Save trained matting model"""
        self.matting_model.save(path)
        print(f"Matting model saved to {path}")
    
    def visualize_results(self, image, alpha_matte, trimap, seg_mask, save_path=None):
        """Visualize pipeline results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image[0])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Segmentation mask
        axes[0, 1].imshow(seg_mask[0, :, :, 0], cmap='gray')
        axes[0, 1].set_title('Segmentation Mask (U-Net)')
        axes[0, 1].axis('off')
        
        # Trimap
        axes[0, 2].imshow(trimap[0, :, :, 0], cmap='gray')
        axes[0, 2].set_title('Trimap')
        axes[0, 2].axis('off')
        
        # Alpha matte
        axes[1, 0].imshow(alpha_matte[0, :, :, 0], cmap='gray')
        axes[1, 0].set_title('Alpha Matte (Matting Model)')
        axes[1, 0].axis('off')
        
        # Composite (foreground extraction)
        composite = image[0] * alpha_matte[0]
        axes[1, 1].imshow(composite)
        axes[1, 1].set_title('Extracted Foreground')
        axes[1, 1].axis('off')
        
        # Comparison: Seg vs Matte
        axes[1, 2].imshow(np.concatenate([seg_mask[0, :, :, 0], alpha_matte[0, :, :, 0]], axis=1), cmap='gray')
        axes[1, 2].set_title('Seg Mask vs Alpha Matte')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("PET SEGMENTATION AND MATTING PIPELINE")
    print("="*60)
    
    # Load dataset
    print("\nLoading Oxford-IIIT Pet dataset...")
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
    
    # Initialize pipeline
    segmentation_model_path = '/Users/vikhas/Documents/projects/MLD_project/pet_unet_final_10.keras'
    pipeline = PetMattingPipeline(segmentation_model_path, img_size=IMG_SIZE)
    
    # Train matting model
    print("\n" + "="*60)
    print("STEP 1: Training Matting Model")
    print("="*60)
    history = pipeline.train_matting_model(train, test, epochs=15, batch_size=16)
    
    # Matting model path
    matting_model_path = '../models/modnet/modnet_pet_matting.keras'
    pipeline.save_matting_model(matting_model_path)
    
    # Test on sample images
    print("\n" + "="*60)
    print("STEP 2: Testing on Sample Images")
    
    for i, (image, mask) in enumerate(test.take(5)):
        print(f"\nProcessing test image {i+1}/5...")
        image_np = np.expand_dims(image.numpy(), 0)
        
        # Run complete pipeline
        alpha_matte, trimap, seg_mask = pipeline.predict_alpha_matte(image_np)
        
        # Visualize
        save_path = f'./outputs/result_{i+1}.png'
        os.makedirs('./outputs', exist_ok=True)
        pipeline.visualize_results(image_np, alpha_matte, trimap, seg_mask, save_path)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
