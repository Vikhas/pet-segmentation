"""
Simplified Pet Segmentation Pipeline
Uses U-Net segmentation output directly as alpha matte (no separate matting model)
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 128

class SimplePetPipeline:
    """Simplified pipeline using U-Net segmentation as alpha matte"""
    
    def __init__(self, segmentation_model_path, img_size=128):
        self.img_size = img_size
        
        # Load pre-trained U-Net segmentation model
        print("Loading U-Net segmentation model...")
        self.segmentation_model = tf.keras.models.load_model(segmentation_model_path)
        print("Segmentation model loaded")
    
    def predict_alpha_matte(self, image):
        """
        Generate alpha matte using U-Net segmentation
        
        Args:
            image: Input RGB image (already normalized to [0, 1])
            
        Returns:
            alpha_matte: Predicted alpha matte (soft segmentation)
            binary_mask: Binary segmentation mask
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        # Get segmentation prediction (this is our alpha matte)
        alpha_matte = self.segmentation_model.predict(image, verbose=0)
        
        # Also create binary mask for comparison
        binary_mask = (alpha_matte > 0.5).astype(np.float32)
        
        return alpha_matte, binary_mask
    
    def extract_foreground(self, image, alpha_matte):
        """
        Extract foreground using alpha matte
        
        Args:
            image: Original RGB image
            alpha_matte: Alpha matte
            
        Returns:
            foreground: Extracted foreground with alpha channel
        """
        # Ensure shapes match
        if len(image.shape) == 4:
            image = image[0]
        if len(alpha_matte.shape) == 4:
            alpha_matte = alpha_matte[0]
        
        # Apply alpha matte
        foreground = image * alpha_matte
        
        return foreground
    
    def visualize_results(self, image, alpha_matte, binary_mask, ground_truth=None, save_path=None):
        """Visualize pipeline results"""
        # Ensure correct shapes for visualization
        if len(image.shape) == 4:
            image = image[0]
        if len(alpha_matte.shape) == 4:
            alpha_matte = alpha_matte[0, :, :, 0]
        if len(binary_mask.shape) == 4:
            binary_mask = binary_mask[0, :, :, 0]
        
        # Create composite
        foreground = self.extract_foreground(image, np.expand_dims(alpha_matte, -1))
        
        # Setup figure
        if ground_truth is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Alpha matte (soft segmentation)
        axes[0, 1].imshow(alpha_matte, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Alpha Matte (U-Net)\nMean: {alpha_matte.mean():.3f}', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        if ground_truth is not None:
            # Ground truth
            axes[0, 2].imshow(ground_truth, cmap='gray')
            axes[0, 2].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
        
        # Binary mask
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title('Binary Mask (threshold=0.5)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Extracted foreground
        axes[1, 1].imshow(foreground)
        axes[1, 1].set_title('Extracted Foreground', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        if ground_truth is not None:
            # Comparison overlay
            axes[1, 2].imshow(image)
            axes[1, 2].imshow(alpha_matte, cmap='Reds', alpha=0.5)
            axes[1, 2].set_title('Alpha Matte Overlay', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.close()

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("SIMPLIFIED PET SEGMENTATION PIPELINE")
    print("Using U-Net segmentation as alpha matte")
    print("="*60)
    
    # Load dataset
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
    
    # Initialize pipeline
    # Model path (relative to project root)
    segmentation_model_path = '../models/fba/unet/pet_unet_improved_final.keras'
    pipeline = SimplePetPipeline(segmentation_model_path, img_size=IMG_SIZE)
    
    # Test on sample images
    print("\n" + "="*60)
    print("Processing test images...")
    print("="*60)
    
    for i, (image, mask) in enumerate(test.take(5)):
        print(f"\nProcessing test image {i+1}/5...")
        image_np = np.expand_dims(image.numpy(), 0)
        
        # Run pipeline
        alpha_matte, binary_mask = pipeline.predict_alpha_matte(image_np)
        
        # Visualize
        save_path = f'./outputs/simple_result_{i+1}.png'
        os.makedirs('./outputs', exist_ok=True)
        pipeline.visualize_results(
            image_np, 
            alpha_matte, 
            binary_mask, 
            ground_truth=mask.numpy(),
            save_path=save_path
        )
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED!")
    print("="*60)
    print("\nResults saved to:")
    for i in range(1, 6):
        print(f"  - simple_result_{i}.png")

if __name__ == "__main__":
    main()
