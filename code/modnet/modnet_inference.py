"""
MODNet inference script
Generate high-quality pet cutouts using trained MODNet model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_modnet(model_path='../models/modnet_pet_matting.keras'):
    """Load trained MODNet model"""
    print(f"Loading MODNet from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully")
    return model


def preprocess_image(image_path, target_size=128):
    """
    Load and preprocess image for MODNet inference
    
    Args:
        image_path: Path to input image
        target_size: Target size for model input
        
    Returns:
        Preprocessed image tensor and original image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Resize for model
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to array and normalize
    img_array = np.array(img_resized) / 255.0
    img_tensor = np.expand_dims(img_array, 0).astype(np.float32)
    
    return img_tensor, img, original_size


def postprocess_alpha(alpha_pred, original_size):
    """
    Postprocess alpha matte to original image size
    
    Args:
        alpha_pred: Predicted alpha matte (1, H, W, 1)
        original_size: Original image size (W, H)
        
    Returns:
        Resized alpha matte
    """
    # Remove batch dimension and invert (model trained with bg=1, fg=0)
    alpha = 1.0 - alpha_pred[0, :, :, 0]
    
    # Convert to PIL Image
    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
    
    # Resize to original size
    alpha_resized = alpha_img.resize(original_size, Image.BILINEAR)
    
    return np.array(alpha_resized) / 255.0


def create_cutout(image_path, alpha_matte, output_path):
    """
    Create transparent cutout from image and alpha matte
    
    Args:
        image_path: Path to original image
        alpha_matte: Alpha matte (H, W)
        output_path: Path to save cutout
    """
    # Load original image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Create RGBA image
    rgba = np.dstack([img_array, (alpha_matte * 255).astype(np.uint8)])
    
    # Save as PNG
    cutout = Image.fromarray(rgba, 'RGBA')
    cutout.save(output_path)
    
    return cutout


def batch_inference(model, image_dir, output_dir, target_size=128):
    """
    Run inference on all images in a directory
    
    Args:
        model: Trained MODNet model
        image_dir: Directory containing input images
        output_dir: Directory to save cutouts
        target_size: Model input size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        
        # Preprocess
        img_tensor, _, original_size = preprocess_image(image_path, target_size)
        
        # Predict
        alpha_pred = model.predict(img_tensor, verbose=0)
        
        # Postprocess
        alpha_matte = postprocess_alpha(alpha_pred, original_size)
        
        # Create cutout
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_cutout.png")
        create_cutout(image_path, alpha_matte, output_path)
        
        print(f"  [{i+1}/{len(image_files)}] Processed {filename}")
    
    print(f"\nAll cutouts saved to {output_dir}")


def visualize_inference(image_path, model, save_path=None):
    """
    Visualize inference result
    
    Args:
        image_path: Path to input image
        model: Trained MODNet model
        save_path: Optional path to save visualization
    """
    # Preprocess
    img_tensor, original_img, original_size = preprocess_image(image_path)
    
    # Predict
    alpha_pred = model.predict(img_tensor, verbose=0)
    
    # Postprocess
    alpha_matte = postprocess_alpha(alpha_pred, original_size)
    
    # Create cutout
    img_array = np.array(original_img)
    rgba = np.dstack([img_array, (alpha_matte * 255).astype(np.uint8)])
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(alpha_matte, cmap='gray')
    axes[1].set_title('Predicted Alpha Matte', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(rgba)
    axes[2].set_title('Transparent Cutout', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print(" " * 25 + "MODNet Inference")
    print("=" * 70)
    
    # Load model
    model = load_modnet()
    
    # Example: Process a single image
    # Uncomment and modify the path below to test on your own image
    # visualize_inference('path/to/your/image.jpg', model, 'output_visualization.png')
    
    print("\n" + "=" * 70)
    print("MODNet inference module ready!")
    print("=" * 70)
    print("\nUsage examples:")
    print("  1. Single image:")
    print("     visualize_inference('image.jpg', model, 'output.png')")
    print("\n  2. Batch processing:")
    print("     batch_inference(model, 'input_dir/', 'output_dir/')")
