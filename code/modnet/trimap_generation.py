import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

def generate_trimap(mask, erosion_kernel_size=5, dilation_kernel_size=5):
    """
    Generate trimap from binary mask.
    
    Args:
        mask: Binary mask (0 for background, 1 for foreground)
        erosion_kernel_size: Size of erosion kernel for foreground
        dilation_kernel_size: Size of dilation kernel for background
        
    Returns:
        trimap: Trimap with values 0 (background), 128 (unknown), 255 (foreground)
    """
    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)
    
    # Create kernels
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (erosion_kernel_size, erosion_kernel_size))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (dilation_kernel_size, dilation_kernel_size))
    
    # Erode mask to get definite foreground
    foreground = cv2.erode(mask, erosion_kernel, iterations=1)
    
    # Dilate mask to get definite background
    background = 1 - cv2.dilate(mask, dilation_kernel, iterations=1)
    
    # Create trimap
    trimap = np.zeros_like(mask, dtype=np.uint8)
    trimap[foreground == 1] = 255  # Definite foreground
    trimap[background == 1] = 0    # Definite background
    trimap[(foreground == 0) & (background == 0)] = 128  # Unknown region
    
    return trimap

def generate_trimap_adaptive(mask, uncertainty_region=10):
    """
    Generate trimap using distance transform for adaptive boundary width.
    
    Args:
        mask: Binary mask (0 for background, 1 for foreground)
        uncertainty_region: Width of uncertainty region in pixels
        
    Returns:
        trimap: Trimap with values 0 (background), 128 (unknown), 255 (foreground)
    """
    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Compute distance transforms
    dt_fg = distance_transform_edt(mask)
    dt_bg = distance_transform_edt(1 - mask/255)
    
    # Create trimap
    trimap = np.zeros_like(mask, dtype=np.uint8)
    trimap[dt_fg > uncertainty_region] = 255  # Definite foreground
    trimap[dt_bg > uncertainty_region] = 0    # Definite background
    trimap[(dt_fg <= uncertainty_region) & (dt_bg <= uncertainty_region)] = 128  # Unknown
    
    return trimap

def batch_generate_trimaps(masks, method='adaptive', **kwargs):
    """
    Generate trimaps for a batch of masks.
    
    Args:
        masks: Batch of binary masks (batch_size, height, width)
        method: 'simple' or 'adaptive'
        **kwargs: Additional arguments for trimap generation
        
    Returns:
        trimaps: Batch of trimaps
    """
    trimaps = []
    
    for mask in masks:
        if method == 'adaptive':
            trimap = generate_trimap_adaptive(mask, **kwargs)
        else:
            trimap = generate_trimap(mask, **kwargs)
        trimaps.append(trimap)
    
    return np.array(trimaps)

if __name__ == "__main__":
    # Test trimap generation
    import matplotlib.pyplot as plt
    
    # Create a simple test mask
    test_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(test_mask, (64, 64), 40, 1, -1)
    
    # Generate trimaps
    trimap_simple = generate_trimap(test_mask)
    trimap_adaptive = generate_trimap_adaptive(test_mask, uncertainty_region=5)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(test_mask, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[1].imshow(trimap_simple, cmap='gray')
    axes[1].set_title('Simple Trimap')
    axes[2].imshow(trimap_adaptive, cmap='gray')
    axes[2].set_title('Adaptive Trimap')
    plt.tight_layout()
    plt.savefig('trimap_test.png')
    print("Trimap generation test completed!")
