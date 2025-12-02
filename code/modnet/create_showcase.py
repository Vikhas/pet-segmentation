"""
Create showcase composites by placing pet cutouts on different backgrounds.
"""

import numpy as np
from PIL import Image
import os
import glob

# Directories
CUTOUTS_DIR = 'final_cutouts'
SHOWCASE_DIR = 'showcase_results'
# Background images directory (you can place your own backgrounds here)
BACKGROUNDS_DIR = './data/backgrounds'  # Create this directory and add background images

# Ensure output directory exists
os.makedirs(SHOWCASE_DIR, exist_ok=True)

def composite_on_background(cutout_path, background_path, output_path, scale=1.0):
    """
    Composite a transparent cutout onto a background.
    
    Args:
        cutout_path: Path to RGBA cutout image
        background_path: Path to background image
        output_path: Path to save the composite
        scale: Scale factor for the cutout (default 1.0)
    """
    # Load images
    cutout = Image.open(cutout_path).convert('RGBA')
    background = Image.open(background_path).convert('RGB')
    
    # Resize cutout if needed
    if scale != 1.0:
        new_size = (int(cutout.width * scale), int(cutout.height * scale))
        cutout = cutout.resize(new_size, Image.Resampling.LANCZOS)
    
    # Resize background to match cutout size (or vice versa)
    # Let's resize background to be larger and center the cutout
    bg_size = (512, 512)
    background = background.resize(bg_size, Image.Resampling.LANCZOS)
    
    # Resize cutout to fit nicely (e.g., 70% of background)
    cutout_size = (int(bg_size[0] * 0.7), int(bg_size[1] * 0.7))
    cutout = cutout.resize(cutout_size, Image.Resampling.LANCZOS)
    
    # Create a new image for the composite
    composite = background.copy()
    
    # Calculate position to center the cutout
    x = (bg_size[0] - cutout.width) // 2
    y = (bg_size[1] - cutout.height) // 2
    
    # Paste cutout onto background using alpha channel as mask
    composite.paste(cutout, (x, y), cutout)
    
    # Save
    composite.save(output_path, quality=95)
    print(f"Created composite: {output_path}")

def create_grid(image_paths, output_path, cols=3):
    """Create a grid of images"""
    images = [Image.open(p) for p in image_paths]
    
    # Get dimensions
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    
    # Create grid
    grid = Image.new('RGB', (w * cols, h * rows), color='white')
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * w, row * h))
    
    grid.save(output_path, quality=95)
    print(f"Created grid: {output_path}")

def main():
    print("="*60)
    print("CREATING SHOWCASE COMPOSITES")
    print("="*60)
    
    # Find all cutouts
    cutout_files = sorted(glob.glob(os.path.join(CUTOUTS_DIR, 'cutout_*.png')))
    print(f"\nFound {len(cutout_files)} cutouts")
    
    # Find background images
    background_files = glob.glob(os.path.join(BACKGROUNDS_DIR, '*_background_*.png'))
    print(f"Found {len(background_files)} backgrounds")
    
    if len(background_files) == 0:
        print("Warning:  No background images found. Please ensure backgrounds are generated first.")
        return
    
    # Create composites for first 3 cutouts on each background
    composite_paths = []
    
    for bg_idx, bg_path in enumerate(background_files):
        bg_name = os.path.basename(bg_path).split('_')[0]
        print(f"\n📸 Compositing on {bg_name} background...")
        
        for cutout_idx, cutout_path in enumerate(cutout_files[:3]):
            output_name = f"composite_{bg_name}_{cutout_idx+1}.jpg"
            output_path = os.path.join(SHOWCASE_DIR, output_name)
            
            composite_on_background(cutout_path, bg_path, output_path)
            composite_paths.append(output_path)
    
    # Create a grid of all composites
    if composite_paths:
        grid_path = os.path.join(SHOWCASE_DIR, 'showcase_grid.jpg')
        create_grid(composite_paths, grid_path, cols=3)
    
    print("\n" + "="*60)
    print(f"DONE! Check the '{SHOWCASE_DIR}' folder for results.")
    print("="*60)

if __name__ == "__main__":
    main()
