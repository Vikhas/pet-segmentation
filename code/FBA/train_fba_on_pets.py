"""
Minimal training loop to fine-tune the FBA Matting model using the
Oxford-IIIT Pet dataset masks as pseudo alpha mattes.  We leverage
TensorFlow Datasets to read the data but perform training in PyTorch.

NOTE: FBA Matting expects high-resolution matting pairs (image, alpha,
trimap).  Oxford-IIIT Pet only provides coarse segmentation masks, so
this script simply demonstrates how to wrap the dataset and run a
fine-tuning loop.  For best results you should derive soft alpha mattes
and produce high-quality trimaps (e.g. from MODNet or manual labeling).
"""

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset
from torch.optim import Adam
import torch.nn as nn
from torch.nn import functional as F

from FBAMattingRefiner import FBAMattingRefiner

IMG_SIZE = 256
BATCH_SIZE = 4
NUM_STEPS = 10000
LEARNING_RATE = 1e-5
CHECKPOINT_DIR = "fba_pet_checkpoints"


def gaussian_blur(mask: tf.Tensor, sigma: float = 1.5, kernel_size: int = 5) -> tf.Tensor:
    """Apply Gaussian blur using depthwise conv (TF lacks gaussian_filter2d)."""
    radius = kernel_size // 2
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    gauss_1d = tf.exp(-(x ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / tf.reduce_sum(gauss_1d)
    gauss_2d = tf.tensordot(gauss_1d, gauss_1d, axes=0)
    gauss_2d = gauss_2d / tf.reduce_sum(gauss_2d)
    kernel = tf.reshape(gauss_2d, (kernel_size, kernel_size, 1, 1))

    mask_2d = mask
    if mask_2d.shape.rank == 3:
        mask_2d = tf.squeeze(mask_2d, axis=-1)

    mask4d = mask_2d[tf.newaxis, :, :, tf.newaxis]
    blurred = tf.nn.depthwise_conv2d(
        mask4d, kernel, strides=[1, 1, 1, 1], padding="SAME"
    )
    return tf.squeeze(blurred, axis=[0, 3])


def preprocess_sample(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    mask = tf.cast(data["segmentation_mask"], tf.float32)
    mask = tf.where(mask > 1, 1.0, mask)  # reduce to binary

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")

    # Produce soft alpha by dilating/blur
    alpha = gaussian_blur(mask, sigma=1.5, kernel_size=5)
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)

    sample = {
        "image": image,
        "alpha": alpha,
    }
    return sample


import glob
import cv2
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class PseudoLabelDataset(Dataset):
    def __init__(self, root_dir, max_samples=None):
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.png")))
        self.alpha_paths = sorted(glob.glob(os.path.join(root_dir, "alphas", "*.png")))
        
        if max_samples is not None and max_samples < len(self.image_paths):
            # Shuffle and slice to keep random subset
            combined = list(zip(self.image_paths, self.alpha_paths))
            random.seed(42) # Fixed seed for reproducibility
            random.shuffle(combined)
            combined = combined[:max_samples]
            self.image_paths, self.alpha_paths = zip(*combined)
            print(f"Dataset reduced to {len(self.image_paths)} samples.")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read Image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Keep as uint8 for now for transforms, or convert to PIL
        image = torch.from_numpy(image.transpose((2, 0, 1))) # (C, H, W) uint8
        
        # Read Alpha
        alpha_path = self.alpha_paths[idx]
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        alpha = torch.from_numpy(alpha).unsqueeze(0) # (1, H, W) uint8
        
        # Resize to fixed size for batching
        image = TF.resize(image, [IMG_SIZE, IMG_SIZE])
        alpha = TF.resize(alpha, [IMG_SIZE, IMG_SIZE])

        # --- Augmentation ---
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            alpha = TF.hflip(alpha)
            
        # 2. Random Affine (Rotation, Scale, Shift)
        # We apply the same parameters to both
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            scale = random.uniform(0.9, 1.1)
            translate = (random.randint(-10, 10), random.randint(-10, 10))
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0)
            alpha = TF.affine(alpha, angle=angle, translate=translate, scale=scale, shear=0)
            
        # 3. Color Jitter (Image only)
        if random.random() > 0.5:
            # Brightness
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            # Contrast
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            # Saturation
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            
        # Normalize to [0, 1] float
        image_t = image.float() / 255.0
        alpha_t = alpha.float() / 255.0
        
        # Build Trimap (Random Erosion/Dilation)
        # Need numpy for cv2 operations in build_trimap
        alpha_np = alpha_t.squeeze().numpy()
        trimap = self.build_trimap(alpha_np)
        trimap_t = torch.from_numpy(trimap[np.newaxis, :, :]).float()
        
        return image_t, alpha_t, trimap_t

    @staticmethod
    def build_trimap(alpha: np.ndarray) -> np.ndarray:
        # Randomly choose an unknown region width between 10 and 30
        k_size = np.random.randint(10, 30)
        if k_size % 2 == 0:
            k_size += 1
            
        # Create binary masks
        fg = (alpha >= 0.9).astype(np.uint8)
        bg = (alpha <= 0.1).astype(np.uint8)
        
        # Erode FG and BG to widen the unknown region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        fg_eroded = cv2.erode(fg, kernel, iterations=1)
        bg_eroded = cv2.erode(bg, kernel, iterations=1)
        
        trimap = np.full_like(alpha, 0.5, dtype=np.float32)
        trimap[fg_eroded == 1] = 1.0
        trimap[bg_eroded == 1] = 0.0
        
        return trimap

        return trimap


@dataclass
class FBATrainingBatch:
    image: torch.Tensor  # (B,3,H,W)
    trimap_two_chan: torch.Tensor  # (B,2,H,W)
    image_norm: torch.Tensor  # (B,3,H,W)
    trimap_feats: torch.Tensor  # (B,6,H,W)
    target_alpha: torch.Tensor  # (B,1,H,W)


def prepare_batch(refiner: FBAMattingRefiner, images, alphas, trimaps) -> FBATrainingBatch:
    device = refiner.device

    images = images.to(device)
    alphas = alphas.to(device)
    trimaps = trimaps.to(device)

    trimap_tensor = trimaps
    cat_dim = 1 if trimap_tensor.ndim == 4 else 0
    two_chan = torch.cat([(trimap_tensor == 1.0).float(), (trimap_tensor == 0.0).float()], dim=cat_dim)
    
    # refiner._trimap_distance_features uses cv2 and expects single image (or handles it poorly for batches)
    # We iterate to be safe and correct
    feats_list = []
    for i in range(trimap_tensor.shape[0]):
        # Extract single item: (1, H, W) or (1, 1, H, W)
        t_single = trimap_tensor[i]
        # Ensure it has the shape expected by refiner (1, 1, H, W) or (1, H, W)
        if t_single.ndim == 2:
            t_single = t_single.unsqueeze(0).unsqueeze(0)
        elif t_single.ndim == 3:
            t_single = t_single.unsqueeze(0)
            
        feat = refiner._trimap_distance_features(t_single)
        feats_list.append(feat)
        
    trimap_feats = torch.cat(feats_list, dim=0)
    imgs_norm = refiner._normalize(images)

    return FBATrainingBatch(
        image=images,
        trimap_two_chan=two_chan,
        image_norm=imgs_norm,
        trimap_feats=trimap_feats,
        target_alpha=alphas,
    )


# --- LOSS FUNCTIONS ---
class LaplacianLoss(nn.Module):
    def __init__(self, max_levels=3, device="cpu"):
        super(LaplacianLoss, self).__init__()
        self.max_levels = max_levels
        self.device = device
        self.kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device).view(1, 1, 3, 3)

    def forward(self, pred, target):
        loss = 0
        current_pred = pred
        current_target = target
        
        for i in range(self.max_levels):
            # Apply Laplacian Kernel
            pred_lap = F.conv2d(current_pred, self.kernel, padding=1)
            target_lap = F.conv2d(current_target, self.kernel, padding=1)
            
            # L1 Loss on Laplacian
            loss += F.l1_loss(pred_lap, target_lap) * (2 ** i) # Weight higher levels more? Or less? Usually 2**i or 4**i
            
            # Downsample for next level
            current_pred = F.avg_pool2d(current_pred, 2)
            current_target = F.avg_pool2d(current_target, 2)
            
        return loss

def train_fba_on_pets():
    print("Loading Pseudo-Label Dataset (v2)...")
    # Use the new v2 labels and full dataset
    dataset = PseudoLabelDataset("pseudo_labels_v2") 
    
    # Increase num_workers to parallelize data loading and augmentation
    # Use os.cpu_count() or a fixed number like 4
    num_workers = 0 # Multiprocessing crashes with MPS on Mac
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    
    iterable = iter(dataloader) # Make it iterable for the loop structure

    ckpt_path = "/Users/lalithasravantidasu/UCD Masters/MLD/pet_segmentation_working/FBA_Matting/fba_matting.pth"
    
    # Select Device: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"🚀 Training on device: {device} (MPS workaround applied)")

    refiner = FBAMattingRefiner(
        ckpt_path,
        device=device,
        fg_threshold=0.85,
        bg_threshold=0.05,
        unknown_width=15,
        blur_ksize=3,
    )

    model = refiner.model
    model.train()
    # Added weight decay for regularization
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Initialize Laplacian Loss
    laplacian_loss = LaplacianLoss(max_levels=3, device=device)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    step = 0
    while step < NUM_STEPS:
        for images, alphas, trimaps in dataloader:
            if step >= NUM_STEPS:
                break
            
            batch = prepare_batch(refiner, images, alphas, trimaps)

            optimizer.zero_grad()

            # --- ROBUST INPUT PIPELINE START ---
            t_img = batch.image
            t_two_chan = batch.trimap_two_chan
            t_img_norm = batch.image_norm
            t_feats = batch.trimap_feats

            # Helper to standardise to (Batch, Channel, Height, Width)
            def standardize_input(tensor, expected_channels):
                # 1. Add Batch Dimension if missing: (C, H, W) -> (1, C, H, W)
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0)
                
                # 2. Fix Channel Ordering
                # If (B, H, W, C), permute to (B, C, H, W)
                if tensor.shape[-1] == expected_channels:
                    tensor = tensor.permute(0, 3, 1, 2)
                
                # If (B, H, C, W), permute to (B, C, H, W) [Rare case]
                elif tensor.shape[2] == expected_channels:
                    tensor = tensor.permute(0, 2, 1, 3)
                    
                return tensor

            # Apply the standardizer
            # Image: Expect 3 channels (RGB)
            t_img = standardize_input(t_img, expected_channels=3)
            
            # Image Norm: Expect 3 channels (RGB)
            t_img_norm = standardize_input(t_img_norm, expected_channels=3)
            
            # Trimap: Expect 2 channels
            t_two_chan = standardize_input(t_two_chan, expected_channels=2)
            
            # Feats: Expect 6 channels (from refiner)
            # Feats usually comes correct now, but ensure batch size matches
            if t_feats.ndim == 3: # (6, 256, 256) -> (1, 6, 256, 256)
                 t_feats = t_feats.unsqueeze(0)
            
            # Expand feats if batch sizes don't match (e.g. feats calculated for 1 item, applied to batch)
            if t_feats.shape[0] != t_img.shape[0]:
                 t_feats = t_feats.repeat(t_img.shape[0], 1, 1, 1)

            # Run Model
            output = model(t_img, t_two_chan, t_img_norm, t_feats)
            # --- ROBUST INPUT PIPELINE END ---

            pred_alpha = torch.clamp(output[:, 0:1], 0.0, 1.0)
            
            if batch.target_alpha.ndim == 3:
                batch.target_alpha = batch.target_alpha.unsqueeze(1)
                
            loss_alpha = F.l1_loss(pred_alpha, batch.target_alpha)
            loss_grad = F.l1_loss(
                pred_alpha[:, :, :, 1:] - pred_alpha[:, :, :, :-1],
                batch.target_alpha[:, :, :, 1:] - batch.target_alpha[:, :, :, :-1],
            ) + F.l1_loss(
                pred_alpha[:, :, 1:, :] - pred_alpha[:, :, :-1, :],
                batch.target_alpha[:, :, 1:, :] - batch.target_alpha[:, :, :-1, :],
            )
            loss_lap = laplacian_loss(pred_alpha, batch.target_alpha)
            
            loss = loss_alpha + 0.5 * loss_grad + 0.5 * loss_lap
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"[Step {step}] loss={loss.item():.4f} (L1={loss_alpha.item():.4f}, Grad={loss_grad.item():.4f}, Lap={loss_lap.item():.4f})")

            if (step + 1) % 500 == 0:
                ckpt_file = os.path.join(CHECKPOINT_DIR, f"fba_pet_step_{step+1}.pth")
                torch.save(model.state_dict(), ckpt_file)
                print(f"✅ Saved checkpoint to {ckpt_file}")
                
            step += 1

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "fba_pet_final.pth"))
    print("🎉 Training completed and checkpoint exported.")


if __name__ == "__main__":
    train_fba_on_pets()

