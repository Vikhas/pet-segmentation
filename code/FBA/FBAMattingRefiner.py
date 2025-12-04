import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

import sys

FBA_ROOT = "FBA_Matting"
sys.path.append(FBA_ROOT)

from networks.models import build_model


class FBAMattingRefiner:
    """
    Thin wrapper around the official FBA Matting network.

    Args:
        ckpt_path: Path to the pretrained `.pth` weights.
        device: Torch device string; defaults to CUDA when available.
    """

    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        fg_threshold: float = 0.8,
        bg_threshold: float = 0.2,
        unknown_width: int = 21,
        blur_ksize: int = 5,
        blend_with_coarse: float = 0.0,
    ):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"FBA checkpoint not found at {ckpt_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.fg_threshold = float(np.clip(fg_threshold, 0.0, 1.0))
        self.bg_threshold = float(np.clip(bg_threshold, 0.0, 1.0))
        self.unknown_width = self._ensure_odd_int(max(1, int(unknown_width)))
        self.blur_ksize = self._ensure_odd_int(max(1, int(blur_ksize))) if blur_ksize else 0
        self.blend_with_coarse = float(np.clip(blend_with_coarse, 0.0, 1.0))

        # build_model loads architecture + weights when a path is provided
        self.model = build_model(ckpt_path).to(self.device)
        self.model.eval()

    def refine(
        self,
        image_np: np.ndarray,
        coarse_alpha: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image_np: RGB float array in [0,1], shape (H,W,3) or (1,H,W,3).
            coarse_alpha: Soft mask (same spatial size); used to derive the trimap.

        Returns:
            refined_alpha: np.ndarray (1,H,W,1) in [0,1].
            trimap: np.ndarray (1,H,W,1) with {0,0.5,1} values.
        """
        # Ensure shapes are canonical (H, W, C) and (H, W)
        if image_np.ndim == 4:
            image_np = image_np.squeeze(0)
        if coarse_alpha.ndim == 4:
            coarse_alpha = coarse_alpha.squeeze(0)
        if coarse_alpha.ndim == 3:
            coarse_alpha = coarse_alpha.squeeze(-1)
            
        # Pad to multiple of 8 to avoid model size mismatch
        h, w = image_np.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            coarse_alpha = np.pad(coarse_alpha, ((0, pad_h), (0, pad_w)), mode='reflect')

        image = self._to_tensor(image_np).to(self.device)
        coarse_np = self._prepare_alpha_np(coarse_alpha)
        trimap = self._build_trimap(coarse_np)
        trimap_tensor = torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0).float().to(self.device)

        two_chan_trimap = self._trimap_to_two_chan(trimap_tensor)
        image_n = self._normalize(image)
        trimap_feats = self._trimap_distance_features(trimap_tensor)

        with torch.no_grad():
            output = self.model(image, two_chan_trimap, image_n, trimap_feats)

        alpha = output[:, 0:1]

        # Clamp in known regions
        alpha = self._respect_trimap(alpha, trimap_tensor)

        alpha_np = alpha.squeeze().cpu().numpy()
        
        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            alpha_np = alpha_np[:h, :w]
            trimap = trimap[:h, :w]

        if self.blend_with_coarse > 0:
            # Note: coarse_np was padded, so we need original coarse_alpha for blending if we crop
            # But we already cropped alpha_np, so we need original coarse_alpha (unpadded)
            # Re-fetch original coarse_alpha passed to function (before padding)
            # Actually, I modified the local variable 'coarse_alpha'. 
            # Let's use the original argument or slice the padded one.
            coarse_original = coarse_np[:h, :w]
            alpha_np = (
                (1.0 - self.blend_with_coarse) * alpha_np
                + self.blend_with_coarse * coarse_original
            )
        alpha_np = np.clip(alpha_np, 0.0, 1.0)

        alpha_np = alpha_np[np.newaxis, :, :, np.newaxis]
        trimap_np = trimap[np.newaxis, :, :, np.newaxis]
        return alpha_np, trimap_np

    # ------------------------------------------------------------------ #
    # Helper methods                                                     #
    # ------------------------------------------------------------------ #

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 4:
            image = image[0]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.IMG_MEAN.to(image.device)) / self.IMG_STD.to(image.device)

    def _build_trimap(self, alpha: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        if self.blur_ksize > 1:
            alpha = cv2.GaussianBlur(alpha, (self.blur_ksize, self.blur_ksize), 0)

        sure_fg = (alpha >= self.fg_threshold).astype(np.uint8)
        sure_bg = (alpha <= self.bg_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.unknown_width, self.unknown_width)
        )
        fg_core = cv2.erode(sure_fg, kernel, iterations=1)
        bg_core = cv2.erode(sure_bg, kernel, iterations=1)

        trimap = np.full_like(alpha, 0.5, dtype=np.float32)
        trimap[bg_core == 1] = 0.0
        trimap[fg_core == 1] = 1.0
        return trimap

    def _trimap_to_two_chan(self, trimap: torch.Tensor) -> torch.Tensor:
        fg = (trimap == 1.0).float()
        bg = (trimap == 0.0).float()
        return torch.cat([fg, bg], dim=1)

    def _trimap_distance_features(self, trimap: torch.Tensor) -> torch.Tensor:
        """
        Recreates the 6-channel distance transform used in the FBA paper:
        two signed distance maps (FG/BG), their complements, and two
        indicator masks for unknown / foreground regions.
        """
        trimap_np = trimap.squeeze().cpu().numpy()
        fg_mask = (trimap_np == 1.0).astype(np.uint8)
        bg_mask = (trimap_np == 0.0).astype(np.uint8)
# ...
        unk_mask = 1 - fg_mask - bg_mask

        # FIX: Use (mask == 0) to measure distance TO the region
        # We want distance from Unknown pixels TO the Foreground (so FG must be 0)
       # --- FIX START ---
        # Create the binary masks
        fg_input = (fg_mask == 0).astype(np.uint8)
        bg_input = (bg_mask == 0).astype(np.uint8)

       # --- SMART SHAPE FIX START ---
        # Helper to force (H, W) shape regardless of input format
        def force_spatial_shape(arr):
            # If already 2D, return it
            if arr.ndim == 2:
                return arr
            # If 3D, find which dimensions are the spatial ones (256x256)
            # We assume spatial dims are larger than channel dims (usually < 4)
            shape = arr.shape
            if shape[0] < shape[1] and shape[0] < shape[2]: 
                # Case: (C, H, W) -> (2, 256, 256) -> Take channel 0
                return arr[0, :, :]
            elif shape[2] < shape[0] and shape[2] < shape[1]:
                # Case: (H, W, C) -> (256, 256, 2) -> Take channel 0
                return arr[:, :, 0]
            else:
                # Fallback: standard channel-last
                return arr[:, :, 0]

        fg_input = force_spatial_shape(fg_input)
        bg_input = force_spatial_shape(bg_input)

        # Compute distance transforms (Now guaranteed to be 256x256)
        dist_fg = cv2.distanceTransform(fg_input, cv2.DIST_L2, 5)
        dist_bg = cv2.distanceTransform(bg_input, cv2.DIST_L2, 5)
        # --- SMART SHAPE FIX END ---
       
        # ... (after calculating dist_fg and dist_bg with cv2) ...

        # 1. DELETE the lines that added [..., np.newaxis]. 
        # We want dist_fg and dist_bg to remain 2D (Height, Width).

        # 2. Ensure the MASKS are also 2D (Height, Width)
        # If they are (H, W, 1) or (H, W, 3), this flattens them.
        if unk_mask.ndim == 3:
            unk_mask = unk_mask[:, :, 0]
        if fg_mask.ndim == 3:
            fg_mask = fg_mask[:, :, 0]

        # 3. Now everything is 2D, so they can stack perfectly on axis=0
        # 3. Now everything is 2D, so they can stack perfectly on axis=0
        feats = np.stack(
            [
                dist_fg,
                dist_bg,
                1.0 - dist_fg,
                1.0 - dist_bg,
                unk_mask.astype(np.float32),
                fg_mask.astype(np.float32),
            ],
            axis=0,
        )

        feats = torch.from_numpy(feats).unsqueeze(0).float()
        return feats.to(self.device)

    def _respect_trimap(self, alpha: torch.Tensor, trimap: torch.Tensor) -> torch.Tensor:
        alpha = alpha.clone()
        alpha[trimap == 1.0] = 1.0
        alpha[trimap == 0.0] = 0.0
        return torch.clamp(alpha, 0.0, 1.0)

    def _prepare_alpha_np(self, alpha: np.ndarray) -> np.ndarray:
        if alpha.ndim == 4:
            alpha = alpha[0, :, :, 0]
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _ensure_odd_int(value: int) -> int:
        return value if value % 2 == 1 else value + 1