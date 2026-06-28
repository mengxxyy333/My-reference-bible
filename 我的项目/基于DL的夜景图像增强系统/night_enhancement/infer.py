from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .config import TrainConfig
from .dataset import ResizePad
from .models import RetinexEnhancementNet
from .utils import ensure_dir


class EnhancementInferencer:
    def __init__(self, checkpoint_path: Optional[Path] = None, config: Optional[TrainConfig] = None) -> None:
        self.config = config or TrainConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.device == "cuda" else "cpu")
        self.model = RetinexEnhancementNet(freeze_encoder=False).to(self.device)
        self.model.eval()
        self.resize = ResizePad(self.config.image_size)
        self.checkpoint_path = checkpoint_path or (self.config.checkpoints_dir / "latest.pth")
        self._load_checkpoint(self.checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def preprocess(self, image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"无法读取图像: {image_path}")
        image_bgr = self.resize(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return image_rgb, tensor

    @torch.no_grad()
    def enhance(self, image_path: Path) -> dict[str, np.ndarray]:
        original_rgb, tensor = self.preprocess(image_path)
        outputs = self.model(tensor)
        enhanced_rgb = outputs["enhanced"][0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        illumination = outputs["illumination"][0, 0].cpu().numpy()
        reflectance = outputs["reflectance"][0].permute(1, 2, 0).cpu().numpy()
        return {
            "original_rgb": original_rgb,
            "enhanced_rgb": enhanced_rgb,
            "illumination": illumination,
            "reflectance": reflectance,
        }

    def export(self, image_path: Path, enhanced_rgb: np.ndarray, output_dir: Path) -> Path:
        ensure_dir(output_dir)
        output_path = output_dir / f"{image_path.stem}_enhanced.png"
        enhanced_bgr = cv2.cvtColor((enhanced_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), enhanced_bgr)
        return output_path
