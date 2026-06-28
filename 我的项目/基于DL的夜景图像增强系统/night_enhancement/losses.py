from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights


class VGGPerceptual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.features = vgg.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size
        self.avg_pool = nn.AvgPool2d(window_size, stride=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = self.avg_pool(x)
        mu_y = self.avg_pool(y)
        sigma_x = self.avg_pool(x * x) - mu_x * mu_x
        sigma_y = self.avg_pool(y * y) - mu_y * mu_y
        sigma_xy = self.avg_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim_map = ssim_n / (ssim_d + 1e-6)
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, perceptual_weight: float = 0.2, ssim_weight: float = 0.5, color_weight: float = 1.0) -> None:
        super().__init__()
        self.vgg = VGGPerceptual()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pred_features = self.vgg(prediction)
        target_features = self.vgg(target)
        perceptual_loss = self.l1(pred_features, target_features)
        ssim_loss = self.ssim(prediction, target)
        color_loss = self.l1(prediction, target)
        total = (
            self.perceptual_weight * perceptual_loss
            + self.ssim_weight * ssim_loss
            + self.color_weight * color_loss
        )
        return total, {
            "perceptual": float(perceptual_loss.detach().item()),
            "ssim": float(ssim_loss.detach().item()),
            "color": float(color_loss.detach().item()),
            "total": float(total.detach().item()),
        }
