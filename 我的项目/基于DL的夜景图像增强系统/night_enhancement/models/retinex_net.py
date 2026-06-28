from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights

# 双层卷积 + ReLU，基本构建块
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# 上采样块：转置卷积 + 跳跃连接拼接 + ConvBlock
# 处理尺寸不匹配时用双线性插值微调
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# 专门处理瓶颈层特征，增强光照分量
class IlluminationEnhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# 最后几层，将解码特征映射回 3 通道 RGB，输出用 Sigmoid 约束在 [0,1]
class DetailRestorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(x)

class RetinexEnhancementNet(nn.Module):
    def __init__(self, freeze_encoder: bool = True) -> None:
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # 加载预训练 VGG16，截取前几层作为编码器：
        #    encoder1: conv1_1, conv1_2 (输出64通道)
        #    encoder2: conv2_1, conv2_2 (输出128通道)
        #    encoder3: conv3_1, conv3_2, conv3_3 (输出256通道)
        self.encoder1 = nn.Sequential(*features[:4])
        self.pool1 = features[4]
        self.encoder2 = nn.Sequential(*features[5:9])
        self.pool2 = features[9]
        self.encoder3 = nn.Sequential(*features[10:16])

        # 瓶颈层: ConvBlock(256,256)
        self.bottleneck = ConvBlock(256, 256)

        # 光照增强分支: IlluminationEnhancer (256→128)
        self.illumination = IlluminationEnhancer()

        # 解码器: UpBlock(128,128,128) + UpBlock(128,64,64)
        self.decode2 = UpBlock(128, 128, 128)
        self.decode1 = UpBlock(128, 64, 64)

        # 细节恢复: DetailRestorer (64→3)
        self.detail = DetailRestorer()

        if freeze_encoder:
            self.freeze_shallow_encoder()

    def freeze_shallow_encoder(self) -> None:
        for module in (self.encoder1, self.encoder2):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # 编码路径，保存 e1, e2 用于跳跃连接
        input_size = x.shape[-2:]
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)

        # 解码路径
        bottleneck = self.bottleneck(e3)
        illum = self.illumination(bottleneck) # 增强后的光照特征
        d2 = self.decode2(illum, e2) # 上采样并与 e2 拼接
        d1 = self.decode1(d2, e1)  # 上采样并与 e1 拼接
        enhanced = self.detail(d1) # 输出增强图像

        if enhanced.shape[-2:] != input_size:
            enhanced = nn.functional.interpolate(
                enhanced,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        # 计算光照图和反射图（符合 Retinex 理论：S = R * I）
        illumination_map = torch.mean(illum, dim=1, keepdim=True)
        if illumination_map.shape[-2:] != input_size:
            illumination_map = nn.functional.interpolate(
                illumination_map,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        reflectance_map = torch.clamp(enhanced / (illumination_map + 1e-4), 0.0, 1.0)

        return {
            "enhanced": enhanced,
            "illumination": illumination_map,
            "reflectance": reflectance_map,
        }
