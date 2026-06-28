from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import pair_image_paths

# 保持宽高比的缩放 + 边缘填充，确保输出固定尺寸，不扭曲图像
class ResizePad:
    def __init__(self, size: Tuple[int, int], pad_value: Tuple[int, int, int] = (0, 0, 0)) -> None:
        self.target_h, self.target_w = size
        self.pad_value = pad_value # 填充颜色 (B, G, R)，默认黑色

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]

        # 计算缩放比例，取较小边适配目标尺寸
        scale = min(self.target_w / w, self.target_h / h)
        # 为什么取 min？保证缩放后图像不会超出目标尺寸，同时至少有一边能贴合目标边。
        # 例如原图 800×600，目标 400×400，宽缩放比 0.5，高缩放比 0.666，取 0.5 缩放后为 400×300，恰好宽度撑满，高度留白。

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA) # 缩放图像，缩小用 INTER_AREA 效果好，避免锯齿和摩尔纹

        # 创建目标尺寸的画布，用 pad_value 填充
        canvas = np.full((self.target_h, self.target_w, 3), self.pad_value, dtype=np.uint8)
        top = (self.target_h - new_h) // 2
        left = (self.target_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

# 对低光/正常光图像同步进行随机翻转、旋转、亮度微调，增强泛化能力
class RandomPairedAugment:
    def __init__(self, output_size: Tuple[int, int]) -> None:
        self.resize_pad = ResizePad(output_size)

    # 接受已经标准化为 [0,1] 浮点 RGB 格式的低光图和高清图，返回增强后的同类数据
    def __call__(self, low_rgb: np.ndarray, high_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 随机水平/垂直翻转、旋转90度的倍数
        # 对低光图额外增加亮度和对比度的小扰动（模拟不同曝光）

        # 随机水平翻转，50%概率
        # axis = 1 表示沿宽度方向翻转（左右镜像）
        # .copy()确保不共享底层数据，避免修改原数组
        if np.random.rand() < 0.5:
            low_rgb = np.flip(low_rgb, axis=1).copy()
            high_rgb = np.flip(high_rgb, axis=1).copy()

        # 随机垂直翻转，30%概率
        # axis = 0 表示沿高度方向翻转（上下镜像）
        # 垂直翻转在真实场景中较少出现，但仍能轻微提升鲁棒性
        if np.random.rand() < 0.3:
            low_rgb = np.flip(low_rgb, axis=0).copy()
            high_rgb = np.flip(high_rgb, axis=0).copy()

        # 随机90°倍数旋转，50% 概率
        if np.random.rand() < 0.5:
            k = int(np.random.choice([0, 1, 2, 3]))
            low_rgb = np.rot90(low_rgb, k).copy()
            high_rgb = np.rot90(high_rgb, k).copy()

        alpha = 0.9 + np.random.rand() * 0.2   # alpha ∈ [0.9, 1.1)，模拟曝光变化（乘性增益），使亮度在 0.9 倍到 1.1 倍之间波动
        beta = (np.random.rand() - 0.5) * 0.08   # beta  ∈ [-0.04, 0.04)，模拟环境光偏移（加性偏移），范围为 -0.04 ~ 0.04
        low_rgb = np.clip(low_rgb * alpha + beta, 0.0, 1.0)   # 仅对低光图施加，因为低光图像本身曝光不足、噪声大，这种小扰动可以让模型学会处理不同低光程度的场景，而不是简单记忆亮度映射

        # 对图像做最终尺寸确认，并返回
        low_rgb = self._ensure_size(low_rgb)
        high_rgb = self._ensure_size(high_rgb)
        return low_rgb, high_rgb

    # 辅助方法，虽然翻转、旋转本身不改变尺寸，但 90° 旋转会使宽高对调，例如原 400×600 变成 600×400。保证最终输出统一
    # 步骤就是先恢复，再归一化：
    # 将已经归一化的反归一化回[0,255]，转为 OpenCV 可处理的 uint8 数据类型
    # RGB -> BGR
    # 归一化，BGR -> RGB，返回
    def _ensure_size(self, image_rgb: np.ndarray) -> np.ndarray:
        image_bgr = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        image_bgr = self.resize_pad(image_bgr)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# 主数据集类
class LowLightDataset(Dataset):
    # 配对文件名相同的图像 (通过 utils.pair_image_paths)
    # 初始化 ResizePad 和可选的 RandomPairedAugment
    def __init__(self, low_dir: Path, high_dir: Path, image_size: Tuple[int, int], augment: bool = False) -> None:
        self.pairs = pair_image_paths(low_dir, high_dir)
        if not self.pairs:
            raise RuntimeError(f"No paired images found in {low_dir} and {high_dir}.")
        self.resize = ResizePad(image_size)
        self.augment = RandomPairedAugment(image_size) if augment else None

    def __len__(self) -> int:
        return len(self.pairs)

    # 用 OpenCV 读取 BGR 图像，转为 RGB，归一化到 [0,1]
    # 应用 ResizePad
    # 返回归一化后的rgb图像，ndarray
    @staticmethod
    def _load_image(path: Path, resize: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image_bgr = resize(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float32) / 255.0
        return image_rgb

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]: # 返回一个字典键值对：[字符串，张量或字符串]
        low_path, high_path = self.pairs[index]
        low_rgb = self._load_image(low_path, self.resize)
        high_rgb = self._load_image(high_path, self.resize)

        # 如果是训练模式，应用同步增强
        if self.augment is not None:
            low_rgb, high_rgb = self.augment(low_rgb, high_rgb)

        # 转换为 PyTorch 张量，并调整维度顺序为 (C, H, W)
        # numpy 图像形状为 (H, W, C)，PyTorch 卷积层要求输入为 (C, H, W)
        low_tensor = torch.from_numpy(low_rgb).permute(2, 0, 1).float()
        high_tensor = torch.from_numpy(high_rgb).permute(2, 0, 1).float()

        return {
            "low": low_tensor,
            "high": high_tensor,
            "name": low_path.name,
        }
