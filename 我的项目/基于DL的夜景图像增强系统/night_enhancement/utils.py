from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from .config import SUPPORTED_IMAGE_EXTENSIONS

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def set_seed(seed: int) -> None:
    # 固定随机种子，是的每次调用随机函数生成数据相同，使得实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_images(directory: Path) -> list[Path]:
    return sorted(
        # 筛文件，满足两个条件：1、是文件而不是目录；2、后缀需在系统支持的格式内
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    )

def pair_image_paths(low_dir: Path, high_dir: Path) -> list[tuple[Path, Path]]:
    low_files = {path.name: path for path in list_images(low_dir)}
    high_files = {path.name: path for path in list_images(high_dir)}
    shared_names = sorted(set(low_files) & set(high_files)) # 取出同名配对图像，排序后返回，按位与筛掉未配对成功的图片
    return [(low_files[name], high_files[name]) for name in shared_names]

def tensor_to_bgr_image(tensor: torch.Tensor) -> np.ndarray:
    # detach:从计算图中分离，不再记录梯度。因为转换图像仅用于保存或显示，不需要反向传播
    # cpu:将张量从 GPU 移动到 CPU，因为后续的 .numpy() 无法直接在 CUDA 张量上调用
    image = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() 
    image = (image * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB -> BGR

def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), tensor_to_bgr_image(tensor))

def brightness_histogram(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # 计算图像直方图，直方图统计了图像中每个像素值出现的频次
    # 参数：传入灰度图像，单通道，不使用掩码，像素值分为256个区域，像素值的范围在[0, 256)
    # flatten的作用是将其展平为一维数组
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    bins = np.arange(256) # 生成一个[0, 255]的数组，作为直方图的横坐标
    return bins, hist

def collate_file_names(paths: Iterable[Path]) -> list[str]:
    return [path.name for path in paths]
