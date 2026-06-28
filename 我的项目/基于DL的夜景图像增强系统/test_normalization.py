from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from night_enhancement.config import TrainConfig
from night_enhancement.dataset import ResizePad


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = PROJECT_ROOT / "test_pic" / "1.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "normalization_test"


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def add_title(image_rgb: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    image_bgr = cv2.cvtColor(to_uint8_rgb(image_rgb), cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    title_bar_h = 72
    canvas = np.full((h + title_bar_h, w, 3), 245, dtype=np.uint8)
    canvas[title_bar_h:, :] = image_bgr
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def fit_for_grid(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    image_uint8 = to_uint8_rgb(image_rgb)
    return cv2.resize(image_uint8, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_channel_panel(tensor_chw: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    channels = tensor_chw.detach().cpu().numpy()
    target_h, target_w = size
    channel_images = []
    names = ["R", "G", "B"]
    colors = [(255, 0, 0), (0, 180, 0), (0, 80, 255)]

    for index, name in enumerate(names):
        gray = (np.clip(channels[index], 0.0, 1.0) * 255).astype(np.uint8)
        gray = cv2.resize(gray, (target_w // 3, target_h), interpolation=cv2.INTER_AREA)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.putText(color, f"{name} channel", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[index], 2, cv2.LINE_AA)
        channel_images.append(color)

    panel = np.hstack(channel_images)
    if panel.shape[1] != target_w:
        panel = cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return panel


def build_visualization(
    original_bgr: np.ndarray,
    rgb: np.ndarray,
    resized_rgb: np.ndarray,
    normalized_rgb: np.ndarray,
    tensor_chw: torch.Tensor,
) -> np.ndarray:
    cell_size = (300, 450)
    bgr_as_rgb = fit_for_grid(original_bgr, cell_size)
    original_rgb = fit_for_grid(rgb, cell_size)
    resized_show = fit_for_grid(resized_rgb, cell_size)
    normalized_show = fit_for_grid(normalized_rgb, cell_size)
    channel_panel = make_channel_panel(tensor_chw, cell_size)

    panels = [
        add_title(
            original_rgb,
            "Original image (RGB view)",
            f"before: BGR shape={original_bgr.shape}, dtype={original_bgr.dtype}",
        ),
        add_title(
            bgr_as_rgb,
            "BGR interpreted as RGB",
            "shows why BGR -> RGB conversion is required",
        ),
        add_title(
            resized_show,
            "Fixed size after ResizePad",
            f"after resize: HWC shape={resized_rgb.shape}",
        ),
        add_title(
            normalized_show,
            "Normalized to [0, 1]",
            f"dtype={normalized_rgb.dtype}, min={normalized_rgb.min():.4f}, max={normalized_rgb.max():.4f}",
        ),
        add_title(
            channel_panel,
            "Dimension conversion HWC -> CHW",
            f"tensor shape={tuple(tensor_chw.shape)}, dtype={tensor_chw.dtype}",
        ),
    ]

    blank = np.full_like(panels[0], 245)
    row1 = np.hstack(panels[:3])
    row2 = np.hstack([panels[3], panels[4], blank])
    return np.vstack([row1, row2])


def preprocess_for_test(image_path: Path, image_size: tuple[int, int]) -> dict[str, np.ndarray | torch.Tensor]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resize = ResizePad(image_size)
    resized_bgr = resize(image_bgr)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    normalized_rgb = resized_rgb.astype(np.float32) / 255.0
    tensor_chw = torch.from_numpy(normalized_rgb).permute(2, 0, 1).float()

    return {
        "image_bgr": image_bgr,
        "image_rgb": image_rgb,
        "resized_rgb": resized_rgb,
        "normalized_rgb": normalized_rgb,
        "tensor_chw": tensor_chw,
    }


def print_summary(image_path: Path, result: dict[str, np.ndarray | torch.Tensor]) -> None:
    image_bgr = result["image_bgr"]
    image_rgb = result["image_rgb"]
    resized_rgb = result["resized_rgb"]
    normalized_rgb = result["normalized_rgb"]
    tensor_chw = result["tensor_chw"]

    assert isinstance(image_bgr, np.ndarray)
    assert isinstance(image_rgb, np.ndarray)
    assert isinstance(resized_rgb, np.ndarray)
    assert isinstance(normalized_rgb, np.ndarray)
    assert isinstance(tensor_chw, torch.Tensor)

    print("图像标准化处理测试")
    print(f"输入图像: {image_path}")
    print(f"1. OpenCV读取BGR: shape={image_bgr.shape}, dtype={image_bgr.dtype}, range=[{image_bgr.min()}, {image_bgr.max()}]")
    print(f"2. BGR转换RGB: shape={image_rgb.shape}, dtype={image_rgb.dtype}, range=[{image_rgb.min()}, {image_rgb.max()}]")
    print(f"3. 固定尺寸ResizePad: shape={resized_rgb.shape}, dtype={resized_rgb.dtype}, range=[{resized_rgb.min()}, {resized_rgb.max()}]")
    print(f"4. 归一化[0,1]: shape={normalized_rgb.shape}, dtype={normalized_rgb.dtype}, range=[{normalized_rgb.min():.6f}, {normalized_rgb.max():.6f}]")
    print(f"5. 维度转换HWC->CHW: shape={tuple(tensor_chw.shape)}, dtype={tensor_chw.dtype}, range=[{tensor_chw.min().item():.6f}, {tensor_chw.max().item():.6f}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test image normalization pipeline independently.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="待测试图片路径，默认使用 test_pic/1.png")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="可视化结果保存目录")
    parser.add_argument("--no-show", action="store_true", help="只保存结果，不弹出OpenCV可视化窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = preprocess_for_test(image_path, config.image_size)
    print_summary(image_path, result)

    visualization = build_visualization(
        result["image_bgr"],
        result["image_rgb"],
        result["resized_rgb"],
        result["normalized_rgb"],
        result["tensor_chw"],
    )
    assert isinstance(visualization, np.ndarray)

    output_path = output_dir / f"{image_path.stem}_normalization_compare.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"可视化对比结果已保存: {output_path}")

    if not args.no_show:
        cv2.imshow("Image normalization test", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print("按任意键关闭可视化窗口。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
