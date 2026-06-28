from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..config import SUPPORTED_IMAGE_EXTENSIONS

try:
    import lpips
except Exception:
    lpips = None

_LPIPS_MODEL = None


def numpy_rgb_to_qpixmap(image_rgb: np.ndarray) -> QPixmap:
    image_uint8 = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    image_uint8 = np.ascontiguousarray(image_uint8)
    h, w, ch = image_uint8.shape
    bytes_per_line = ch * w
    qimage = QImage(image_uint8.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def create_histogram_image(image_rgb: np.ndarray, width: int = 512, height: int = 240) -> np.ndarray:
    gray = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / max(hist.max(), 1.0)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (239, 236, 231)
    left, right, top, bottom = 16, 12, 10, 28
    plot_w = width - left - right
    plot_h = height - top - bottom
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (189, 181, 171), 1)
    bar_w = max(plot_w / 256.0, 1.0)
    for idx, value in enumerate(hist):
        x0 = int(left + idx * bar_w)
        x1 = int(left + (idx + 1) * bar_w)
        if x1 <= x0:
            x1 = x0 + 1
        bar_h = int(value * (plot_h - 2))
        y0 = top + plot_h - bar_h
        y1 = top + plot_h
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1), (126, 121, 114), -1)

    tick_values = [0, 64, 128, 192, 255]
    for tick in tick_values:
        x = left + int(tick / 255 * plot_w)
        cv2.line(canvas, (x, top + plot_h), (x, top + plot_h + 5), (150, 143, 136), 1)
        label = str(tick)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, label, (x - label_size[0] // 2, top + plot_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (92, 86, 80), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def create_metric_chart(history: list[dict], width: int = 760, height: int = 260) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (244, 242, 238)
    if not history:
        return canvas.astype(np.float32) / 255.0
    left, right, top, bottom = 56, 20, 24, 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    train_vals = [float(item.get("train_loss", 0.0)) for item in history]
    val_vals = [float(item.get("val_l1", 0.0)) for item in history]
    y_max = max(max(train_vals), max(val_vals), 1e-6)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (191, 184, 174), 1)
    for i in range(5):
        y = top + int(i * plot_h / 4)
        cv2.line(canvas, (left, y), (left + plot_w, y), (225, 220, 213), 1)

    def draw_series(values: list[float], color: tuple[int, int, int]) -> None:
        pts = []
        for idx, value in enumerate(values):
            x = left + int(idx * plot_w / max(1, len(values) - 1))
            y = top + plot_h - int(value / y_max * plot_h)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 2)
        for p in pts:
            cv2.circle(canvas, p, 3, color, -1)

    draw_series(train_vals, (122, 116, 108))
    draw_series(val_vals, (160, 152, 142))
    cv2.putText(canvas, "Train Loss", (left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (98, 92, 86), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Val L1", (left + 130, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (132, 124, 116), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def create_metric_detail_image(metric_name: str, reference_value: float, enhanced_value: float, detail_text: str, width: int = 540, height: int = 240, reference_label: str = "Reference", enhanced_label: str = "Enhanced") -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (244, 242, 238)
    left, right, top, bottom = 50, 30, 48, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max(float(reference_value), float(enhanced_value), 1e-6)
    bar_w = 96
    x1 = left + plot_w // 4 - bar_w // 2
    x2 = left + plot_w * 3 // 4 - bar_w // 2
    cv2.putText(canvas, metric_name, (24, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (92, 86, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, detail_text, (24, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (99, 93, 87), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (191, 184, 174), 1)
    for i in range(5):
        y = top + int(i * plot_h / 4)
        cv2.line(canvas, (left, y), (left + plot_w, y), (225, 220, 213), 1)

    def draw_bar(x: int, value: float, color: tuple[int, int, int], label: str) -> None:
        h = int(max(0.0, value) / max_val * (plot_h - 8))
        y = top + plot_h - h
        cv2.rectangle(canvas, (x, y), (x + bar_w, top + plot_h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + bar_w, top + plot_h), (150, 143, 136), 1)
        cv2.putText(canvas, f"{value:.4f}", (x - 2, max(top + 18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (78, 73, 67), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 4, top + plot_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (92, 86, 80), 1, cv2.LINE_AA)

    draw_bar(x1, reference_value, (178, 169, 159), reference_label)
    draw_bar(x2, enhanced_value, (131, 124, 116), enhanced_label)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def compute_psnr(reference_rgb: np.ndarray, image_rgb: np.ndarray) -> float:
    ref = np.clip(reference_rgb.astype(np.float32), 0.0, 1.0)
    img = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
    mse = float(np.mean((ref - img) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(reference_rgb: np.ndarray, image_rgb: np.ndarray) -> float:
    ref_gray = cv2.cvtColor((np.clip(reference_rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    img_gray = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(ref_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img_gray, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(ref_gray * ref_gray, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img_gray * img_gray, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(ref_gray * img_gray, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)
    return float(np.mean(ssim_map))


def compute_smd2(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]
    h = min(dx.shape[0], dy.shape[0])
    w = min(dx.shape[1], dy.shape[1])
    if h == 0 or w == 0:
        return 0.0
    return float(np.mean(np.abs(dx[:h, :w] * dy[:h, :w])))


def _get_lpips_model():
    global _LPIPS_MODEL
    if lpips is None:
        return None
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex")
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


def compute_lpips(reference_rgb: np.ndarray, image_rgb: np.ndarray) -> float:
    model = _get_lpips_model()
    ref = np.clip(reference_rgb.astype(np.float32), 0.0, 1.0)
    img = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
    if model is None:
        return float(np.mean(np.abs(ref - img)))
    ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        value = model(ref_tensor, img_tensor)
    return float(value.squeeze().item())


def supported_image_filters() -> str:
    return "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"


def list_supported_images(directory: Path) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])


class ZoomableImageView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#ece7e1"))
        self._zoom = 0

    def has_image(self) -> bool:
        return not self._pixmap_item.pixmap().isNull()

    def fit_image(self) -> None:
        if not self.has_image():
            self.resetTransform()
            self._zoom = 0
            return
        self.resetTransform()
        self._zoom = 0
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(self._pixmap_item.boundingRect())
        self.fit_image()

    def clear_image(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self.scene().setSceneRect(self._pixmap_item.boundingRect())
        self.resetTransform()
        self._zoom = 0

    def wheelEvent(self, event) -> None:
        if self._pixmap_item.pixmap().isNull():
            return super().wheelEvent(event)
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom += 1 if event.angleDelta().y() > 0 else -1
        if self._zoom < -8:
            self._zoom = -8
            return
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.has_image():
            self.fit_image()
        super().mouseDoubleClickEvent(event)
