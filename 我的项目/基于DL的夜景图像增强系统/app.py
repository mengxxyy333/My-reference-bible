from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from night_enhancement.config import TrainConfig
from night_enhancement.infer import EnhancementInferencer
from night_enhancement.ui.helpers import (
    ZoomableImageView,
    compute_lpips,
    compute_psnr,
    compute_smd2,
    compute_ssim,
    create_histogram_image,
    create_metric_chart,
    create_metric_detail_image,
    list_supported_images,
    numpy_rgb_to_qpixmap,
    supported_image_filters,
)
from night_enhancement.utils import ensure_dir
from train import Trainer


class TrainingWorker(QThread):
    event_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config: TrainConfig, resume: bool = True):
        super().__init__()
        self.config = config
        self.resume = resume

    def run(self) -> None:
        try:
            Trainer(
                self.config,
                progress_callback=self.event_signal.emit,
                resume=self.resume,
            ).train()
        except Exception as exc:
            self.error_signal.emit(str(exc))


class EnhancementWorker(QThread):
    result_signal = pyqtSignal(str, dict)
    error_signal = pyqtSignal(str)

    def __init__(self, inferencer: EnhancementInferencer, image_key: str, image_path: Path):
        super().__init__()
        self.inferencer = inferencer
        self.image_key = image_key
        self.image_path = image_path

    def run(self) -> None:
        try:
            self.result_signal.emit(self.image_key, self.inferencer.enhance(self.image_path))
        except Exception as exc:
            self.error_signal.emit(str(exc))


class NightEnhancementWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = TrainConfig()
        self.inferencer: EnhancementInferencer | None = None
        self.training_worker: TrainingWorker | None = None
        self.enhancement_worker: EnhancementWorker | None = None
        self.low_images: dict[str, Path] = {}
        self.enhanced_results: dict[str, dict] = {}
        self.current_key: str | None = None
        self.active_preview_epoch: int | None = None
        self.epoch_inferencers: dict[int, EnhancementInferencer] = {}
        self.history: list[dict] = []
        self.preview_map = self._build_preview_map()
        self.metric_buttons: dict[str, QPushButton] = {}
        self.metric_values: dict[str, float] = {}
        self.metric_detail_views: dict[str, ZoomableImageView] = {}
        self.active_metric_key: str = "psnr"

        self._build_ui()
        self._apply_style()
        self.chart_view.clear_image()
        self._sync_preview_buttons()
        self._show_pending_enhanced_state(None)
        self._clear_metric_views()

    def _build_preview_map(self) -> dict[int, list[Path]]:
        out: dict[int, list[Path]] = {}
        if self.config.progress_dir.exists():
            for path in sorted(self.config.progress_dir.glob("epoch_*.png")):
                parts = path.stem.split("_")
                if len(parts) >= 3:
                    out.setdefault(int(parts[1]), []).append(path)
        return out

    def _named_panel(self, title, view):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 18, 8, 8)
        lay.addWidget(view)
        return box

    def _create_metric_button(self, key: str, title: str):
        button = QPushButton(f"{title}\n--")
        button.setProperty("metricButton", True)
        button.setCheckable(True)
        button.clicked.connect(lambda _=False, metric_key=key: self._set_active_metric(metric_key))
        return button

    def _build_ui(self):
        self.setWindowTitle("夜景图像增强系统"); self.resize(1460, 930)
        self.open_file_btn = QPushButton("导入单张"); self.open_folder_btn = QPushButton("导入文件夹"); self.load_model_btn = QPushButton("加载模型"); self.fit_view_btn = QPushButton("重置视图")
        self.enhance_btn = QPushButton("增强当前图像"); self.export_btn = QPushButton("导出当前结果"); self.export_all_btn = QPushButton("批量导出"); self.normal_compare_btn = QPushButton("加入正常光图片对比"); self.clear_log_btn = QPushButton("清空控制台")
        self.input_combo = QComboBox(); self.output_combo = QComboBox(); self.device_label = QLabel("未加载模型"); self.checkpoint_label = QLabel("latest.pth"); self.summary_label = QLabel("请先加载模型并导入图像"); self.enhanced_hint_label = QLabel("请先导入图像并加载模型")
        self.original_view = ZoomableImageView(); self.enhanced_view = ZoomableImageView(); self.hist_original = ZoomableImageView(); self.hist_enhanced = ZoomableImageView(); self.chart_view = ZoomableImageView()
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 500); self.epochs_spin.setValue(self.config.epochs)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 32); self.batch_spin.setValue(self.config.batch_size)
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setDecimals(6); self.lr_spin.setRange(0.000001, 0.01); self.lr_spin.setValue(self.config.learning_rate)
        self.train_btn = QPushButton("开始训练"); self.resume_btn = QPushButton("继续训练"); self.reload_history_btn = QPushButton("加载训练日志")
        self.metrics_text = QPlainTextEdit(); self.metrics_text.setReadOnly(True); self.metrics_text.setMaximumBlockCount(400); self.progress = QProgressBar(); self.progress.setRange(0, 100)

        c = QWidget(); root = QHBoxLayout(c); root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)

        left = QWidget(); ll = QVBoxLayout(left); ll.setSpacing(8)
        loss_box = QGroupBox("训练损失曲线")
        loss_lay = QVBoxLayout(loss_box)
        loss_lay.addWidget(self.chart_view)
        ll.addWidget(loss_box, 1)
        hist_box = QGroupBox("亮度直方图对比")
        hist_lay = QVBoxLayout(hist_box)
        hist_lay.addWidget(QLabel("原始图片亮度直方图"))
        hist_lay.addWidget(self.hist_original, 1)
        hist_lay.addWidget(QLabel("增强图片亮度直方图"))
        hist_lay.addWidget(self.hist_enhanced, 1)
        ll.addWidget(hist_box, 1)

        center = QWidget()
        center.setMaximumWidth(590)
        cl = QVBoxLayout(center)
        cl.setSpacing(6)

        ob = QGroupBox("原始图像预览")
        ol = QVBoxLayout(ob)
        ol.setSpacing(3)
        original_actions = QHBoxLayout()
        original_actions.setSpacing(4)
        original_actions.addWidget(self.open_file_btn)
        original_actions.addWidget(self.open_folder_btn)
        original_actions.addWidget(self.load_model_btn)
        original_actions.addWidget(self.fit_view_btn)
        original_actions.addStretch(1)
        ol.addLayout(original_actions)
        ol.addWidget(QLabel("原图列表"))
        ol.addWidget(self.input_combo)
        self.original_view_alignment_spacer = QWidget()
        self.original_view_alignment_spacer.setFixedHeight(4)
        ol.addWidget(self.original_view_alignment_spacer)
        ol.addWidget(self.original_view, 1)

        eb = QGroupBox("增强图像预览")
        el = QVBoxLayout(eb)
        el.setSpacing(4)
        enhance_actions = QHBoxLayout()
        enhance_actions.setSpacing(4)
        enhance_actions.addWidget(self.enhance_btn)
        enhance_actions.addWidget(self.export_btn)
        enhance_actions.addWidget(self.export_all_btn)
        enhance_actions.addWidget(self.normal_compare_btn)
        enhance_actions.addStretch(1)
        el.addLayout(enhance_actions)
        el.addWidget(QLabel("中间轮次结果查看"))
        preview_row = QHBoxLayout(); preview_row.setSpacing(3); self.preview_buttons = {}
        for epoch in [0, 20, 50, 80, 100]:
            btn = QPushButton(str(epoch))
            btn.setMinimumHeight(20)
            btn.setMaximumHeight(22)
            btn.setMinimumWidth(34)
            btn.clicked.connect(lambda _=False, ep=epoch: self.show_epoch_preview(ep))
            self.preview_buttons[epoch] = btn
            preview_row.addWidget(btn, 0)
        preview_row.addStretch(1)
        el.addLayout(preview_row)
        el.addWidget(self.enhanced_hint_label)
        el.addWidget(self.enhanced_view, 1)

        status_box = QGroupBox("模型与运行状态")
        sl = QGridLayout(status_box)
        sl.addWidget(QLabel("设备状态"), 0, 0)
        sl.addWidget(self.device_label, 0, 1)
        sl.addWidget(QLabel("当前摘要"), 1, 0)
        sl.addWidget(self.summary_label, 1, 1)

        cl.addWidget(ob, 5)
        cl.addWidget(eb, 5)
        cl.addWidget(status_box, 1)

        right = QWidget(); rl = QVBoxLayout(right); rl.setSpacing(8)
        metric_box = QGroupBox("增强质量指标")
        metric_lay = QVBoxLayout(metric_box)
        metric_lay.setSpacing(8)
        metric_row = QGridLayout()
        metric_row.setHorizontalSpacing(6)
        metric_row.setVerticalSpacing(6)
        metric_items = [("psnr", "PSNR"), ("ssim", "SSIM"), ("smd2", "SMD2"), ("lpips", "LPIPS")]
        for idx, (key, title) in enumerate(metric_items):
            btn = self._create_metric_button(key, title)
            self.metric_buttons[key] = btn
            metric_row.addWidget(btn, 0, idx)
            detail_view = ZoomableImageView()
            self.metric_detail_views[key] = detail_view
        metric_lay.addLayout(metric_row)
        self.metric_title_label = QLabel("请选择或等待增强结果")
        self.metric_title_label.setProperty("metricTitle", True)
        metric_lay.addWidget(self.metric_title_label)
        self.metric_stack_view = ZoomableImageView()
        metric_lay.addWidget(self.metric_stack_view, 1)
        console_box = QGroupBox("训练控制台")
        console_lay = QVBoxLayout(console_box)
        console_lay.addWidget(self.metrics_text, 1)
        log_actions = QHBoxLayout(); log_actions.setSpacing(4); log_actions.addWidget(self.reload_history_btn); log_actions.addWidget(self.clear_log_btn); log_actions.addStretch(1)
        console_lay.addLayout(log_actions); console_lay.addWidget(self.progress)
        train_box = QGroupBox("训练参数与操作")
        tl = QGridLayout(train_box)
        tl.addWidget(QLabel("Epochs"), 0, 0); tl.addWidget(self.epochs_spin, 0, 1); tl.addWidget(QLabel("Batch"), 0, 2); tl.addWidget(self.batch_spin, 0, 3)
        tl.addWidget(QLabel("LR"), 1, 0); tl.addWidget(self.lr_spin, 1, 1); tl.addWidget(self.train_btn, 1, 2); tl.addWidget(self.resume_btn, 1, 3)
        rl.addWidget(metric_box, 3); rl.addWidget(console_box, 4); rl.addWidget(train_box, 1)

        root.addWidget(left, 2)
        root.addWidget(center, 3)
        root.addWidget(right, 2)
        self.setCentralWidget(c)

        self.open_file_btn.clicked.connect(self.open_file); self.open_folder_btn.clicked.connect(self.open_folder); self.load_model_btn.clicked.connect(self.load_model); self.fit_view_btn.clicked.connect(self.reset_views)
        self.enhance_btn.clicked.connect(self.enhance_current); self.export_btn.clicked.connect(self.export_current); self.export_all_btn.clicked.connect(self.export_all); self.normal_compare_btn.clicked.connect(self.add_normal_light_comparison); self.clear_log_btn.clicked.connect(self.metrics_text.clear)
        self.input_combo.currentTextChanged.connect(self.on_input_selected); self.output_combo.currentTextChanged.connect(self.on_output_selected)
        self.train_btn.clicked.connect(self.start_training); self.resume_btn.clicked.connect(self.resume_training); self.reload_history_btn.clicked.connect(lambda: self.refresh_history_view(initial=False))

    def _apply_style(self):
        self.setFont(QFont("Microsoft YaHei UI", 10)); p = self.palette(); p.setColor(self.backgroundRole(), QColor("#ece9e4")); self.setPalette(p)
        self.setStyleSheet("QMainWindow, QWidget { background:#ece9e4; color:#2f3133; } QGroupBox { border:1px solid #cfc8bf; border-radius:12px; margin-top:10px; padding-top:16px; font-weight:700; background:#f8f7f4; } QGroupBox::title { left:12px; padding:0 6px; color:#67625c; } QPushButton { background:#dfd9d0; border:1px solid #b8afa4; border-radius:6px; padding:2px 6px; color:#2f3133; font-weight:600; min-height:22px; } QPushButton:hover { background:#d3ccc2; border-color:#9e968c; } QPushButton:pressed { background:#c7bfb4; } QPushButton[metricButton='true'] { min-height:56px; text-align:center; padding:6px 8px; } QPushButton[metricButton='true']:checked { background:#bdb1a3; border-color:#8f867b; } QComboBox, QPlainTextEdit, QSpinBox, QDoubleSpinBox { background:#fbfaf8; border:1px solid #c9c2b8; padding:6px; color:#2f3133; border-radius:8px; selection-background-color:#d8d0c5; selection-color:#2f3133; } QProgressBar { background:#e6e1db; border:1px solid #c9c2b8; border-radius:8px; text-align:center; min-height:22px; color:#2f3133; } QProgressBar::chunk { background:#a79d91; border-radius:8px; } QLabel[metricTitle='true'] { color:#7a736a; font-size:12px; font-weight:600; background:transparent; }")
        self.summary_label.setWordWrap(True); self.enhanced_hint_label.setWordWrap(True); self.enhanced_hint_label.setStyleSheet("color:#736d66; padding:4px 2px 0 2px;")
        self.summary_label.setWordWrap(True); self.enhanced_hint_label.setWordWrap(True); self.enhanced_hint_label.setStyleSheet("color:#736d66; padding:4px 2px 0 2px;")

    def log(self, text): self.metrics_text.appendPlainText(text)
    def _set_summary(self, text): self.summary_label.setText(text)
    def _format_metric_value(self, value: float, digits: int = 4) -> str:
        if np.isnan(value) or np.isinf(value):
            return "--"
        return f"{value:.{digits}f}"
    def _clear_metric_views(self) -> None:
        self.metric_values = {}
        metric_titles = {"psnr": "PSNR", "ssim": "SSIM", "smd2": "SMD2", "lpips": "LPIPS"}
        for key, button in self.metric_buttons.items():
            button.setText(f"{metric_titles[key]}\n--")
            button.setChecked(False)
        self.metric_stack_view.clear_image()
        self.metric_title_label.setText("请选择或等待增强结果")
    def _set_active_metric(self, metric_key: str) -> None:
        self.active_metric_key = metric_key
        metric_titles = {"psnr": "PSNR", "ssim": "SSIM", "smd2": "SMD2", "lpips": "LPIPS"}
        for key, button in self.metric_buttons.items():
            button.setChecked(key == metric_key)
        view = self.metric_detail_views.get(metric_key)
        if view is None or not view.has_image():
            self.metric_stack_view.clear_image()
            self.metric_title_label.setText(f"{metric_titles.get(metric_key, metric_key.upper())} 暂无可视化结果")
            return
        self.metric_stack_view.set_image(view._pixmap_item.pixmap())
        value = self.metric_values.get(metric_key)
        value_text = self._format_metric_value(value, 3 if metric_key == "psnr" else 4) if value is not None else "--"
        self.metric_title_label.setText(f"{metric_titles.get(metric_key, metric_key.upper())} 当前值：{value_text}")
    def _calculate_metrics(self, original_rgb: np.ndarray, enhanced_rgb: np.ndarray) -> dict[str, float]:
        return {
            "psnr": compute_psnr(original_rgb, enhanced_rgb),
            "ssim": compute_ssim(original_rgb, enhanced_rgb),
            "smd2": compute_smd2(enhanced_rgb),
            "lpips": compute_lpips(original_rgb, enhanced_rgb),
        }
    def _calculate_reference_metrics(self, reference_rgb: np.ndarray, enhanced_rgb: np.ndarray) -> dict[str, float]:
        return {
            "psnr": compute_psnr(reference_rgb, enhanced_rgb),
            "ssim": compute_ssim(reference_rgb, enhanced_rgb),
            "smd2": compute_smd2(enhanced_rgb),
            "lpips": compute_lpips(reference_rgb, enhanced_rgb),
        }
    def _match_reference_size(self, reference_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
        target_h, target_w = target_rgb.shape[:2]
        if reference_rgb.shape[:2] == (target_h, target_w):
            return reference_rgb
        resized = cv2.resize(reference_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return np.clip(resized, 0.0, 1.0).astype(np.float32)
    def _load_rgb_image(self, path: Path) -> np.ndarray | None:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    def _render_metric_views(self, reference_rgb: np.ndarray, enhanced_rgb: np.ndarray, metrics: dict[str, float], reference_label: str = "Original") -> None:
        metric_config = {
            "psnr": ("PSNR", "Higher usually means closer to reference."),
            "ssim": ("SSIM", "Higher usually means better structural similarity."),
            "smd2": ("SMD2", "Higher usually means stronger detail response."),
            "lpips": ("LPIPS", "Lower usually means better perceptual similarity."),
        }
        reference_metrics = {
            "psnr": compute_psnr(reference_rgb, reference_rgb),
            "ssim": compute_ssim(reference_rgb, reference_rgb),
            "smd2": compute_smd2(reference_rgb),
            "lpips": compute_lpips(reference_rgb, reference_rgb),
        }
        metric_titles = {"psnr": "PSNR", "ssim": "SSIM", "smd2": "SMD2", "lpips": "LPIPS"}
        for key, value in metrics.items():
            digits = 3 if key == "psnr" else 4
            self.metric_values[key] = float(value)
            self.metric_buttons[key].setText(f"{metric_titles[key]}\n{self._format_metric_value(value, digits)}")
            title, detail_text = metric_config[key]
            detail_img = create_metric_detail_image(title, float(reference_metrics[key]), float(value), detail_text, reference_label=reference_label)
            self.metric_detail_views[key].set_image(numpy_rgb_to_qpixmap(detail_img))
        self._set_active_metric(self.active_metric_key if self.active_metric_key in self.metric_buttons else "psnr")
    def _refresh_enhancement_analysis(self, result: dict) -> None:
        original_rgb = result.get("original_rgb")
        enhanced_rgb = result.get("enhanced_rgb")
        if original_rgb is None or enhanced_rgb is None:
            self._clear_metric_views()
            return
        self.hist_enhanced.set_image(numpy_rgb_to_qpixmap(create_histogram_image(enhanced_rgb)))
        metrics = result.get("normal_reference_metrics")
        reference_rgb = result.get("normal_reference_rgb")
        reference_label = "Normal" if metrics is not None and reference_rgb is not None else "Original"
        if metrics is None or reference_rgb is None:
            metrics = result.get("analysis_metrics")
            reference_rgb = original_rgb
        if metrics is None:
            metrics = self._calculate_metrics(original_rgb, enhanced_rgb)
            result["analysis_metrics"] = metrics
        self._render_metric_views(reference_rgb, enhanced_rgb, metrics, reference_label=reference_label)
    def refresh_history_view(self, initial: bool = False):
        path = self.config.logs_dir / "train_history.json"; self.history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        self.chart_view.set_image(numpy_rgb_to_qpixmap(create_metric_chart(self.history))); self.preview_map = self._build_preview_map(); self._sync_preview_buttons(); message = f"已加载训练日志，共 {len(self.history)} 条记录。"; self._set_summary(message)
        if not initial: self.log(message)
    def _sync_preview_buttons(self):
        for epoch, btn in self.preview_buttons.items(): btn.setEnabled(epoch in self.preview_map)
    def build_runtime_config(self):
        cfg = TrainConfig(); cfg.epochs = self.epochs_spin.value(); cfg.batch_size = self.batch_spin.value(); cfg.learning_rate = self.lr_spin.value(); return cfg
    def start_training(self): self._launch_training(False)
    def resume_training(self): self._launch_training(True)
    def _launch_training(self, resume):
        if self.training_worker and self.training_worker.isRunning(): QMessageBox.information(self, "提示", "训练正在进行中。"); return
        cfg = self.build_runtime_config(); self.progress.setValue(0); self.log(f"{'继续训练' if resume else '重新训练'}: epochs={cfg.epochs}, batch={cfg.batch_size}, lr={cfg.learning_rate}"); self._set_summary(f"{'继续训练' if resume else '重新训练'}已启动")
        self.training_worker = TrainingWorker(cfg, resume); self.training_worker.event_signal.connect(self.on_train_event); self.training_worker.error_signal.connect(self.on_train_error); self.train_btn.setEnabled(False); self.resume_btn.setEnabled(False); self.training_worker.finished.connect(lambda: self.train_btn.setEnabled(True)); self.training_worker.finished.connect(lambda: self.resume_btn.setEnabled(True)); self.training_worker.start()
    def on_train_event(self, event):
        stage = event.get("stage")
        if stage in {"status", "log"}: self.log(str(event.get("message", ""))); return
        if stage == "batch":
            e, te, b, tb = int(event["epoch"]), int(event["total_epochs"]), int(event["batch"]), int(event["total_batches"]); self.progress.setValue(int((((e - 1) + b / max(1, tb)) / max(1, te)) * 100)); self.device_label.setText(f"训练中 | Epoch {e}/{te} | Loss {float(event['loss']):.4f}"); self._set_summary(f"训练进行到第 {e} 轮，第 {b}/{tb} 批"); return
        if stage == "epoch":
            m = event.get("metrics", {}); self.history = [i for i in self.history if int(i.get("epoch", -1)) != int(m.get("epoch", -1))]; self.history.append(m); self.history.sort(key=lambda i: int(i.get("epoch", 0))); self.chart_view.set_image(numpy_rgb_to_qpixmap(create_metric_chart(self.history))); self.log(str(event.get("message", ""))); self.preview_map = self._build_preview_map() if event.get("preview_path") else self.preview_map; self._sync_preview_buttons(); self._set_summary(f"训练已完成到第 {int(m.get('epoch', 0))} 轮"); return
        if stage == "finished": self.progress.setValue(100); self.log(str(event.get("message", "训练完成。"))); self.refresh_history_view(initial=True); self.device_label.setText("训练完成，可加载最新模型/最佳模型"); self._set_summary("训练完成，可查看日志与中间轮次预览")
    def on_train_error(self, message): self.train_btn.setEnabled(True); self.resume_btn.setEnabled(True); self.log("训练失败: " + message); self._set_summary("训练失败，请查看控制台日志"); QMessageBox.critical(self, "训练失败", message)
    def _set_original_preview(self, image_key):
        image_path = self.low_images.get(image_key)
        if image_path is None: return False
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None: self.log(f"读取失败: {image_path}"); return False
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.original_view.set_image(numpy_rgb_to_qpixmap(image_rgb)); self.hist_original.set_image(numpy_rgb_to_qpixmap(create_histogram_image(image_rgb))); return True
    def _show_pending_enhanced_state(self, image_key):
        self.enhanced_view.clear_image(); self.hist_enhanced.clear_image(); self._clear_metric_views()
        if image_key and image_key in self.low_images: self.enhanced_hint_label.setText(f"{image_key} 还未增强，请先点击“增强当前图像”。")
        else: self.enhanced_hint_label.setText("请先导入图像并选择要查看的图片")
    def _set_current_image(self, image_key):
        self.current_key = image_key
        self.active_preview_epoch = None
        if not self._set_original_preview(image_key): return
        if image_key in self.enhanced_results: self._show_enhanced_result(image_key); self._set_summary(f"当前查看: {image_key}（已增强）")
        else: self._show_pending_enhanced_state(image_key); self._set_summary(f"当前查看: {image_key}（原图已显示，增强结果未生成）")
    def _show_enhanced_result(self, image_key):
        result = self.enhanced_results.get(image_key)
        if not result: self._show_pending_enhanced_state(image_key); return
        enhanced_rgb = result["enhanced_rgb"]; self.enhanced_view.set_image(numpy_rgb_to_qpixmap(enhanced_rgb)); self._refresh_enhancement_analysis(result); self.enhanced_hint_label.setText(f"{image_key} 的增强结果已显示")
        idx = self.output_combo.findText(image_key)
        if idx >= 0 and self.output_combo.currentIndex() != idx: self.output_combo.setCurrentIndex(idx)
    def _get_epoch_checkpoint_path(self, epoch: int) -> Path | None:
        if epoch == 0: return None
        snapshot_path = self.config.checkpoints_dir / f"retinex_epoch_{epoch:03d}.pth"
        if snapshot_path.exists(): return snapshot_path
        latest_path = self.config.checkpoints_dir / "latest.pth"
        if epoch == self.config.epochs and latest_path.exists(): return latest_path
        return None
    def _get_epoch_inferencer(self, epoch: int) -> EnhancementInferencer | None:
        checkpoint_path = self._get_epoch_checkpoint_path(epoch)
        if checkpoint_path is None: return None
        cached = self.epoch_inferencers.get(epoch)
        if cached is not None and cached.checkpoint_path == checkpoint_path: return cached
        inferencer = EnhancementInferencer(checkpoint_path=checkpoint_path, config=self.config)
        self.epoch_inferencers[epoch] = inferencer
        return inferencer
    def _load_images(self, paths):
        self.low_images = {path.name: path for path in paths}; self.enhanced_results.clear(); self.input_combo.blockSignals(True); self.output_combo.blockSignals(True); self.input_combo.clear(); self.output_combo.clear(); self.input_combo.addItems(self.low_images.keys()); self.output_combo.addItems(self.low_images.keys()); self.input_combo.blockSignals(False); self.output_combo.blockSignals(False)
        if self.low_images:
            first_key = next(iter(self.low_images)); self.input_combo.setCurrentText(first_key); self.output_combo.setCurrentText(first_key); self._set_current_image(first_key); self.log(f"已载入 {len(self.low_images)} 张图像。"); self._set_summary(f"已导入 {len(self.low_images)} 张图像，当前为 {first_key}")
        else:
            self.current_key = None; self.original_view.clear_image(); self.hist_original.clear_image(); self._show_pending_enhanced_state(None); self._set_summary("当前没有可预览的图像")
    def load_model(self):
        default_path = self.config.checkpoints_dir / "latest.pth"; selected, _ = QFileDialog.getOpenFileName(self, "选择模型检查点", str(default_path.parent if default_path.parent.exists() else self.config.project_root), "PyTorch Checkpoints (*.pth)"); checkpoint_path = Path(selected) if selected else default_path
        if not checkpoint_path.exists(): QMessageBox.warning(self, "提示", f"未找到检查点: {checkpoint_path}"); return
        try: self.inferencer = EnhancementInferencer(checkpoint_path=checkpoint_path, config=self.config)
        except Exception as exc: QMessageBox.critical(self, "加载失败", str(exc)); return
        self.epoch_inferencers.clear(); self.refresh_history_view(initial=True); self.checkpoint_label.setText(checkpoint_path.name); self.device_label.setText(f"模型已加载 | {self.inferencer.device}"); self._set_summary(f"已加载模型: {checkpoint_path.name}"); self.log(f"已加载模型: {checkpoint_path}")
    def open_file(self):
        selected, _ = QFileDialog.getOpenFileName(self, "选择图像", str(self.config.project_root), supported_image_filters())
        if selected: self._load_images([Path(selected)])
    def open_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "选择图像文件夹", str(self.config.project_root))
        if not selected: return
        paths = list_supported_images(Path(selected))
        if not paths: QMessageBox.information(self, "提示", "所选文件夹中没有支持的图像。"); return
        self._load_images(paths)
    def add_normal_light_comparison(self):
        image_key = self.output_combo.currentText() or self.current_key
        if not image_key or image_key not in self.enhanced_results:
            QMessageBox.information(self, "提示", "请先生成或选择一张增强结果，再加入正常光图片对比。")
            return
        selected, _ = QFileDialog.getOpenFileName(self, "选择正常光参考图像", str(self.config.project_root), supported_image_filters())
        if not selected:
            return
        reference_path = Path(selected)
        reference_rgb = self._load_rgb_image(reference_path)
        if reference_rgb is None:
            QMessageBox.warning(self, "提示", f"无法读取正常光参考图像: {reference_path}")
            return
        result = self.enhanced_results[image_key]
        enhanced_rgb = result.get("enhanced_rgb")
        if enhanced_rgb is None:
            QMessageBox.information(self, "提示", "当前增强结果无效，请重新增强后再对比。")
            return
        original_shape = reference_rgb.shape[:2]
        reference_rgb = self._match_reference_size(reference_rgb, enhanced_rgb)
        if original_shape != reference_rgb.shape[:2]:
            self.log(f"正常光参考图尺寸已自动匹配到增强图尺寸: {reference_rgb.shape[1]}x{reference_rgb.shape[0]}")
        metrics = self._calculate_reference_metrics(reference_rgb, enhanced_rgb)
        result["normal_reference_path"] = reference_path
        result["normal_reference_rgb"] = reference_rgb
        result["normal_reference_metrics"] = metrics
        self._render_metric_views(reference_rgb, enhanced_rgb, metrics, reference_label="Normal")
        self.enhanced_hint_label.setText(f"已使用正常光参考图 {reference_path.name} 对比 {image_key} 的增强结果")
        self._set_summary(f"已完成正常光图片对比: {reference_path.name}")
        self.log(f"正常光对比完成: {image_key} vs {reference_path.name} | PSNR={metrics['psnr']:.3f}, SSIM={metrics['ssim']:.4f}, SMD2={metrics['smd2']:.4f}, LPIPS={metrics['lpips']:.4f}")
    def on_input_selected(self, key):
        if not key: return
        idx = self.output_combo.findText(key)
        if idx >= 0 and self.output_combo.currentIndex() != idx: self.output_combo.setCurrentIndex(idx)
        self._set_current_image(key)
    def on_output_selected(self, key):
        if not key or key not in self.low_images: return
        self.current_key = key
        self.active_preview_epoch = None
        idx = self.input_combo.findText(key)
        if idx >= 0 and self.input_combo.currentIndex() != idx: self.input_combo.setCurrentIndex(idx); return
        if not self._set_original_preview(key): return
        if key in self.enhanced_results: self._show_enhanced_result(key); self._set_summary(f"当前查看: {key}（原图与增强结果已同步）")
        else: self._show_pending_enhanced_state(key); self._set_summary(f"当前查看: {key}（该图片还未增强）")
    def enhance_current(self):
        if self.inferencer is None: QMessageBox.information(self, "提示", "请先加载模型。"); return
        if not self.current_key or self.current_key not in self.low_images: QMessageBox.information(self, "提示", "请先选择待增强图像。"); return
        if self.enhancement_worker and self.enhancement_worker.isRunning(): QMessageBox.information(self, "提示", "当前已有增强任务正在执行。"); return
        image_path = self.low_images[self.current_key]; self.device_label.setText("正在推理..."); self.enhanced_hint_label.setText(f"正在增强 {image_path.name} ..."); self._set_summary(f"正在增强: {image_path.name}"); self.log(f"开始增强: {image_path.name}")
        self.enhancement_worker = EnhancementWorker(self.inferencer, self.current_key, image_path); self.enhancement_worker.result_signal.connect(self.on_enhancement_result); self.enhancement_worker.error_signal.connect(self.on_enhancement_error); self.enhancement_worker.start()
    def on_enhancement_result(self, image_key, result):
        self.enhanced_results[image_key] = result; self.device_label.setText(f"增强完成 | {self.inferencer.device}"); self.log(f"增强完成: {image_key}"); self._set_summary(f"增强完成: {image_key}")
        if self.current_key == image_key and self.active_preview_epoch is None: self._show_enhanced_result(image_key)
    def on_enhancement_error(self, message):
        self.device_label.setText("推理失败"); self.log("增强失败: " + message); self.enhanced_hint_label.setText("增强失败，请查看控制台日志"); self._set_summary("增强失败，请查看控制台日志"); QMessageBox.critical(self, "增强失败", message)
    def export_current(self):
        export_key = self.output_combo.currentText() or self.current_key
        if export_key not in self.enhanced_results: QMessageBox.information(self, "提示", "当前选中的图片还没有增强结果。"); return
        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", str(self.config.outputs_dir))
        if not out_dir: return
        output_path = self.inferencer.export(self.low_images[export_key], self.enhanced_results[export_key]["enhanced_rgb"], Path(out_dir)); self.log(f"已导出: {output_path}"); self._set_summary(f"已导出当前增强结果: {output_path.name}")
    def export_all(self):
        ready_keys = [key for key in self.low_images if key in self.enhanced_results]
        if not ready_keys: QMessageBox.information(self, "提示", "导入图片中暂无已增强完成的结果可导出。"); return
        out_dir = QFileDialog.getExistingDirectory(self, "选择批量导出目录", str(self.config.outputs_dir))
        if not out_dir: return
        output_dir = ensure_dir(Path(out_dir)); count = 0
        for image_key in ready_keys: self.inferencer.export(self.low_images[image_key], self.enhanced_results[image_key]["enhanced_rgb"], output_dir); count += 1
        self.log(f"批量导出完成，共 {count} 张，目录: {output_dir}"); self._set_summary(f"已批量导出 {count} 张增强图像")
    def reset_views(self):
        self.original_view.fit_image(); self.enhanced_view.fit_image(); self.hist_original.fit_image(); self.hist_enhanced.fit_image(); self.chart_view.fit_image(); [view.fit_image() for view in self.metric_detail_views.values()]; self._set_summary("已重置当前视图缩放"); self.log("已重置所有预览视图。")
    def show_epoch_preview(self, epoch):
        if not self.current_key or self.current_key not in self.low_images: QMessageBox.information(self, "提示", "请先选择要查看中间结果的图片。"); return
        if epoch == 0:
            paths = self.preview_map.get(epoch, [])
            if not paths: QMessageBox.information(self, "提示", "暂无第 0 轮预览图。"); return
            image_bgr = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
            if image_bgr is None: QMessageBox.warning(self, "提示", f"无法读取预览图: {paths[0]}"); return
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            result = {"original_rgb": image_rgb, "enhanced_rgb": image_rgb}
            self.active_preview_epoch = epoch
            self.enhanced_view.set_image(numpy_rgb_to_qpixmap(image_rgb)); self._refresh_enhancement_analysis(result); self.enhanced_hint_label.setText(f"当前显示第 0 轮固定验证预览: {paths[0].name}"); self._set_summary("当前显示第 0 轮固定验证预览"); self.log(f"显示第 0 轮预览: {paths[0].name}"); return
        checkpoint_path = self._get_epoch_checkpoint_path(epoch)
        if checkpoint_path is None: QMessageBox.information(self, "提示", f"未找到第 {epoch} 轮模型检查点。"); return
        try: inferencer = self._get_epoch_inferencer(epoch)
        except Exception as exc: QMessageBox.critical(self, "加载失败", str(exc)); return
        image_path = self.low_images[self.current_key]
        try: result = inferencer.enhance(image_path)
        except Exception as exc: QMessageBox.critical(self, "预览失败", str(exc)); return
        self.active_preview_epoch = epoch
        enhanced_rgb = result["enhanced_rgb"]
        self.enhanced_view.set_image(numpy_rgb_to_qpixmap(enhanced_rgb)); self._refresh_enhancement_analysis(result); self.enhanced_hint_label.setText(f"当前显示 {self.current_key} 在第 {epoch} 轮的增强结果"); self._set_summary(f"当前显示 {self.current_key} 在第 {epoch} 轮的增强结果"); self.log(f"显示 {self.current_key} 的第 {epoch} 轮预览: {checkpoint_path.name}")

def main():
    app = QApplication(sys.argv); window = NightEnhancementWindow(); window.show(); sys.exit(app.exec_())
if __name__ == "__main__": main()
