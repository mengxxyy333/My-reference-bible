from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

@dataclass
class TrainConfig:
    # 项目根目录自动定位
    project_root: Path = Path(__file__).resolve().parents[1]
    data_root: Path = project_root / "data"

    # 训练集与验证集路径（遵循LOL数据集结构）
    train_low_dir: Path = data_root / "train" / "low"
    train_high_dir: Path = data_root / "train" / "high"
    val_low_dir: Path = data_root / "val" / "low"
    val_high_dir: Path = data_root / "val" / "high"

    # 输出目录：模型权重、日志、样本图
    checkpoints_dir: Path = project_root / "checkpoints"
    outputs_dir: Path = project_root / "outputs"
    sample_dir: Path = outputs_dir / "samples"
    progress_dir: Path = outputs_dir / "progress"
    logs_dir: Path = outputs_dir / "logs"

    # 图像预处理尺寸 (高, 宽)
    image_size: Tuple[int, int] = (400, 600)

    # 训练参数
    batch_size: int = 2
    num_workers: int = 2
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    device: str = "cuda"
    seed: int = 42
    save_every: int = 10

    # 在这些特定轮次生成预览图（用于GUI查看中间效果）
    preview_epochs: List[int] = field(default_factory=lambda: [0, 20, 50, 80, 100])

    # 前10轮冻结VGG编码器，之后解冻微调
    freeze_encoder_until: int = 10

# 支持的图片格式
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
