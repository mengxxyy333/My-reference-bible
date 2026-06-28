from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from night_enhancement.config import TrainConfig
from night_enhancement.infer import EnhancementInferencer
from night_enhancement.utils import ensure_dir


def export_preview_grid() -> None:
    config = TrainConfig()
    preview_dir = config.progress_dir
    ensure_dir(preview_dir)
    records = []
    for image_path in sorted(preview_dir.glob("epoch_*.png")):
        parts = image_path.stem.split("_")
        if len(parts) >= 3:
            epoch = parts[1]
            name = "_".join(parts[2:])
        else:
            epoch = "unknown"
            name = image_path.stem
        records.append({"epoch": epoch, "name": name, "path": str(image_path)})

    csv_path = config.logs_dir / "preview_index.csv"
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "name", "path"])
        writer.writeheader()
        writer.writerows(records)


def run_single(image_path: str, checkpoint_path: Optional[str] = None, output_dir: Optional[str] = None) -> Path:
    config = TrainConfig()
    inferencer = EnhancementInferencer(
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        config=config,
    )
    result = inferencer.enhance(Path(image_path))
    final_output_dir = Path(output_dir) if output_dir else config.outputs_dir / "inference"
    return inferencer.export(Path(image_path), result["enhanced_rgb"], final_output_dir)


if __name__ == "__main__":
    export_preview_grid()
