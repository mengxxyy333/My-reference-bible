from __future__ import annotations

import argparse
from pathlib import Path

from night_enhancement.config import TrainConfig
from night_enhancement.infer import EnhancementInferencer
from night_enhancement.ui.helpers import list_supported_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch enhance low-light images.")
    parser.add_argument("input", help="Input image or folder path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--output", default=None, help="Output folder")
    args = parser.parse_args()

    config = TrainConfig()
    inferencer = EnhancementInferencer(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        config=config,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else (config.outputs_dir / "batch")
    if input_path.is_file():
        result = inferencer.enhance(input_path)
        saved = inferencer.export(input_path, result["enhanced_rgb"], output_dir)
        print(f"Saved: {saved}")
        return

    for image_path in list_supported_images(input_path):
        result = inferencer.enhance(image_path)
        saved = inferencer.export(image_path, result["enhanced_rgb"], output_dir)
        print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
