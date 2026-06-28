from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from night_enhancement.config import TrainConfig
from night_enhancement.dataset import LowLightDataset
from night_enhancement.losses import CombinedLoss
from night_enhancement.models import RetinexEnhancementNet
from night_enhancement.utils import ensure_dir, save_tensor_image, set_seed


ProgressCallback = Callable[[dict[str, Any]], None]


class Trainer:
    def __init__(self, config: TrainConfig, progress_callback: Optional[ProgressCallback] = None, resume: bool = True) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self.resume = resume
        self.start_epoch = 1
        self.best_val_l1 = float("inf")
        self._resume_optimizer_state: Optional[dict[str, Any]] = None
        self._resume_scheduler_state: Optional[dict[str, Any]] = None
        self._optimizer_state_loaded = False
        set_seed(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")

        ensure_dir(config.checkpoints_dir)
        ensure_dir(config.sample_dir)
        ensure_dir(config.progress_dir)
        ensure_dir(config.logs_dir)

        self.train_dataset = LowLightDataset(config.train_low_dir, config.train_high_dir, config.image_size, augment=True)
        self.val_dataset = LowLightDataset(config.val_low_dir, config.val_high_dir, config.image_size, augment=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=self.device.type == "cuda")
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)

        self.model = RetinexEnhancementNet(freeze_encoder=True).to(self.device)
        self.criterion = CombinedLoss().to(self.device)
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
        self.history: list[dict[str, Any]] = []

        if self.resume:
            self.load_checkpoint_if_available()

    def emit(self, stage: str, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"stage": stage, **payload})

    def load_checkpoint_if_available(self) -> None:
        latest_path = self.config.checkpoints_dir / "latest.pth"
        if not latest_path.exists():
            return
        checkpoint = torch.load(latest_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", [])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_val_l1 = float(checkpoint.get("best_val_l1", float("inf")))
        self._resume_optimizer_state = checkpoint.get("optimizer_state_dict")
        self._resume_scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.start_epoch <= self.config.freeze_encoder_until:
            self.model.freeze_shallow_encoder()
        else:
            self.model.unfreeze_all()
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            self.scheduler = StepLR(self.optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma)
        self._try_restore_optimizer_state()
        self.emit("log", message=f"检测到断点，已从第 {self.start_epoch} 轮继续训练。")

    def _try_restore_optimizer_state(self) -> None:
        if self._optimizer_state_loaded or self._resume_optimizer_state is None:
            return
        try:
            self.optimizer.load_state_dict(self._resume_optimizer_state)
            if self._resume_scheduler_state is not None:
                self.scheduler.load_state_dict(self._resume_scheduler_state)
            self._optimizer_state_loaded = True
        except ValueError:
            self.emit("log", message="检测到旧版或阶段不一致的优化器状态，已跳过优化器恢复，仅恢复模型权重继续训练。")

    def _refresh_optimizer_if_needed(self, epoch: int) -> None:
        if epoch == self.config.freeze_encoder_until:
            self.model.unfreeze_all()
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            self.scheduler = StepLR(self.optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma)
            self._try_restore_optimizer_state()
            self.emit("log", message=f"第 {epoch} 轮开始解冻全部编码器参数。")

    def train(self) -> None:
        self.emit("status", message=f"训练开始，设备: {self.device}")
        if not self.history:
            self.save_preview(epoch=0)
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self._refresh_optimizer_if_needed(epoch)
            self.model.train()
            epoch_loss = 0.0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)

            for batch_idx, batch in enumerate(progress, start=1):
                low = batch["low"].to(self.device)
                high = batch["high"].to(self.device)
                outputs = self.model(low)
                loss, loss_items = self.criterion(outputs["enhanced"], high)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress.set_postfix(loss=f"{loss_items['total']:.4f}")
                self.emit("batch", epoch=epoch, total_epochs=self.config.epochs, batch=batch_idx, total_batches=len(self.train_loader), loss=loss_items["total"])

            self.scheduler.step()
            val_metrics = self.validate()
            avg_loss = epoch_loss / max(1, len(self.train_loader))
            history_item = {"epoch": epoch, "train_loss": avg_loss, **val_metrics, "lr": self.optimizer.param_groups[0]["lr"]}
            self.history = [item for item in self.history if int(item.get("epoch", -1)) != epoch]
            self.history.append(history_item)
            self.history.sort(key=lambda item: int(item.get("epoch", 0)))
            self._save_history()

            preview_path = None
            if epoch in self.config.preview_epochs:
                preview_path = self.save_preview(epoch)
            checkpoint_path = self.save_checkpoint(epoch)
            if val_metrics["val_l1"] < self.best_val_l1:
                self.best_val_l1 = val_metrics["val_l1"]
                self.save_best_checkpoint(epoch)
                self.emit("log", message=f"第 {epoch} 轮刷新最佳模型，val_l1={self.best_val_l1:.4f}")
            if epoch % self.config.save_every == 0 or epoch == self.config.epochs:
                self.save_epoch_snapshot(checkpoint_path)

            message = f"Epoch {epoch:03d} | train_loss={avg_loss:.4f} | val_l1={val_metrics['val_l1']:.4f} | val_ssim_loss={val_metrics['val_ssim_loss']:.4f}"
            print(message)
            self.emit("epoch", metrics=history_item, preview_path=str(preview_path) if preview_path else None, message=message)

        self.emit("finished", history=self.history, message="训练完成。")

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        l1_total = 0.0
        ssim_total = 0.0
        count = 0
        for batch in self.val_loader:
            low = batch["low"].to(self.device)
            high = batch["high"].to(self.device)
            outputs = self.model(low)
            _, loss_items = self.criterion(outputs["enhanced"], high)
            l1_total += loss_items["color"]
            ssim_total += loss_items["ssim"]
            count += 1
        return {"val_l1": l1_total / max(1, count), "val_ssim_loss": ssim_total / max(1, count)}

    @torch.no_grad()
    def save_preview(self, epoch: int) -> Path:
        self.model.eval()
        batch = next(iter(self.val_loader))
        low = batch["low"].to(self.device)
        outputs = self.model(low)
        image = outputs["enhanced"][0]
        name = Path(batch["name"][0]).stem
        path = self.config.progress_dir / f"epoch_{epoch:03d}_{name}.png"
        save_tensor_image(image, path)
        return path

    def save_checkpoint(self, epoch: int) -> Path:
        checkpoint_path = self.config.checkpoints_dir / "latest.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_l1": self.best_val_l1,
            "config": self.config.__dict__,
        }, checkpoint_path)
        return checkpoint_path

    def save_best_checkpoint(self, epoch: int) -> Path:
        best_path = self.config.checkpoints_dir / "best.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_l1": self.best_val_l1,
            "config": self.config.__dict__,
        }, best_path)
        return best_path

    def save_epoch_snapshot(self, latest_path: Path) -> Path:
        epoch = int(self.history[-1]["epoch"])
        snapshot_path = self.config.checkpoints_dir / f"retinex_epoch_{epoch:03d}.pth"
        torch.save(torch.load(latest_path, map_location="cpu", weights_only=False), snapshot_path)
        return snapshot_path

    def _save_history(self) -> None:
        history_path = self.config.logs_dir / "train_history.json"
        serializable = [{key: float(value) if isinstance(value, (int, float)) else value for key, value in item.items()} for item in self.history]
        history_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    Trainer(TrainConfig()).train()


if __name__ == "__main__":
    main()
