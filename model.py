"""
model.py - Định nghĩa model EfficientNet-B0 (phù hợp máy cấu hình vừa)
"""
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# ChickenDiseaseClassifier
# ─────────────────────────────────────────────────────────────────────────────
class ChickenDiseaseClassifier(nn.Module):
    """
    EfficientNet-B0 với custom classification head.

    Lý do chọn EfficientNet-B0:
    - ~5.3M params — nhỏ gọn, chạy tốt trên CPU/GPU vừa
    - ImageNet accuracy cao hơn ResNet-18 cùng mức tham số
    - Pretrained weights giúp converge nhanh với dataset nhỏ (~8K ảnh)
    """

    SUPPORTED_MODELS: Dict[str, callable] = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "mobilenet_v3_small": models.mobilenet_v3_small,
        "resnet18": models.resnet18,
    }

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone, in_features = self._build_backbone()
        self.classifier = self._build_head(in_features)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    # ── Utility ───────────────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Đóng băng backbone, chỉ train head (warm-up phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone đã đóng băng. Chỉ train classification head.")

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 3) -> None:
        """
        Mở N block cuối của backbone để fine-tune.
        Gọi sau khi warm-up xong.
        """
        # Với EfficientNet, features là Sequential — lấy N block cuối
        all_layers = list(self.backbone.children())
        for layer in all_layers[-unfreeze_last_n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Mở {unfreeze_last_n_blocks} block cuối. Trainable params: {trainable:,}")

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_backbone(self):
        """Load pretrained backbone, bỏ classification head gốc."""
        model_fn = self.SUPPORTED_MODELS.get(self.cfg.model_name)
        if model_fn is None:
            raise ValueError(
                f"Model '{self.cfg.model_name}' không hỗ trợ. "
                f"Chọn một trong: {list(self.SUPPORTED_MODELS.keys())}"
            )

        if self.cfg.model_name.startswith("efficientnet"):
            weights = EfficientNet_B0_Weights.DEFAULT if self.cfg.pretrained else None
            base = model_fn(weights=weights)
            in_features = base.classifier[1].in_features
            # Thay classifier bằng Identity để lấy features
            base.classifier = nn.Identity()

        elif self.cfg.model_name == "mobilenet_v3_small":
            base = model_fn(pretrained=self.cfg.pretrained)
            in_features = base.classifier[3].in_features
            base.classifier = nn.Identity()

        elif self.cfg.model_name == "resnet18":
            base = model_fn(pretrained=self.cfg.pretrained)
            in_features = base.fc.in_features
            base.fc = nn.Identity()

        else:
            raise ValueError(f"Chưa xử lý model: {self.cfg.model_name}")

        return base, in_features

    def _build_head(self, in_features: int) -> nn.Sequential:
        """Custom classification head với BatchNorm + Dropout."""
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=self.cfg.dropout_rate),
            nn.Linear(in_features, 256),
            nn.SiLU(),                           # Smooth activation (dùng trong EfficientNet)
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.cfg.dropout_rate / 2),
            nn.Linear(256, self.cfg.num_classes),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ModelFactory
# ─────────────────────────────────────────────────────────────────────────────
class ModelFactory:
    """Factory để tạo model từ config hoặc load từ checkpoint."""

    @staticmethod
    def create(cfg: Config) -> ChickenDiseaseClassifier:
        model = ChickenDiseaseClassifier(cfg)
        stats = model.count_parameters()
        print(f"[Model] {cfg.model_name} — "
              f"Total: {stats['total']:,} | Trainable: {stats['trainable']:,}")
        return model

    @staticmethod
    def load_checkpoint(path: str, cfg: Config, device: torch.device) -> ChickenDiseaseClassifier:
        model = ChickenDiseaseClassifier(cfg)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Model] Loaded checkpoint từ: {path} (epoch {checkpoint.get('epoch', '?')})")
        return model
