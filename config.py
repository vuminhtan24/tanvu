"""
config.py - Cấu hình toàn bộ pipeline
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ── Đường dẫn ──────────────────────────────────────────────────────────────
    csv_path: str = "train_data.csv"           # File CSV chứa tên ảnh + nhãn
    image_dir: str = "Train"          # Thư mục chứa ảnh
    output_dir: str = "outputs"                # Lưu model, logs, kết quả

    # ── Dữ liệu ────────────────────────────────────────────────────────────────
    image_col: str = "images"
    label_col: str = "label"
    img_size: Tuple[int, int] = (224, 224)
    val_split: float = 0.15                    # 15% validation
    test_split: float = 0.10                   # 10% test
    random_seed: int = 42

    # ── Model ───────────────────────────────────────────────────────────────────
    model_name: str = "efficientnet_b0"        # Nhẹ, phù hợp máy cấu hình vừa
    num_classes: int = 4
    pretrained: bool = True
    dropout_rate: float = 0.3

    # ── Training ────────────────────────────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 32                       # Giảm xuống 16 nếu RAM thấp
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7                          # Early stopping
    num_workers: int = 2                       # Giảm xuống 0 nếu lỗi multiprocessing

    # ── Augmentation ────────────────────────────────────────────────────────────
    use_augmentation: bool = True
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.3
    rotation_degrees: int = 15
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2

    # ── Scheduler ───────────────────────────────────────────────────────────────
    scheduler: str = "cosine"                  # "cosine" | "step" | "none"
    step_size: int = 10
    gamma: float = 0.1

    # ── Class names (tự động fill từ CSV nếu để rỗng) ──────────────────────────
    class_names: List[str] = field(default_factory=lambda: [
        "Coccidiosis",
        "Healthy",
        "New Castle Disease",
        "Salmonella",
    ])

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
