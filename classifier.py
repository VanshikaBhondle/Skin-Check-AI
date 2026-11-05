"""Runs YOUR trained model (models/skin_model.onnx) - no PyTorch needed, only onnxruntime.

Preprocessing here must match the evaluation transform in training/train.py:
resize shorter side -> center crop -> scale to 0..1 -> normalize with the model's mean/std.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "skin_model.onnx"
LABELS_PATH = MODEL_DIR / "labels.json"


class SkinClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
        import onnxruntime as ort
        self.meta = json.loads(Path(labels_path).read_text())
        self.labels: list[str] = self.meta["labels"]
        self.size, self.resize = int(self.meta["img_size"]), int(self.meta["resize"])
        self.mean = np.array(self.meta["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(self.meta["std"], dtype=np.float32).reshape(3, 1, 1)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        w, h = img.size  # same rounding as torchvision.transforms.Resize(int) + CenterCrop
        short, long = min(w, h), max(w, h)
        new_short, new_long = self.resize, int(self.resize * long / short)
        img = img.resize((new_short, new_long) if w <= h else (new_long, new_short), Image.BICUBIC)
        w, h = img.size
        left, top = int(round((w - self.size) / 2.0)), int(round((h - self.size) / 2.0))
        img = img.crop((left, top, left + self.size, top + self.size))
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None].astype(np.float32)

    def predict(self, image_bytes: bytes, top: int = 3) -> list[dict]:
        x = self.preprocess(Image.open(io.BytesIO(image_bytes)))
        logits = self.session.run(None, {self.input_name: x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)[:top]
        return [{"name": self.labels[i], "confidence": round(float(p[i]) * 100, 1)} for i in order]

    @property
    def short(self) -> str:
        acc = self.meta.get("test_accuracy")
        return f"{len(self.labels)} conditions" + (f" · {acc * 100:.0f}% test accuracy" if acc is not None else "")

    @property
    def info(self) -> str:
        acc = self.meta.get("test_accuracy")
        return (f"{self.meta.get('model', 'model')} trained on {len(self.labels)} conditions"
                + (f", test accuracy {acc * 100:.0f}%" if acc is not None else ""))


def load() -> SkinClassifier | None:
    """The classifier, or None if no trained model has been added to models/ yet."""
    if not (MODEL_PATH.exists() and LABELS_PATH.exists()):
        return None
    return SkinClassifier()
