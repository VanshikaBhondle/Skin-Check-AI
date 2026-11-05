"""Train a skin-condition image classifier and export it for the app.

Run on Google Colab (free GPU) or any PC with PyTorch:
    python train.py --data /path/to/dataset --epochs 15

Dataset layout (any of these work):
    dataset/train_set/<class>/*.jpg + dataset/test_set/<class>/*.jpg
    dataset/train/<class>/... + dataset/val/<class>/... + dataset/test/<class>/...
    dataset/<class>/*.jpg                      (split automatically 70/15/15)

Outputs (in --out, default ./export):
    skin_model.onnx   the model, for the app (runs with onnxruntime, no PyTorch needed)
    labels.json       class names + preprocessing settings the app needs
    metrics.json      accuracy, per-class precision/recall/F1, confusion matrix
    confusion_matrix.png, training_curve.png
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def pretty(name: str) -> str:
    """'BA- cellulitis' -> 'Cellulitis', 'FU-athlete-foot' -> 'Athlete Foot', 'Seborrh_Keratoses' -> 'Seborrh Keratoses'."""
    name = re.sub(r"^[A-Z]{2}-\s*", "", name.strip())
    return re.sub(r"[-_]+", " ", name).strip().title()


def find_splits(root: Path) -> tuple[Path, Path | None, Path | None]:
    """Locate train / val / test folders, descending one level if the zip has a wrapper folder."""
    for base in [root, *[p for p in root.iterdir() if p.is_dir()]]:
        subs = {p.name.lower(): p for p in base.iterdir() if p.is_dir()}
        train = next((subs[k] for k in subs if k.startswith("train")), None)
        if train:
            val = next((subs[k] for k in subs if k.startswith(("val", "valid"))), None)
            test = next((subs[k] for k in subs if k.startswith("test")), None)
            return train, val, test
    return root, None, None  # flat: <class>/images


def stratified_split(targets: list[int], fracs: tuple[float, ...], seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, t in enumerate(targets):
        by_class.setdefault(t, []).append(i)
    parts = [[] for _ in fracs]
    for idx in by_class.values():
        rng.shuffle(idx)
        start = 0
        for k, f in enumerate(fracs):
            n = len(idx) - start if k == len(fracs) - 1 else max(1, round(len(idx) * f))
            parts[k] += idx[start:start + n]
            start += n
    return parts


def per_class_metrics(cm: np.ndarray, labels: list[str]) -> list[dict]:
    out = []
    for i, name in enumerate(labels):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"class": name, "precision": round(float(prec), 3), "recall": round(float(rec), 3),
                    "f1": round(float(f1), 3), "support": int(cm[i, :].sum())})
    return out


@torch.no_grad()
def evaluate(model, loader, device, n_classes) -> tuple[float, float, np.ndarray]:
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    loss_sum, n = 0.0, 0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        n += len(y)
        for t, p in zip(y.cpu().numpy(), logits.argmax(1).cpu().numpy()):
            cm[t, p] += 1
    return loss_sum / max(n, 1), float(np.trace(cm) / max(cm.sum(), 1)), cm


def save_plots(out: Path, history: list[dict], cm: np.ndarray, labels: list[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, [h["train_acc"] for h in history], lw=2, color="#2a78d6", label="train accuracy")
    ax.plot(ep, [h["val_acc"] for h in history], lw=2, color="#eb6834", label="validation accuracy")
    ax.set_xlabel("epoch"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(); ax.set_title("Training curve")
    fig.tight_layout(); fig.savefig(out / "training_curve.png", dpi=120); plt.close(fig)

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1 + .6 * len(labels), 1 + .55 * len(labels)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > .5 else "#0b0b0b")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="export")
    ap.add_argument("--model", default="efficientnet_b0", help="any timm model, e.g. efficientnet_b0, convnext_tiny")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5, help="stop after this many epochs without improvement")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="only for quick tests")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}" + ("" if device == "cuda" else "  (no GPU - training will be slow)"))

    train_dir, val_dir, test_dir = find_splits(Path(args.data))
    base = datasets.ImageFolder(train_dir)
    raw_classes = base.classes
    labels = [pretty(c) for c in raw_classes]
    n_classes = len(labels)
    print(f"Classes ({n_classes}): {labels}")

    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=n_classes)
    cfg = resolve_data_config({}, model=model)
    size = cfg["input_size"][-1]
    mean, std, crop_pct = list(cfg["mean"]), list(cfg["std"]), cfg.get("crop_pct", 0.875)
    resize = int(math.floor(size / crop_pct))
    norm = transforms.Normalize(mean, std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.15, 0.02), transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size), transforms.ToTensor(), norm])

    # Build train / val / test sets (same class order everywhere)
    def folder(path, tf):
        ds = datasets.ImageFolder(path, transform=tf)
        assert ds.classes == raw_classes, f"Class folders in {path} don't match the training folders"
        return ds

    if val_dir and test_dir:
        train_ds, val_ds, test_ds = folder(train_dir, train_tf), folder(val_dir, eval_tf), folder(test_dir, eval_tf)
    elif test_dir:  # carve validation out of train
        tr_idx, va_idx = stratified_split(base.targets, (0.85, 0.15), args.seed)
        train_ds = Subset(folder(train_dir, train_tf), tr_idx)
        val_ds = Subset(folder(train_dir, eval_tf), va_idx)
        test_ds = folder(test_dir, eval_tf)
    else:
        tr_idx, va_idx, te_idx = stratified_split(base.targets, (0.7, 0.15, 0.15), args.seed)
        train_ds = Subset(folder(train_dir, train_tf), tr_idx)
        val_ds, test_ds = Subset(folder(train_dir, eval_tf), va_idx), Subset(folder(train_dir, eval_tf), te_idx)
    print(f"Images - train: {len(train_ds)}, validation: {len(val_ds)}, test: {len(test_ds)}")

    train_targets = [base.targets[i] for i in train_ds.indices] if isinstance(train_ds, Subset) else train_ds.targets
    counts = np.bincount(train_targets, minlength=n_classes)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    print("Images per class (train):", dict(zip(labels, counts.tolist())))

    kw = dict(batch_size=args.bs, num_workers=args.workers, pin_memory=device == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.bs, **kw)
    val_dl, test_dl = DataLoader(val_ds, **kw), DataLoader(test_ds, **kw)

    model.to(device)
    crit = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)  # weights fix class imbalance
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * max(len(train_dl), 1)
    warm = max(len(train_dl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_acc, best_state, history, stale = -1.0, None, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, correct, seen, loss_sum = time.time(), 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.item() * len(y); seen += len(y); correct += (logits.argmax(1) == y).sum().item()
        val_loss, val_acc, _ = evaluate(model, val_dl, device, n_classes)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "train_acc": correct / max(seen, 1),
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:2d}/{args.epochs}  train acc {correct / max(seen, 1):.3f}  "
              f"val acc {val_acc:.3f}  val loss {val_loss:.3f}  ({time.time() - t0:.0f}s)")
        if val_acc > best_acc:
            best_acc, stale = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"No improvement for {args.patience} epochs - stopping early.")
                break

    model.load_state_dict(best_state)
    _, test_acc, cm = evaluate(model, test_dl, device, n_classes)
    per_class = per_class_metrics(cm, labels)
    macro_f1 = float(np.mean([c["f1"] for c in per_class]))
    print(f"\nTEST accuracy: {test_acc:.3f}   macro F1: {macro_f1:.3f}")
    for c in per_class:
        print(f"  {c['class']:<28} precision {c['precision']:.2f}  recall {c['recall']:.2f}  f1 {c['f1']:.2f}  (n={c['support']})")

    # Export to ONNX so the app needs only onnxruntime (small) instead of PyTorch (huge)
    model.eval().cpu()
    dummy = torch.randn(1, 3, size, size)
    onnx_path = out / "skin_model.onnx"
    torch.onnx.export(model, (dummy,), str(onnx_path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
    try:  # fold any external weight file back into a single .onnx file
        import onnx
        m = onnx.load(str(onnx_path))
        onnx.save_model(m, str(onnx_path), save_as_external_data=False)
        for extra in out.glob("skin_model.onnx.data"):
            extra.unlink()
    except ImportError:
        pass

    try:  # check the exported model gives the same answers as PyTorch
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = torch.randn(2, 3, size, size)
        diff = np.abs(sess.run(None, {"image": x.numpy()})[0] - model(x).detach().numpy()).max()
        print(f"ONNX check: max difference vs PyTorch = {diff:.2e}")
    except ImportError:
        print("onnxruntime not installed - skipped ONNX check")

    meta = {"labels": labels, "raw_classes": raw_classes, "model": args.model, "img_size": size,
            "resize": resize, "mean": mean, "std": std, "test_accuracy": round(test_acc, 4),
            "macro_f1": round(macro_f1, 4), "trained_at": time.strftime("%Y-%m-%d")}
    (out / "labels.json").write_text(json.dumps(meta, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"test_accuracy": test_acc, "macro_f1": macro_f1, "best_val_accuracy": best_acc, "per_class": per_class,
         "confusion_matrix": cm.tolist(), "labels": labels, "history": history,
         "images": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}}, indent=2))
    save_plots(out, history, cm, labels)
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"\nSaved to {out.resolve()}: skin_model.onnx ({size_mb:.1f} MB), labels.json, metrics.json, plots")


if __name__ == "__main__":
    main()
