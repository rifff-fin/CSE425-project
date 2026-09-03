from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from evaluate import evaluate_tagging
class FMAMelDataset(Dataset):
    def __init__(self, manifest_path: Path, label_names: List[str], sample_rate: int = 22050, duration: float = 5.0) -> None:
        self.records = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.labels = {label.lower(): i for i, label in enumerate(label_names)}
        self.sample_rate = sample_rate
        self.samples = int(sample_rate * duration)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        signal, _ = librosa.load(record["audio_path"], sr=self.sample_rate, mono=True, duration=self.samples / self.sample_rate)
        signal = np.asarray(signal, dtype=np.float32)
        if signal.size < self.samples:
            signal = np.pad(signal, (0, self.samples - signal.size))
        signal = signal[: self.samples]
        mel = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_fft=2048, hop_length=512, n_mels=128)
        mel = librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        target = torch.zeros(len(self.labels), dtype=torch.float32)
        for tag in record.get("tags", []):
            if str(tag).lower() in self.labels:
                target[self.labels[str(tag).lower()]] = 1.0
        return torch.from_numpy(mel).unsqueeze(0), target
class MelCNN(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN mel-spectrogram FMA baseline.")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("results/fma_cnn_baseline_metrics.json"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = args.manifest_root if args.manifest_root.is_absolute() else project_root / args.manifest_root
    output = args.output if args.output.is_absolute() else project_root / args.output
    labels = sorted({str(tag).lower() for split in ("train.json", "val.json", "test.json") for row in json.loads((root / split).read_text(encoding="utf-8")) for tag in row.get("tags", [])})
    train_ds = FMAMelDataset(root / "train.json", labels)
    val_ds = FMAMelDataset(root / "val.json", labels)
    test_ds = FMAMelDataset(root / "test.json", labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    model = MelCNN(len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for features, targets in train_loader:
            loss = criterion(model(features), targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); running += loss.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                val_loss += criterion(model(features), targets).item()
        print(f"[CNN] Epoch {epoch}/{args.epochs} | train_loss={running / max(1, len(train_loader)):.4f} | val_loss={val_loss / max(1, len(val_loader)):.4f}")
    logits, targets = [], []
    with torch.no_grad():
        for features, batch_targets in test_loader:
            logits.append(model(features)); targets.append(batch_targets)
    scores = evaluate_tagging(torch.cat(logits), torch.cat(targets))
    result: Dict[str, Any] = {"task": "task2_baseline", "model": "cnn_mel_spectrogram", "split": "test", "train_samples": len(train_ds), "val_samples": len(val_ds), "test_samples": len(test_ds), "labels": labels, **scores}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
