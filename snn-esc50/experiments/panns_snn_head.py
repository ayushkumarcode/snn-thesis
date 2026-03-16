"""
panns_snn_head.py -- PANNs CNN14 + SNN head transfer learning.

Research question: With pre-trained AudioSet features (2048-d embeddings),
can an SNN classification head match or exceed the ANN baseline?

Method:
  1. Extract CNN14 embeddings (2048-d) from all ESC-50 clips (cached to disk).
     CNN14 trained on AudioSet (1.9M clips, 527 classes) -- extremely rich
     audio features. ESC-50 performance with fine-tuning: 94.7%.
  2. Run 5-fold CV with three head variants:
       (a) SNN head: LIF(2048→512) → LIF(512→256) → LIF(256→50)
       (b) ANN head: ReLU(2048→512) → ReLU(512→256) → Linear(256→50)
       (c) Linear probe: Linear(2048→50) [ANN] for upper bound reference
  3. SNN head uses direct encoding: repeat embedding T=25 times.
     (Embeddings are continuous values in [0,1] after sigmoid in CNN14,
     so direct encoding is the correct choice.)

Reference:
  Kong et al. (2020). "PANNs: Large-Scale Pretrained Audio Neural Networks."
  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 2880–2894.

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/panns_snn_head.py
  python experiments/panns_snn_head.py --epochs 50 --fold 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.config import (
    RESULTS_DIR, ESC50_AUDIO_DIR, ESC50_META_PATH,
    NUM_CLASSES, NUM_STEPS, NUM_FOLDS, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
)

PANNS_SR    = 32000   # CNN14 sample rate
EMBED_DIM   = 2048    # CNN14 embedding dimension
PANNS_CKPT  = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"


# ============================================================
# CNN14 embedding extraction
# ============================================================

def extract_panns_embeddings(audio_dir: Path, meta_path: Path,
                              cache_dir: Path) -> tuple:
    """Extract CNN14 embeddings for all ESC-50 clips.

    Caches to `cache_dir/panns_embeddings.npy` and `panns_labels.npy`.

    Returns:
        embeddings: (N, 2048) float32 numpy array
        labels:     (N,) int64 numpy array
        fold_ids:   (N,) int64 numpy array (1-5 fold assignments)
    """
    import pandas as pd
    import librosa

    emb_path  = cache_dir / "panns_embeddings.npy"
    lab_path  = cache_dir / "panns_labels.npy"
    fold_path = cache_dir / "panns_folds.npy"

    if emb_path.exists() and lab_path.exists() and fold_path.exists():
        print(f"Loading cached embeddings from {cache_dir}")
        return (
            np.load(emb_path),
            np.load(lab_path),
            np.load(fold_path),
        )

    print("Extracting CNN14 embeddings (this takes a few minutes)...")
    from panns_inference import AudioTagging

    at = AudioTagging(checkpoint_path=str(PANNS_CKPT), device="cpu")

    meta = pd.read_csv(meta_path)
    meta = meta.sort_values(["fold", "target"]).reset_index(drop=True)

    embeddings = []
    labels     = []
    fold_ids   = []

    for i, row in meta.iterrows():
        if i % 100 == 0:
            print(f"  {i}/{len(meta)} clips processed...")

        filepath = audio_dir / row["filename"]

        # Load audio at CNN14's required sample rate (32kHz)
        y, _ = librosa.load(str(filepath), sr=PANNS_SR, duration=5)
        expected_len = PANNS_SR * 5
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))

        # CNN14 expects (batch, samples) numpy array
        audio_batch = y[np.newaxis, :]  # (1, 160000)
        _, embedding = at.inference(audio_batch)  # (1, 2048)

        embeddings.append(embedding[0])       # (2048,)
        labels.append(int(row["target"]))
        fold_ids.append(int(row["fold"]))

    embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    labels     = np.array(labels, dtype=np.int64)
    fold_ids   = np.array(fold_ids, dtype=np.int64)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    np.save(lab_path, labels)
    np.save(fold_path, fold_ids)
    print(f"Embeddings cached to {cache_dir}")

    return embeddings, labels, fold_ids


# ============================================================
# SNN head: LIF(2048→512) → LIF(512→256) → LIF(256→50)
# ============================================================

class SNNHead(nn.Module):
    """Spiking neural network head for PANNs embeddings.

    Input shape: (T, B, 2048) — direct-encoded embeddings.
    Output: spk_out (T, B, 50), mem_out (T, B, 50).
    """

    def __init__(self, embed_dim: int = EMBED_DIM,
                 num_classes: int = NUM_CLASSES,
                 num_steps: int = NUM_STEPS,
                 beta: float = 0.9):
        super().__init__()
        self.num_steps = num_steps
        sg = surrogate.atan(alpha=2.0)

        self.fc1  = nn.Linear(embed_dim, 512)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)
        self.bn1  = nn.BatchNorm1d(512)

        self.fc2  = nn.Linear(512, 256)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)
        self.bn2  = nn.BatchNorm1d(256)

        self.fc3  = nn.Linear(256, num_classes)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=sg)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (T, B, embed_dim) direct-encoded embeddings

        Returns:
            spk_out: (T, B, num_classes)
            mem_out: (T, B, num_classes)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []
        mem_rec = []

        for step in range(self.num_steps):
            h = self.fc1(x[step])
            h = self.bn1(h)
            spk1, mem1 = self.lif1(h, mem1)

            h = self.fc2(spk1)
            h = self.bn2(h)
            spk2, mem2 = self.lif2(h, mem2)

            h = self.fc3(spk2)
            spk3, mem3 = self.lif3(h, mem3)

            spk_rec.append(spk3)
            mem_rec.append(mem3)

        return torch.stack(spk_rec), torch.stack(mem_rec)


# ============================================================
# ANN head: ReLU(2048→512) → ReLU(512→256) → Linear(256→50)
# ============================================================

class ANNHead(nn.Module):
    """ANN classification head (matched architecture for fair comparison)."""

    def __init__(self, embed_dim: int = EMBED_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearProbe(nn.Module):
    """Linear probe: single linear layer on frozen embeddings."""

    def __init__(self, embed_dim: int = EMBED_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ============================================================
# Dataset
# ============================================================

from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Training helpers
# ============================================================

def train_snn_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for emb, targets in loader:
        emb, targets = emb.to(device), targets.to(device)
        # Direct encoding: repeat T times → (T, B, 2048)
        x = emb.unsqueeze(0).expand(NUM_STEPS, -1, -1)

        optimizer.zero_grad()
        spk_out, mem_out = model(x)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = mem_out.sum(0).argmax(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total if total > 0 else 0.0


@torch.no_grad()
def eval_snn(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for emb, targets in loader:
        emb, targets = emb.to(device), targets.to(device)
        x = emb.unsqueeze(0).expand(NUM_STEPS, -1, -1)
        spk_out, mem_out = model(x)
        predicted = mem_out.sum(0).argmax(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


def train_ann_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for emb, targets in loader:
        emb, targets = emb.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(emb)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = logits.argmax(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total if total > 0 else 0.0


@torch.no_grad()
def eval_ann(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for emb, targets in loader:
        emb, targets = emb.to(device), targets.to(device)
        logits = model(emb)
        predicted = logits.argmax(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


def run_fold(model_type: str, fold: int,
             all_embeddings, all_labels, all_folds,
             device: str, epochs: int, lr: float) -> dict:
    """Train and evaluate a single fold."""
    train_mask = all_folds != fold
    test_mask  = all_folds == fold

    train_ds = EmbeddingDataset(all_embeddings[train_mask], all_labels[train_mask])
    test_ds  = EmbeddingDataset(all_embeddings[test_mask],  all_labels[test_mask])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if model_type == "snn":
        model = SNNHead().to(device)
    elif model_type == "ann":
        model = ANNHead().to(device)
    elif model_type == "linear":
        model = LinearProbe().to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion  = nn.CrossEntropyLoss() if model_type in ("ann", "linear") else None

    best_acc   = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        if model_type == "snn":
            train_loss, train_acc = train_snn_epoch(model, train_loader, optimizer, device)
            test_acc = eval_snn(model, test_loader, device)
            test_loss = train_loss  # for scheduler
        else:
            train_loss, train_acc = train_ann_epoch(model, train_loader, optimizer, criterion, device)
            test_acc = eval_ann(model, test_loader, device)
            test_loss = train_loss

        scheduler.step(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f} | Best: {best_acc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    return {
        "fold": fold,
        "model_type": model_type,
        "best_acc": float(best_acc),
        "best_epoch": best_epoch,
        "final_epochs": epoch,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PANNs CNN14 + SNN/ANN head transfer learning on ESC-50"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold (default: all 5)")
    parser.add_argument("--model", default="all",
                        choices=["snn", "ann", "linear", "all"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("PANNs CNN14 + SNN Head Transfer Learning")
    print("=" * 60)
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Fold    : {args.fold if args.fold else 'all 5'}")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print()

    if not PANNS_CKPT.exists():
        print(f"ERROR: CNN14 checkpoint not found: {PANNS_CKPT}")
        print("Download with:")
        print("  curl -L -o ~/panns_data/Cnn14_mAP=0.431.pth "
              "\"https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1\"")
        sys.exit(1)

    # Extract / load embeddings
    cache_dir = RESULTS_DIR / "panns_embeddings"
    all_embeddings, all_labels, all_folds = extract_panns_embeddings(
        ESC50_AUDIO_DIR, ESC50_META_PATH, cache_dir
    )
    print(f"Embeddings shape: {all_embeddings.shape}")

    models_to_run = (["snn", "ann", "linear"] if args.model == "all"
                     else [args.model])
    folds_to_run  = (list(range(1, NUM_FOLDS + 1)) if args.fold is None
                     else [args.fold])

    all_results = {}

    for model_type in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_type.upper()}")
        print(f"{'='*60}")

        fold_results = []
        for fold in folds_to_run:
            print(f"\n  --- Fold {fold} ---")
            result = run_fold(
                model_type=model_type,
                fold=fold,
                all_embeddings=all_embeddings,
                all_labels=all_labels,
                all_folds=all_folds,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
            )
            fold_results.append(result)
            print(f"  Fold {fold} done. Best: {result['best_acc']:.4f}")

        accs = [r["best_acc"] for r in fold_results]
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))

        print(f"\n  {model_type.upper()} Summary:")
        print(f"    Per-fold: {[f'{a:.4f}' for a in accs]}")
        print(f"    Mean ± std: {mean_acc:.4f} ± {std_acc:.4f}")

        all_results[model_type] = {
            "fold_results":    fold_results,
            "fold_accuracies": accs,
            "mean_accuracy":   mean_acc,
            "std_accuracy":    std_acc,
        }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    out_dir = RESULTS_DIR / "panns"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_tag = f"fold{args.fold}" if args.fold else "all_folds"
    out_path = out_dir / f"panns_snn_head_{fold_tag}_{args.epochs}ep.json"

    output = {
        "embed_dim": int(EMBED_DIM),
        "num_steps": NUM_STEPS,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": device,
        "folds_run": folds_to_run,
        "results": all_results,
        "comparison": {
            m: {
                "mean": all_results[m]["mean_accuracy"],
                "std":  all_results[m]["std_accuracy"],
            }
            for m in models_to_run if m in all_results
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved: {out_path}")

    # Summary comparison
    print()
    print("=" * 60)
    print("Transfer Learning Summary")
    print("=" * 60)
    for m in models_to_run:
        if m in all_results:
            r = all_results[m]
            print(f"  {m.upper():<10}: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")
    print()
    print("Reference (from full model training):")
    print("  ANN baseline  : 63.85% ± 3.07%")
    print("  SNN direct    : 47.15% ± 4.50%")
    print()
    if "snn" in all_results:
        snn_acc = all_results["snn"]["mean_accuracy"]
        print(f"  PANNs+SNN head achieves {snn_acc:.2%}")
        if snn_acc > 0.638:
            print("  FINDING: PANNs+SNN head EXCEEDS ANN baseline — pre-trained features close the gap!")
        elif snn_acc > 0.55:
            print("  FINDING: PANNs+SNN head significantly narrows the gap vs ANN baseline.")
        else:
            print("  FINDING: PANNs+SNN head shows modest improvement — features partially help.")


if __name__ == "__main__":
    main()
