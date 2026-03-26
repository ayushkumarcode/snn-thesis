"""
knowledge_distillation.py -- Knowledge distillation: ANN teacher -> SNN student.

Research question: Can a trained ANN teacher guide an SNN student to higher
accuracy than training from scratch, using soft label knowledge distillation?

Method:
  - ANN teacher: pre-trained ConvANN from results/ann/none/best_fold{fold}.pt
    (frozen, eval mode -- never updated)
  - SNN student: EnhancedSpikingCNN with learn_beta=True, learn_threshold=True,
    Dropout(0.3), spike_rate_escape surrogate gradient
  - Student logits: membrane potential summed across timesteps (mem_out.sum(dim=0))
  - Combined loss: alpha * CE(student_logits, labels) +
                   (1-alpha) * T^2 * KL_div(student_soft, teacher_soft)
  - Soft labels: logits / T before softmax, with T=3.0 (temperature)
  - alpha=0.5 (equal weight CE and KD loss)
  - 5-fold CV, save results to results/experiments/knowledge_distillation/

Key reference:
  - Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
  - Kushawaha et al. (2021) "Distilling Spikes" -- ANN->SNN distillation

Usage:
  cd snn-esc50
  source .venv/bin/activate
  python experiments/knowledge_distillation.py
  python experiments/knowledge_distillation.py --fold 1 --epochs 50
  python experiments/knowledge_distillation.py --temperature 4.0 --alpha 0.3
  python experiments/knowledge_distillation.py --fold 0  # all 5 folds
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
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, NUM_FOLDS,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.ann_model import ConvANN


# ============================================================
# Enhanced SNN student with learnable parameters
# ============================================================

class EnhancedSpikingCNN(nn.Module):
    """SpikingCNN student with learnable beta, threshold, dropout,
    and spike_rate_escape surrogate gradient.

    Same architecture dimensions as ConvANN/SpikingCNN for fair comparison.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
    ):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)

        # -- Convolutional feature extraction --
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        # After two MaxPool2d(2) on input (64, 216): (16, 54)
        # AvgPool2d(4,6) on (16,54) -> (4, 9)
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # -- Fully connected classifier with dropout --
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.dropout = nn.Dropout(0.3)
        self.lif3 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over all timesteps.

        Args:
            x: Input of shape (num_steps, batch, 1, n_mels, time_frames).

        Returns:
            spk_out: Output spikes, shape (num_steps, batch, num_classes).
            mem_out: Output membrane potentials, shape (num_steps, batch, num_classes).
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            x_t = x[step]

            # Conv block 1
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Conv block 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Pool + flatten
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            # FC block 1 with dropout
            cur3 = self.dropout(self.fc1(flat))
            spk3, mem3 = self.lif3(cur3, mem3)

            # FC block 2 (output)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


# ============================================================
# Knowledge distillation loss
# ============================================================

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Combined knowledge distillation loss.

    Loss = alpha * CE(student_logits, labels)
         + (1 - alpha) * T^2 * KL(student_soft || teacher_soft)

    The T^2 factor compensates for the 1/T^2 scaling in gradients from
    softened probabilities (Hinton et al. 2015).

    Args:
        student_logits: Raw logits from SNN student, shape (batch, num_classes).
        teacher_logits: Raw logits from ANN teacher, shape (batch, num_classes).
        labels: Ground truth class indices, shape (batch,).
        temperature: Softmax temperature for soft labels.
        alpha: Weight for hard label CE loss (1-alpha for KD loss).

    Returns:
        Combined scalar loss.
    """
    # Hard label loss
    ce_loss = F.cross_entropy(student_logits, labels)

    # Soft label loss
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

    # Combined loss with T^2 scaling
    return alpha * ce_loss + (1 - alpha) * (temperature ** 2) * kd_loss


# ============================================================
# Training / evaluation
# ============================================================

def train_epoch(student, teacher, loader, optimizer, device, temperature, alpha):
    """Train SNN student with KD loss for one epoch."""
    student.train()
    teacher.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=student.num_steps)

        optimizer.zero_grad()

        # Student forward: SNN with temporal dynamics
        spk_out, mem_out = student(spk_input)
        student_logits = mem_out.sum(dim=0)  # (batch, num_classes)

        # Teacher forward: ANN (frozen, no grad)
        with torch.no_grad():
            teacher_logits = teacher(data)  # (batch, num_classes)

        loss = distillation_loss(
            student_logits, teacher_logits, targets,
            temperature=temperature, alpha=alpha,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = student_logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_model(model, loader, device):
    """Evaluate SNN student on a dataset (standard CE accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=model.num_steps)

        spk_out, mem_out = model(spk_input)
        student_logits = mem_out.sum(dim=0)

        loss = criterion(student_logits, targets)
        total_loss += loss.item()

        predicted = student_logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


# ============================================================
# Single fold training
# ============================================================

def train_fold(fold: int, device, epochs: int, patience: int,
               temperature: float, alpha: float):
    """Train KD student on a single fold.

    Returns:
        Dict with fold results.
    """
    print(f"\n{'='*60}")
    print(f"  Knowledge Distillation: ANN teacher -> SNN student | Fold {fold}/5")
    print(f"  Device: {device} | Epochs: {epochs} | T={temperature} | alpha={alpha}")
    print(f"{'='*60}")

    # Load teacher ANN (frozen)
    ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"
    if not ann_path.exists():
        print(f"FATAL: ANN teacher weights not found: {ann_path}")
        sys.exit(1)

    teacher = ConvANN().to(device)
    teacher.load_state_dict(
        torch.load(ann_path, map_location=device, weights_only=True)
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"  Loaded teacher ANN: {ann_path}")

    # Create student SNN (random init)
    student = EnhancedSpikingCNN().to(device)
    print(f"  Student: EnhancedSpikingCNN (learn_beta, learn_threshold, dropout, sre)")

    # Data
    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)

    # Evaluate teacher accuracy for reference
    _, teacher_acc = eval_teacher(teacher, test_loader, device)
    print(f"  Teacher test accuracy: {teacher_acc:.4f}")

    # Optimizer
    optimizer = torch.optim.Adam(
        student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
    }

    out_dir = RESULTS_DIR / "experiments" / "knowledge_distillation"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(
            student, teacher, train_loader, optimizer, device,
            temperature=temperature, alpha=alpha,
        )
        te_loss, te_acc = eval_model(student, test_loader, device)
        scheduler.step(te_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(student.state_dict(), out_dir / f"best_fold{fold}.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train: {tr_acc:.4f} | Test: {te_acc:.4f} | "
                f"Best: {best_acc:.4f} (ep {best_epoch}) | {elapsed:.0f}s"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"  Fold {fold} done in {elapsed:.1f}s | Best: {best_acc:.4f}")

    result = {
        "fold": fold,
        "method": "knowledge_distillation",
        "teacher_acc": teacher_acc,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "time_seconds": round(elapsed, 1),
        "temperature": temperature,
        "alpha": alpha,
        "lr": LEARNING_RATE,
        "epochs_max": epochs,
        "patience": patience,
        "student_features": ["learn_beta", "learn_threshold", "dropout_0.3", "spike_rate_escape"],
        "history": history,
    }

    with open(out_dir / f"result_fold{fold}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


@torch.no_grad()
def eval_teacher(teacher, loader, device):
    """Evaluate ANN teacher accuracy."""
    teacher.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = teacher(data)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        predicted = logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: ANN teacher -> SNN student"
