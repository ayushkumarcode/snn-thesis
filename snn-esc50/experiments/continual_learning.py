"""
continual_learning.py -- SNN vs ANN catastrophic forgetting comparison.

Research question: Does the temporal spike dynamics of SNNs provide any
natural protection against catastrophic forgetting compared to ANNs?

Method:
  - ESC-50 has 5 predefined super-categories (10 classes each):
      Task 0: Animals    (classes 0–9)
      Task 1: Nature     (classes 10–19)
      Task 2: Human      (classes 20–29)
      Task 3: Domestic   (classes 30–39)
      Task 4: Urban      (classes 40–49)
  - Models start from RANDOM init (no pre-training) for fair comparison.
    OR start from a pre-trained checkpoint (use --pretrained flag).
  - Sequential fine-tuning: train 20 epochs on each task in order.
  - After each task, evaluate on ALL previously seen classes.
  - Measure backward transfer (BWT) = forgetting.

Key papers:
  - Golden et al. (2022) PLoS Comp. Bio.: Sleep prevents catastrophic forgetting in SNNs.
  - Zhang et al. (2023) Science Advances: NACA continual learning.

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/continual_learning.py
  python experiments/continual_learning.py --pretrained --fold 4
  python experiments/continual_learning.py --epochs-per-task 20 --fold 4
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import RESULTS_DIR, NUM_CLASSES, NUM_STEPS, BATCH_SIZE
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN
from src.encoding import encode_direct
from src.dataset import ESC50Dataset, get_class_labels

# ============================================================
# ESC-50 super-category task structure
# ============================================================
# ESC-50 target values 0-49 map to 5 super-categories:
#   Animals:  0-9   | Nature: 10-19 | Human:    20-29
#   Domestic: 30-39 | Urban:  40-49
TASKS = [
    ("Animals",  list(range(0,  10))),
    ("Nature",   list(range(10, 20))),
    ("Human",    list(range(20, 30))),
    ("Domestic", list(range(30, 40))),
    ("Urban",    list(range(40, 50))),
]


# ============================================================
# Dataset helpers
# ============================================================

def get_class_subset_loader(dataset: ESC50Dataset, class_range: list,
                             batch_size: int = BATCH_SIZE) -> DataLoader:
    """Return DataLoader for a specific subset of classes from a dataset."""
    class_set = set(class_range)
    indices = [i for i, label in enumerate(dataset.labels) if label in class_set]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)


def get_class_subset_eval_loader(dataset: ESC50Dataset,
                                  class_range: list, batch_size: int = 32) -> DataLoader:
    """Return DataLoader (no shuffle) for evaluation on a class subset."""
    class_set = set(class_range)
    indices = [i for i, label in enumerate(dataset.labels) if label in class_set]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


# ============================================================
# Train / eval helpers
# ============================================================

def train_snn_epoch(model, loader, optimizer, device):
    """One epoch of SNN training (direct encoding, summed CE loss)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=NUM_STEPS)
        optimizer.zero_grad()
        spk_out, mem_out = model(spk_input)

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
    """Evaluate SNN accuracy on a loader."""
    model.eval()
    correct = 0
    total = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=NUM_STEPS)
        spk_out, mem_out = model(spk_input)
        predicted = mem_out.sum(0).argmax(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


def train_ann_epoch(model, loader, optimizer, device):
    """One epoch of ANN training."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(data)
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
    """Evaluate ANN accuracy on a loader."""
    model.eval()
    correct = 0
    total = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        predicted = logits.argmax(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


# ============================================================
# Continual learning loop
# ============================================================

def run_continual_learning(
    model_type: str,
    train_dataset: ESC50Dataset,
    test_dataset: ESC50Dataset,
    device: str,
    epochs_per_task: int,
    lr: float,
    pretrained_path: Path = None,
) -> dict:
    """Run sequential task training and measure forgetting.

    Returns:
        Dict with:
          - accuracy_matrix[i][j] = accuracy on task j after training on task i
          - backward_transfer: forgetting per task
          - final_accuracy: accuracy on all tasks after full training
    """
    # Initialise model
    if model_type == "snn":
        model = SpikingCNN().to(device)
    else:
        model = ConvANN().to(device)

    if pretrained_path and pretrained_path.exists():
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
        print(f"  Loaded pretrained: {pretrained_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    n_tasks = len(TASKS)
    # accuracy_matrix[after_task][eval_task]
    accuracy_matrix = [[None] * n_tasks for _ in range(n_tasks)]

    # Initial evaluation (before any training) — only on task 0
    print(f"\n  Initial accuracy (before training):")
    for j, (task_name, class_range) in enumerate(TASKS):
        eval_loader = get_class_subset_eval_loader(test_dataset, class_range)
        if len(eval_loader.dataset) == 0:
            acc = 0.0
        else:
            if model_type == "snn":
                acc = eval_snn(model, eval_loader, device)
            else:
                acc = eval_ann(model, eval_loader, device)
        print(f"    Task {j} ({task_name}): {acc:.4f}")

    seen_classes = []

    for task_id, (task_name, class_range) in enumerate(TASKS):
        seen_classes.extend(class_range)
        print(f"\n  --- Task {task_id}: {task_name} (classes {class_range[0]}-{class_range[-1]}) ---")

        task_loader = get_class_subset_loader(train_dataset, class_range)

        if len(task_loader.dataset) == 0:
            print(f"    WARNING: No training samples for task {task_id}")
            continue

        for epoch in range(1, epochs_per_task + 1):
            if model_type == "snn":
                loss, acc = train_snn_epoch(model, task_loader, optimizer, device)
            else:
                loss, acc = train_ann_epoch(model, task_loader, optimizer, device)

            if epoch % 5 == 0 or epoch == epochs_per_task:
                print(f"    Epoch {epoch:3d} | Loss: {loss:.4f} Acc: {acc:.4f}")

        # Evaluate on ALL tasks seen so far
        print(f"  Evaluation after task {task_id}:")
        for j, (eval_name, eval_range) in enumerate(TASKS):
            if j <= task_id:
                eval_loader = get_class_subset_eval_loader(test_dataset, eval_range)
                if model_type == "snn":
                    acc = eval_snn(model, eval_loader, device)
                else:
                    acc = eval_ann(model, eval_loader, device)
                accuracy_matrix[task_id][j] = acc
                print(f"    Task {j} ({eval_name}): {acc:.4f}")
            else:
                accuracy_matrix[task_id][j] = None

    # --------------------------------------------------------
    # Compute backward transfer (BWT) and forgetting
    # --------------------------------------------------------
    # BWT[j] = accuracy on task j after training all tasks − accuracy right after task j
    # Negative BWT = forgetting
    bwt = []
    forgetting = []
    for j in range(n_tasks - 1):  # last task can't be forgotten
        acc_after_task_j = accuracy_matrix[j][j]   # right after learning task j
        acc_final = accuracy_matrix[n_tasks - 1][j]  # after all tasks
        if acc_after_task_j is not None and acc_final is not None:
            bwt_j = acc_final - acc_after_task_j
            bwt.append(bwt_j)
            forgetting.append(max(0.0, acc_after_task_j - acc_final))

    mean_bwt = np.mean(bwt) if bwt else 0.0
    mean_forgetting = np.mean(forgetting) if forgetting else 0.0

    # Final accuracy on all tasks
    final_accs = [accuracy_matrix[n_tasks - 1][j] for j in range(n_tasks)
                  if accuracy_matrix[n_tasks - 1][j] is not None]
    mean_final_acc = np.mean(final_accs) if final_accs else 0.0

    print(f"\n  Summary for {model_type.upper()}:")
    print(f"    Mean BWT (backward transfer): {mean_bwt:+.4f}")
    print(f"    Mean forgetting:              {mean_forgetting:.4f}")
    print(f"    Final avg accuracy:           {mean_final_acc:.4f}")

    return {
        "model_type": model_type,
        "accuracy_matrix": accuracy_matrix,
        "backward_transfer": bwt,
        "mean_bwt": float(mean_bwt),
        "forgetting": forgetting,
        "mean_forgetting": float(mean_forgetting),
        "final_accuracies": final_accs,
        "mean_final_accuracy": float(mean_final_acc),
        "task_names": [t[0] for t in TASKS],
        "task_class_ranges": [[c[0], c[-1]] for _, c in TASKS],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SNN vs ANN continual learning")
    parser.add_argument("--fold", type=int, default=4,
                        help="Test fold (default: 4)")
    parser.add_argument("--epochs-per-task", type=int, default=20,
                        help="Training epochs per task (default: 20)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for fine-tuning (default: 5e-4)")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Start from fold-N pretrained checkpoint (default: random init)")
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
    print("Continual Learning: SNN vs ANN Catastrophic Forgetting")
    print("=" * 60)
    print(f"  Fold            : {args.fold}")
    print(f"  Epochs per task : {args.epochs_per_task}")
    print(f"  LR              : {args.lr}")
    print(f"  Pretrained init : {args.pretrained}")
    print(f"  Device          : {device}")
    print()
    print("Task structure:")
    for i, (name, cls_range) in enumerate(TASKS):
        print(f"  Task {i}: {name} (classes {cls_range[0]}–{cls_range[-1]})")
    print()

    # Load full dataset (train folds + test fold separately)
    train_folds = [f for f in range(1, 6) if f != args.fold]
    test_fold   = [args.fold]

    print("Loading train dataset...")
    train_dataset = ESC50Dataset(folds=train_folds)
    print("Loading test dataset...")
    test_dataset  = ESC50Dataset(folds=test_fold)

    # Pretrained paths
    snn_pretrained = (
        RESULTS_DIR / "snn" / "direct" / f"best_fold{args.fold}.pt"
        if args.pretrained else None
    )
    ann_pretrained = (
        RESULTS_DIR / "ann" / "none" / f"best_fold{args.fold}.pt"
        if args.pretrained else None
    )

    results = {}

    for model_type, pretrained_path in [("snn", snn_pretrained),
                                         ("ann", ann_pretrained)]:
        print(f"\n{'='*60}")
        print(f"Running {model_type.upper()} continual learning")
        print(f"{'='*60}")
        res = run_continual_learning(
            model_type=model_type,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            epochs_per_task=args.epochs_per_task,
            lr=args.lr,
            pretrained_path=pretrained_path,
        )
        results[model_type] = res

    # --------------------------------------------------------
    # Comparison summary
    # --------------------------------------------------------
    print()
    print("=" * 60)
    print("Comparison: SNN vs ANN Forgetting")
    print("=" * 60)
    snn_res = results["snn"]
    ann_res = results["ann"]

    print(f"  SNN mean forgetting : {snn_res['mean_forgetting']:.4f}")
    print(f"  ANN mean forgetting : {ann_res['mean_forgetting']:.4f}")
    print(f"  SNN mean BWT        : {snn_res['mean_bwt']:+.4f}")
    print(f"  ANN mean BWT        : {ann_res['mean_bwt']:+.4f}")

    if snn_res["mean_forgetting"] < ann_res["mean_forgetting"]:
        print("  FINDING: SNN forgets LESS than ANN (better continual learning).")
    elif snn_res["mean_forgetting"] > ann_res["mean_forgetting"]:
        print("  FINDING: ANN forgets LESS than SNN (SNN does not help).")
    else:
        print("  FINDING: SNN and ANN forget similarly.")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    out_dir = RESULTS_DIR / "continual_learning"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "pretrained" if args.pretrained else "random"
    out_path = out_dir / f"forgetting_fold{args.fold}_{mode}_{args.epochs_per_task}ep.json"

    output = {
        "fold": args.fold,
        "epochs_per_task": args.epochs_per_task,
        "lr": args.lr,
        "pretrained": args.pretrained,
        "device": device,
        "tasks": [{"name": n, "class_range": list(c)} for n, c in TASKS],
        "snn": snn_res,
        "ann": ann_res,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
