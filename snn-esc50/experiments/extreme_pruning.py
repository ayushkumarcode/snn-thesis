"""
extreme_pruning.py -- Iterative magnitude pruning to 95%/99% sparsity.

We showed 90% pruning retains 93.2% of accuracy. Push further with
iterative pruning + fine-tuning between rounds on the rhythm model.

Usage:
    python -m experiments.extreme_pruning --device cuda
    python -m experiments.extreme_pruning --fold 1 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    NUM_CLASSES, NUM_STEPS, RESULTS_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from experiments.combo_experiment import ComboSpikingCNN


def build_args(rhythm=True):
    """Build args for ComboSpikingCNN."""
    class Args:
        pass
    a = Args()
    a.rhythm = rhythm
    a.dendritic = False
    a.delays = False
    a.learn_beta = True
    a.learn_threshold = False
    a.dropout = True
    a.sre = True
    a.kd = False
    a.hybrid_init = False
    a.tet = False
    a.cochleagram = False
    a.l1_reg = 0.0
    a.branches = 3
    a.max_delay = 5
    return a


def apply_global_pruning(model, target_sparsity):
    """Apply global L1 unstructured pruning."""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, "weight"))
    if params:
        prune.global_unstructured(
            params, pruning_method=prune.L1Unstructured,
            amount=target_sparsity)


def remove_pruning(model):
    """Make pruning permanent."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass


def count_sparsity(model):
    """Count actual weight sparsity."""
    total = 0
    zeros = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    return zeros / total if total > 0 else 0


def fine_tune(model, train_loader, device, epochs=10, lr=1e-4):
    """Fine-tune after pruning."""
    model.train()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY)
    use_amp = device.type == 'cuda'

    for epoch in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            spk_input = encode_direct(data).to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                     enabled=use_amp):
                _, mem_out, _ = model(spk_input)
                T, B, C = mem_out.shape
                loss = F.cross_entropy(
                    mem_out.reshape(T * B, C),
                    targets.unsqueeze(0).expand(T, -1).reshape(-1),
                )
            loss.backward()
            optimizer.step()


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            _, mem_out, _ = model(spk_input)

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return correct / total


def run_fold(fold, device, sparsity_levels):
    print(f"\n  Fold {fold}")

    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    # Load trained rhythm model
    model_path = (RESULTS_DIR / "experiments" /
                  "combo_rhythm_lbeta_drop_sre" / f"best_fold{fold}.pt")
    if not model_path.exists():
        print(f"  WARNING: {model_path} not found")
        return []

    args = build_args(rhythm=True)
    model = ComboSpikingCNN(args).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))

    # Baseline accuracy
    base_acc = eval_model(model, test_loader, device)
    print(f"  Baseline (0% pruning): {base_acc:.4f}")

    results = [{
        "fold": fold, "target_sparsity": 0.0, "actual_sparsity": 0.0,
        "accuracy": base_acc, "retention": 1.0,
    }]

    # Iterative pruning: prune in steps, fine-tune between
    prev_sparsity = 0.0
    for target in sparsity_levels:
        # Reload fresh model for each target (prune from scratch)
        model = ComboSpikingCNN(args).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True))

        # Iterative pruning in 10% steps with fine-tuning
        current = 0.0
        step_size = 0.1
        while current < target:
            next_target = min(current + step_size, target)
            # Incremental amount to prune from remaining weights
            incremental = (next_target - current) / (1.0 - current + 1e-10)
            incremental = min(incremental, 0.99)

            apply_global_pruning(model, incremental)
            fine_tune(model, train_loader, device, epochs=5, lr=1e-4)
            current = count_sparsity(model)

        remove_pruning(model)
        actual_sparsity = count_sparsity(model)
        acc = eval_model(model, test_loader, device)
        retention = acc / base_acc if base_acc > 0 else 0

        print(f"  Pruning {target*100:.0f}% (actual {actual_sparsity*100:.1f}%): "
              f"acc={acc:.4f} (retention={retention:.4f})")

        results.append({
            "fold": fold, "target_sparsity": target,
            "actual_sparsity": actual_sparsity,
            "accuracy": acc, "retention": retention,
            "energy_reduction_x": 1.0 / (1.0 - actual_sparsity + 1e-10),
        })

        # Save pruned model
        save_dir = RESULTS_DIR / "energy" / f"pruned_{int(target*100)}"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   save_dir / f"best_fold{fold}.pt")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    download_esc50()

    sparsity_levels = [0.5, 0.7, 0.9, 0.95, 0.97, 0.99]
    folds = [args.fold] if args.fold else list(range(1, 6))

    all_results = []
    for fold in folds:
        results = run_fold(fold, device, sparsity_levels)
        all_results.extend(results)

    # Summary
