"""
neurobench_analysis.py -- NeuroBench-compliant energy + efficiency metrics.

Measures:
  - Footprint (bytes), ParameterCount, ConnectionSparsity (static)
  - ActivationSparsity, SynapticOperations (workload)
  - Derived energy estimates (Eff_ACs × 0.9 pJ vs Eff_MACs × 4.6 pJ)

Reference:
  Yik et al. (2025). "The NeuroBench Framework." Nature Communications, 16, 1589.
  Energy constants: Lopez-Randulfe et al. (2024) and standard SNN literature.

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/neurobench_analysis.py
  python experiments/neurobench_analysis.py --fold 4 --num-samples 400
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils

from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.metrics.static import Footprint, ConnectionSparsity, ParameterCount
from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    ClassificationAccuracy,
)

from src.config import RESULTS_DIR, NUM_CLASSES, NUM_STEPS
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN
from src.encoding import encode_direct
from src.dataset import get_fold_dataloaders

# ============================================================
# Energy constants (32-bit float operations, standard values)
# ============================================================
# Source: Horowitz (2014), Dampfhoffer et al. (2023 IEEE TECI)
# MAC (multiply-accumulate): 4.6 pJ per 32-bit float operation
# AC  (accumulate-only):     0.9 pJ per 32-bit float operation
ENERGY_PER_MAC_PJ = 4.6   # pJ
ENERGY_PER_AC_PJ  = 0.9   # pJ


# ============================================================
# NeuroBench wrappers
# ============================================================

class NeuroBenchSNNWrapper(nn.Module):
    """Wrapper so our SpikingCNN works with NeuroBench custom_forward.

    NeuroBench expects: input (B, T, 1, H, W) → output (T, B, 50)
    (NeuroBench then transposes output to (B, T, 50) internally.)
    """

    def __init__(self, snn_model: SpikingCNN):
        super().__init__()
        self.snn = snn_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W) — NeuroBench format
        x_t = x.transpose(0, 1)         # → (T, B, 1, H, W)
        spk_out, mem_out = self.snn(x_t)  # both (T, B, 50)
        return spk_out                   # (T, B, 50) — NeuroBench transposes → (B, T, 50)

    def reset(self):
        utils.reset(self.snn)


class NeuroBenchANNWrapper(nn.Module):
    """Wrapper so ConvANN works with NeuroBench's SNNTorchModel.

    For ANN there is no time dimension, but NeuroBench needs (B, T, 50).
    We unsqueeze T=1 so metrics work correctly.
    """

    def __init__(self, ann_model: ConvANN):
        super().__init__()
        self.ann = ann_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 1, H, W) — one timestep (the raw mel spec)
        # Squeeze time dim, run ANN, add time dim back
        if x.dim() == 5:
            x = x[:, 0, ...]  # (B, 1, H, W)
        logits = self.ann(x)  # (B, 50)
        return logits.unsqueeze(0)  # (1, B, 50) → NeuroBench transposes → (B, 1, 50)

    def reset(self):
        pass  # ANN is stateless


# ============================================================
# Preprocessors
# ============================================================

def snn_preprocessor(batch):
    """Encode mel spec to direct spikes in NeuroBench (B, T, 1, H, W) format."""
    data, targets = batch
    # data: (B, 1, H, W)
    encoded = encode_direct(data, num_steps=NUM_STEPS)  # (T, B, 1, H, W)
    encoded = encoded.transpose(0, 1)                   # (B, T, 1, H, W)
    return encoded, targets


def ann_preprocessor(batch):
    """Add fake timestep dim for ANN so NeuroBench format is consistent."""
    data, targets = batch
    # data: (B, 1, H, W) → (B, 1, 1, H, W)
    return data.unsqueeze(1), targets


# ============================================================
# Postprocessors
# ============================================================

def snn_postprocessor(output: torch.Tensor) -> torch.Tensor:
    """Convert (B, T, 50) spikes to class predictions (B,).

    Uses total spike count (= rate decoding) across all timesteps.
    """
    return output.sum(dim=1).argmax(dim=1)  # (B,)


def ann_postprocessor(output: torch.Tensor) -> torch.Tensor:
    """Convert (B, 1, 50) logits to class predictions (B,)."""
    return output[:, 0, :].argmax(dim=1)   # (B,)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NeuroBench energy + efficiency analysis for SNN vs ANN"
    )
    parser.add_argument("--fold", type=int, default=4,
                        help="Test fold to evaluate on (default: 4)")
    parser.add_argument("--num-samples", type=int, default=400,
                        help="Number of test samples (default: 400 = full fold)")
    parser.add_argument("--device", default=None,
                        help="Device override (default: auto)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("NeuroBench Analysis: SNN vs ANN")
    print("=" * 60)
    print(f"  Fold    : {args.fold}")
    print(f"  Samples : {args.num_samples}")
    print(f"  Device  : {device}")
    print()

    # Load models
    snn_path = RESULTS_DIR / "snn" / "direct" / f"best_fold{args.fold}.pt"
    ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{args.fold}.pt"

    for p in [snn_path, ann_path]:
        if not p.exists():
            print(f"FATAL: Model not found: {p}")
            sys.exit(1)

    snn_model = SpikingCNN().to(device)
    snn_model.load_state_dict(
        torch.load(snn_path, map_location=device, weights_only=True)
    )
    snn_model.eval()
    print(f"Loaded SNN : {snn_path}")

    ann_model = ConvANN().to(device)
    ann_model.load_state_dict(
        torch.load(ann_path, map_location=device, weights_only=True)
    )
    ann_model.eval()
    print(f"Loaded ANN : {ann_path}")

    # Data loader (use a subset if requested)
    _, test_loader = get_fold_dataloaders(
        args.fold, batch_size=32, augment=False
    )

    # --------------------------------------------------------
    # NeuroBench model wrappers
    # --------------------------------------------------------
    snn_wrapper = NeuroBenchSNNWrapper(snn_model)
    ann_wrapper = NeuroBenchANNWrapper(ann_model)

    nb_snn = SNNTorchModel(snn_wrapper, custom_forward=True)
    nb_ann = SNNTorchModel(ann_wrapper, custom_forward=True)

    static_metrics   = [Footprint, ParameterCount, ConnectionSparsity]
    workload_metrics = [ClassificationAccuracy, ActivationSparsity, SynapticOperations]

    print()
    print("Running SNN benchmark...")
    bench_snn = Benchmark(
        nb_snn, test_loader,
        preprocessors=[snn_preprocessor],
        postprocessors=[snn_postprocessor],
        metric_list=[static_metrics, workload_metrics],
    )
    results_snn = bench_snn.run(device=device)
    print("SNN results:", results_snn)

    print()
    print("Running ANN benchmark...")
    bench_ann = Benchmark(
        nb_ann, test_loader,
        preprocessors=[ann_preprocessor],
        postprocessors=[ann_postprocessor],
        metric_list=[static_metrics, workload_metrics],
    )
    results_ann = bench_ann.run(device=device)
    print("ANN results:", results_ann)

    # --------------------------------------------------------
    # Energy estimates
    # --------------------------------------------------------
    # SynapticOperations returns dict with keys:
    #   "Effective_MACs" — multiply-accumulate ops (ANN layers)
    #   "Effective_ACs"  — accumulate-only ops     (SNN spike events)
    #   "Dense"          — theoretical if all neurons active (no sparsity)
    # Values are cumulative totals for ALL samples in the dataloader.
    snn_synops = results_snn.get("SynapticOperations", {})
    ann_synops = results_ann.get("SynapticOperations", {})

    # Total counts across the whole test set
    snn_eff_acs_total  = snn_synops.get("Effective_ACs", 0.0)
    ann_eff_macs_total = ann_synops.get("Effective_MACs", 0.0)
    snn_dense_total    = snn_synops.get("Dense", 0.0)
    ann_dense_total    = ann_synops.get("Dense", 0.0)

    # NeuroBench reports cumulative totals; normalise to per-sample
    n_test = len(test_loader.dataset)
    snn_eff_acs  = snn_eff_acs_total  / max(n_test, 1)
    ann_eff_macs = ann_eff_macs_total / max(n_test, 1)
    snn_dense    = snn_dense_total    / max(n_test, 1)
    ann_dense    = ann_dense_total    / max(n_test, 1)

    snn_energy_pj = snn_eff_acs  * ENERGY_PER_AC_PJ
    ann_energy_pj = ann_eff_macs * ENERGY_PER_MAC_PJ

    energy_ratio = ann_energy_pj / snn_energy_pj if snn_energy_pj > 0 else float("inf")

    print()
    print("=" * 60)
    print("Energy Estimates (32-bit, theoretical)")
    print("=" * 60)
    print(f"  n_test samples             : {n_test}")
    print(f"  SNN Effective_ACs/sample   : {snn_eff_acs:.2e}")
    print(f"  SNN Dense ops/sample       : {snn_dense:.2e}")
    print(f"  ANN Effective_MACs/sample  : {ann_eff_macs:.2e}")
    print(f"  ANN Dense ops/sample       : {ann_dense:.2e}")
    print(f"  SNN energy est.            : {snn_energy_pj:.2e} pJ/sample  "
          f"({snn_eff_acs/max(ann_eff_macs,1):.1f}× more ACs than ANN MACs)")
    print(f"  ANN energy est.            : {ann_energy_pj:.2e} pJ/sample")
    print(f"  ANN/SNN energy ratio       : {energy_ratio:.2f}×  ", end="")
    if energy_ratio > 1:
        print("(SNN cheaper on neuromorphic hardware)")
    else:
        print("(ANN cheaper in software simulation — expected: T timesteps overhead)")
    print()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    def _make_serialisable(d):
        """Convert torch tensors / non-JSON types to Python scalars."""
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.item() if v.numel() == 1 else v.tolist()
            elif isinstance(v, dict):
                out[k] = _make_serialisable(v)
            else:
                try:
                    json.dumps(v)
                    out[k] = v
                except TypeError:
                    out[k] = str(v)
        return out

    output = {
        "fold": args.fold,
        "num_samples": args.num_samples,
        "device": device,
        "num_steps": NUM_STEPS,
        "energy_constants": {
            "energy_per_mac_pj": ENERGY_PER_MAC_PJ,
            "energy_per_ac_pj": ENERGY_PER_AC_PJ,
        },
        "snn": _make_serialisable(results_snn),
        "ann": _make_serialisable(results_ann),
        "energy": {
            "n_test_samples":       n_test,
            "snn_eff_acs_per_sample":  snn_eff_acs,
            "ann_eff_macs_per_sample": ann_eff_macs,
            "snn_dense_per_sample":    snn_dense,
            "ann_dense_per_sample":    ann_dense,
            "snn_energy_pj_per_sample": snn_energy_pj,
            "ann_energy_pj_per_sample": ann_energy_pj,
            "ann_snn_energy_ratio":     energy_ratio,
            "note": (
                "SNN ACs are accum-only on neuromorphic hardware (0.9 pJ each). "
                "ANN MACs require multiply+add (4.6 pJ each). "
                "On SpiNNaker: SNN is cheaper because ACs replace MACs. "
                "In software simulation: SNN is more expensive due to T=25 timesteps."
            ),
        },
    }

    out_dir = RESULTS_DIR / "neurobench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"analysis_fold{args.fold}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
