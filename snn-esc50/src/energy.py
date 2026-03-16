"""
Energy estimation for SNN vs ANN comparison.

Measures computational cost using:
  1. Synaptic Operations (SynOps) counting for SNNs
  2. Multiply-Accumulate Operations (MACs) for ANNs
  3. NeuroBench wrapper (if available)

The energy ratio SynOps/MACs gives a hardware-independent measure of
the SNN's computational advantage, since SynOps are event-driven
(only occur on spikes) while MACs happen on every input.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.config import RESULTS_DIR, NUM_STEPS, get_device


def count_snn_synops(model, sample_input: torch.Tensor) -> dict:
    """Count synaptic operations for an SNN forward pass.

    A synaptic operation occurs when a pre-synaptic spike is transmitted
    through a connection (weight). SynOps = sum of (spikes * fan-out)
    across all layers.

    Args:
        model: SpikingCNN model.
        sample_input: Encoded input, shape (num_steps, batch, C, H, W).

    Returns:
        Dict with SynOps counts per layer and total.
    """
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    # Hook to capture intermediate spike counts
    spike_counts = {}
    layer_fanouts = {}

    def _get_fanout(module):
        """Calculate fan-out (number of outgoing connections per neuron)."""
        if isinstance(module, nn.Conv2d):
            return module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            return module.out_features
        return 0

    # Register fan-outs for weight layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_fanouts[name] = _get_fanout(module)

    # Run forward pass and count spikes at each LIF layer
    with torch.no_grad():
        mem1 = model.lif1.init_leaky()
        mem2 = model.lif2.init_leaky()
        mem3 = model.lif3.init_leaky()
        mem4 = model.lif4.init_leaky()

        spk_counts = {"lif1": 0, "lif2": 0, "lif3": 0, "lif4": 0}

        for step in range(model.num_steps):
            x_t = sample_input[step]

            cur1 = model.pool1(model.bn1(model.conv1(x_t)))
            spk1, mem1 = model.lif1(cur1, mem1)
            spk_counts["lif1"] += spk1.sum().item()

            cur2 = model.pool2(model.bn2(model.conv2(spk1)))
            spk2, mem2 = model.lif2(cur2, mem2)
            spk_counts["lif2"] += spk2.sum().item()

            pooled = model.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = model.fc1(flat)
            spk3, mem3 = model.lif3(cur3, mem3)
            spk_counts["lif3"] += spk3.sum().item()

            cur4 = model.fc2(spk3)
            spk4, mem4 = model.lif4(cur4, mem4)
            spk_counts["lif4"] += spk4.sum().item()

    batch_size = sample_input.shape[1]

    # SynOps = spikes * fan-out of the NEXT layer
    # lif1 spikes feed into conv2 (fan-out = out_channels * k * k)
    # lif2 spikes feed into fc1 (fan-out = fc1.out_features, but through adaptive pool)
    # lif3 spikes feed into fc2 (fan-out = fc2.out_features)
    synops_per_layer = {
        "conv1→lif1": spk_counts["lif1"],
        "conv2→lif2": spk_counts["lif2"],
        "fc1→lif3": spk_counts["lif3"],
        "fc2→lif4": spk_counts["lif4"],
    }

    # Weighted SynOps: spikes * fan-out
    conv2_fanout = model.conv2.out_channels * model.conv2.kernel_size[0] * model.conv2.kernel_size[1]
    fc1_fanout = model.fc1.out_features
    fc2_fanout = model.fc2.out_features

    weighted_synops = (
        spk_counts["lif1"] * conv2_fanout +
        spk_counts["lif2"] * fc1_fanout +
        spk_counts["lif3"] * fc2_fanout +
        spk_counts["lif4"]  # output layer, no further fanout
    )

    total_spikes = sum(spk_counts.values())

    # Sparsity: fraction of possible spikes that actually fire
    # Total possible = sum of (neurons_per_layer * num_steps * batch_size)
    # We approximate from spike counts
    avg_sparsity = {}
    for name, count in spk_counts.items():
        avg_sparsity[name] = count  # raw spike count (divide by possible for rate)

    return {
        "spike_counts": spk_counts,
        "synops_per_layer": synops_per_layer,
        "weighted_synops_total": weighted_synops,
        "weighted_synops_per_sample": weighted_synops / batch_size,
        "total_spikes": total_spikes,
        "total_spikes_per_sample": total_spikes / batch_size,
    }


def count_ann_macs(model, sample_input: torch.Tensor) -> dict:
    """Count Multiply-Accumulate operations for ANN forward pass.

    For Conv2d: MACs = out_h * out_w * out_c * in_c * k_h * k_w
    For Linear: MACs = in_features * out_features

    Args:
        model: ConvANN model.
        sample_input: Input tensor, shape (batch, C, H, W).

    Returns:
        Dict with MAC counts per layer and total.
    """
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    macs_per_layer = {}
    hooks = []

    def _hook_fn(name):
        def hook(module, inp, out):
            if isinstance(module, nn.Conv2d):
                # MACs = output_elements * kernel_ops
                out_h, out_w = out.shape[2], out.shape[3]
                macs = (
                    out_h * out_w * module.out_channels *
                    module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                )
                macs_per_layer[name] = macs
            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
                macs_per_layer[name] = macs
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(_hook_fn(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()

    total_macs = sum(macs_per_layer.values())

    return {
        "macs_per_layer": macs_per_layer,
        "total_macs": total_macs,
        "total_macs_per_sample": total_macs,  # same regardless of batch
    }


def compare_energy(snn_model, ann_model, snn_input: torch.Tensor,
                   ann_input: torch.Tensor) -> dict:
    """Compare energy between SNN and ANN.

    Uses the common assumption that one SynOp ≈ 0.9 pJ on neuromorphic
    hardware (e.g., Loihi) and one MAC ≈ 4.6 pJ on a 45nm CMOS processor.

    Args:
        snn_model: Trained SpikingCNN.
        ann_model: Trained ConvANN.
        snn_input: Encoded spike input (num_steps, batch, C, H, W).
        ann_input: Raw spectrogram input (batch, C, H, W).

    Returns:
        Dict with energy comparison.
    """
    SYNOP_ENERGY_PJ = 0.9   # pJ per SynOp (Loihi-like)
    MAC_ENERGY_PJ = 4.6     # pJ per MAC (45nm CMOS)

    snn_stats = count_snn_synops(snn_model, snn_input)
    ann_stats = count_ann_macs(ann_model, ann_input)

    snn_energy = snn_stats["weighted_synops_per_sample"] * SYNOP_ENERGY_PJ
    ann_energy = ann_stats["total_macs_per_sample"] * MAC_ENERGY_PJ

    ratio = snn_energy / ann_energy if ann_energy > 0 else float("inf")

    result = {
        "snn": {
            "synops_per_sample": snn_stats["weighted_synops_per_sample"],
            "spikes_per_sample": snn_stats["total_spikes_per_sample"],
            "energy_pJ": snn_energy,
            "spike_counts": snn_stats["spike_counts"],
        },
        "ann": {
            "macs_per_sample": ann_stats["total_macs_per_sample"],
            "energy_pJ": ann_energy,
            "macs_per_layer": ann_stats["macs_per_layer"],
        },
        "energy_ratio": ratio,
        "snn_savings_pct": (1 - ratio) * 100 if ratio < 1 else 0,
        "assumptions": {
            "synop_energy_pj": SYNOP_ENERGY_PJ,
            "mac_energy_pj": MAC_ENERGY_PJ,
            "note": "SynOp energy based on Loihi (Intel), MAC energy based on 45nm CMOS (Horowitz 2014)",
        },
    }

    return result


def save_energy_report(result: dict, save_dir: Path = None):
    """Save energy comparison report."""
    if save_dir is None:
        save_dir = RESULTS_DIR / "energy"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "energy_comparison.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Energy Comparison Report")
    print(f"{'='*60}")
    print(f"  SNN SynOps/sample:  {result['snn']['synops_per_sample']:,.0f}")
    print(f"  SNN Energy/sample:  {result['snn']['energy_pJ']:,.1f} pJ")
    print(f"  ANN MACs/sample:    {result['ann']['macs_per_sample']:,.0f}")
    print(f"  ANN Energy/sample:  {result['ann']['energy_pJ']:,.1f} pJ")
    print(f"  Energy ratio (SNN/ANN): {result['energy_ratio']:.4f}")
    if result['energy_ratio'] < 1:
        print(f"  SNN is {result['snn_savings_pct']:.1f}% more energy efficient")
    else:
        print(f"  ANN is more energy efficient in this configuration")
    print(f"{'='*60}")

    print(f"  Report saved to {save_dir / 'energy_comparison.json'}")
