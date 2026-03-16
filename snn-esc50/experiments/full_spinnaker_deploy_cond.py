"""
Full SpiNNaker Deployment: IF_cond_exp (conductance-based) + MaxPool model.

Deploys FC1+FC2 on SpiNNaker using conductance-based synapses (IF_cond_exp)
instead of current-based (IF_curr_exp).

Key insight: conductance-based models prevent excitatory-inhibitory
cancellation because inhibition is SHUNTING (g_syn * (V - E_rev)),
not subtractive. The membrane cannot go below E_rev_I, preventing
the catastrophic cancellation that killed the original FC1 deployment.

Strategy:
    1. Use the MaxPool-retrained model (Option A) for binary FC1 inputs
    2. Extract features using MaxPool model conv layers (on CPU)
    3. Deploy FC1+FC2 on SpiNNaker with IF_cond_exp neurons
    4. Separate exc/inh weights as positive conductances

Prerequisites:
    1. Train MaxPool SNN model -- python experiments/spinnaker_option_a.py
       (saved at results/snn/maxpool/best_fold4.pt)
    2. Run with .venv-spinnaker environment for SpiNNaker steps
    3. Access to SpiNNaker via spalloc

Usage:
    # Step 1: Extract features (use regular .venv, not .venv-spinnaker)
    source .venv/bin/activate
    python experiments/full_spinnaker_deploy_cond.py --fold 4 --extract-only

    # Step 2: Run on SpiNNaker (use .venv-spinnaker)
    source .venv-spinnaker/bin/activate
    python experiments/full_spinnaker_deploy_cond.py --fold 4 --num-samples 20

    # Calibration sweep (recommended first):
    python experiments/full_spinnaker_deploy_cond.py --fold 4 --num-samples 1 --scale-sweep

Calibration notes for IF_cond_exp:
    - Weights are CONDUCTANCES (uS), must be positive
    - Excitatory/inhibitory separated by receptor_type
    - Driving force asymmetry: E_rev_E - V_rest vs V_rest - E_rev_I
    - Weight scale needs empirical tuning per hardware run
    - Start with --scale-sweep to find the right range
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, NUM_STEPS


# ============================================================
# MaxPool model definition (must match spinnaker_option_a.py)
# ============================================================

def _build_maxpool_model(threshold=3.0):
    """Build the SpikingCNN_MaxPool model.

    Imports torch/snn only when called (not needed in .venv-spinnaker).
    """
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate
    from src.config import NUM_CLASSES, BETA, N_MELS

    class SpikingCNN_MaxPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_steps = NUM_STEPS
            spike_grad = surrogate.fast_sigmoid(slope=25)

            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2)
            self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad,
                                   threshold=threshold)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2)
            self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad,
                                   threshold=threshold)

            # KEY: MaxPool instead of AvgPool -> binary FC1 inputs
            self.max_pool = nn.MaxPool2d(kernel_size=(4, 6))

            self.fc1 = nn.Linear(64 * 4 * 9, 256)
            self.lif3 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

            self.fc2 = nn.Linear(256, NUM_CLASSES)
            self.lif4 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            mem4 = self.lif4.init_leaky()

            spk_out_rec = []
            mem_out_rec = []

            for step in range(self.num_steps):
                x_t = x[step]
                cur1 = self.pool1(self.bn1(self.conv1(x_t)))
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.pool2(self.bn2(self.conv2(spk1)))
                spk2, mem2 = self.lif2(cur2, mem2)
                pooled = self.max_pool(spk2)
                flat = pooled.view(pooled.size(0), -1)
                cur3 = self.fc1(flat)
                spk3, mem3 = self.lif3(cur3, mem3)
                cur4 = self.fc2(spk3)
                spk4, mem4 = self.lif4(cur4, mem4)
                spk_out_rec.append(spk4)
                mem_out_rec.append(mem4)

            return torch.stack(spk_out_rec), torch.stack(mem_out_rec)

    return SpikingCNN_MaxPool()


# ============================================================
# Weight loading
# ============================================================

def load_model_weights(fold=4, threshold=3.0):
    """Load FC1 and FC2 weights from the MaxPool-retrained model.

    IMPORTANT: Uses results/snn/maxpool/best_fold{fold}.pt, NOT the
    original AvgPool model. The MaxPool model produces binary FC1 inputs.
    """
    import torch

    model = _build_maxpool_model(threshold=threshold)
    model_path = RESULTS_DIR / "snn" / "maxpool" / f"best_fold{fold}.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"MaxPool model not found: {model_path}\n"
            f"Train it first: python experiments/spinnaker_option_a.py "
            f"--fold {fold} --threshold {threshold}"
        )

    model.load_state_dict(torch.load(model_path, weights_only=True,
                                     map_location="cpu"))
    model.eval()

    fc1_w = model.fc1.weight.detach().numpy()  # (256, 2304)
    fc1_b = model.fc1.bias.detach().numpy() if model.fc1.bias is not None else None
    fc2_w = model.fc2.weight.detach().numpy()  # (50, 256)
    fc2_b = model.fc2.bias.detach().numpy() if model.fc2.bias is not None else None

    # Print weight statistics for debugging
    print(f"  FC1 weight stats: mean={fc1_w.mean():.5f} std={fc1_w.std():.5f} "
          f"range=[{fc1_w.min():.5f}, {fc1_w.max():.5f}]")
    print(f"  FC2 weight stats: mean={fc2_w.mean():.5f} std={fc2_w.std():.5f} "
          f"range=[{fc2_w.min():.5f}, {fc2_w.max():.5f}]")

    exc_count_fc1 = (fc1_w > 0).sum()
    inh_count_fc1 = (fc1_w < 0).sum()
    print(f"  FC1 exc/inh: {exc_count_fc1}/{inh_count_fc1} "
          f"({100*exc_count_fc1/fc1_w.size:.1f}%/{100*inh_count_fc1/fc1_w.size:.1f}%)")

    return {
        "fc1_weight": fc1_w, "fc1_bias": fc1_b,
        "fc2_weight": fc2_w, "fc2_bias": fc2_b,
    }


# ============================================================
# Connection list preparation
# ============================================================

def prepare_connection_lists(weight_matrix, weight_scale=1.0,
                             prune_threshold=0.01):
    """Convert weight matrix to sPyNNaker FromListConnector format.

    For IF_cond_exp, weights must be POSITIVE (conductances in uS).
    Excitatory and inhibitory connections are separated by receptor_type.

    Args:
        weight_matrix: shape (n_post, n_pre), e.g. FC1 is (256, 2304)
        weight_scale: multiplier applied to |weight| for conductance
        prune_threshold: minimum |weight| to include (before scaling)

    Returns:
        exc_conns: list of (pre, post, weight, delay) for excitatory
        inh_conns: list of (pre, post, weight, delay) for inhibitory
    """
    n_post, n_pre = weight_matrix.shape

    # Vectorized extraction -- much faster than nested loops
    post_idx, pre_idx = np.where(np.abs(weight_matrix) > prune_threshold)
    weights = weight_matrix[post_idx, pre_idx]

    exc_mask = weights > 0
    inh_mask = weights < 0

    # Excitatory connections: positive weights as conductances
    exc_pre = pre_idx[exc_mask]
    exc_post = post_idx[exc_mask]
    exc_w = weights[exc_mask] * weight_scale
    exc_conns = [
        (int(pre), int(post), float(w), 1.0)
        for pre, post, w in zip(exc_pre, exc_post, exc_w)
    ]

    # Inhibitory connections: absolute value of negative weights as conductances
    inh_pre = pre_idx[inh_mask]
    inh_post = post_idx[inh_mask]
    inh_w = np.abs(weights[inh_mask]) * weight_scale
    inh_conns = [
        (int(pre), int(post), float(w), 1.0)
        for pre, post, w in zip(inh_pre, inh_post, inh_w)
    ]

    return exc_conns, inh_conns


# ============================================================
# Feature extraction (runs on CPU with snnTorch)
# ============================================================

def extract_input_features(fold=4, num_samples=20, threshold=3.0):
    """Extract FC1 input features from the MaxPool model conv layers.

    Uses the MaxPool-retrained model so that FC1 inputs are BINARY (0 or 1),
    not fractional. This is critical for SpiNNaker deployment.

    Returns:
        features: (N, 25, 2304) binary float array -- FC1 inputs per timestep
        labels: (N,) int array -- true class labels
        snntorch_preds: (N,) int array -- full model predictions for comparison
    """
    import torch
    from src.config import get_device
    from src.dataset import get_fold_dataloaders
    from src.encoding import get_encoder

    device = get_device()
    _, test_loader = get_fold_dataloaders(fold, batch_size=1)
    encoder = get_encoder("direct")

    model = _build_maxpool_model(threshold=threshold).to(device)
    model_path = RESULTS_DIR / "snn" / "maxpool" / f"best_fold{fold}.pt"
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_features = []   # (N, 25, 2304)
    all_labels = []
    all_snntorch_preds = []

    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if i >= num_samples:
                break

            data = data.to(device)
            spk_input = encoder(data).to(device)

            # Run conv layers through the MaxPool model
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()
            mem4 = model.lif4.init_leaky()

            features_per_step = []
            mem_out_acc = torch.zeros(1, 50, device=device)

            for step in range(NUM_STEPS):
                x_t = spk_input[step]
                cur1 = model.pool1(model.bn1(model.conv1(x_t)))
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.pool2(model.bn2(model.conv2(spk1)))
                spk2, mem2 = model.lif2(cur2, mem2)

                # KEY: MaxPool, not AvgPool! Output is binary (0 or 1).
                pooled = model.max_pool(spk2)
                flat = pooled.view(pooled.size(0), -1)  # (1, 2304)
                features_per_step.append(flat.cpu().numpy()[0])

                # Full model forward for snnTorch reference prediction
                cur3 = model.fc1(flat)
                spk3, mem3 = model.lif3(cur3, mem3)
                cur4 = model.fc2(spk3)
                spk4, mem4 = model.lif4(cur4, mem4)
                mem_out_acc += mem4

            sample_features = np.array(features_per_step)  # (25, 2304)
            all_features.append(sample_features)
            all_labels.append(targets[0].item())
            all_snntorch_preds.append(mem_out_acc.argmax(dim=1).item())

            # Verify binary-ness
            binary_frac = np.mean(
                (sample_features == 0) | (sample_features == 1)
            )
            active_per_step = (sample_features > 0.5).sum(axis=1).mean()
            if i < 5 or binary_frac < 0.99:
                print(f"    Sample {i}: binary_frac={binary_frac:.4f} "
                      f"active/step={active_per_step:.0f}/2304 "
                      f"label={targets[0].item()}")

    features = np.array(all_features)
    labels = np.array(all_labels)
    preds = np.array(all_snntorch_preds)

    # Aggregate binary check
    total_binary = np.mean((features == 0) | (features == 1))
    print(f"  Overall binary fraction: {total_binary:.6f}")
    if total_binary < 0.999:
        print(f"  WARNING: FC1 inputs are NOT fully binary ({total_binary:.4f}). "
              f"MaxPool model may have issues.")
    else:
        print(f"  GOOD: FC1 inputs are binary (MaxPool working correctly)")

    return features, labels, preds


# ============================================================
# SpiNNaker deployment
# ============================================================

def run_spinnaker_full(features, labels, weights, weight_scale=0.005,
                       prune_threshold=0.01, scale_sweep=False,
                       sweep_scales=None):
    """Deploy FC1+FC2 on SpiNNaker using IF_cond_exp.

    Args:
        features: (N, 25, 2304) binary FC1 input features
        labels: (N,) true labels
        weights: dict with fc1_weight, fc2_weight, etc.
        weight_scale: conductance multiplier
        prune_threshold: minimum |weight| to include
        scale_sweep: if True, sweep scales on first sample only
        sweep_scales: list of scales to try (for sweep mode)

    Returns:
        results: list of per-sample result dicts
        accuracy: float
    """
    try:
        import pyNN.spiNNaker as sim
    except ImportError:
        print("ERROR: sPyNNaker not available. Use .venv-spinnaker.")
        return None, 0.0

    # Prepare connection lists (once, at the base scale)
    # We will rescale per-sweep-iteration if needed
    fc1_exc_base, fc1_inh_base = prepare_connection_lists(
        weights["fc1_weight"], weight_scale=1.0,
        prune_threshold=prune_threshold,
    )
    fc2_exc_base, fc2_inh_base = prepare_connection_lists(
        weights["fc2_weight"], weight_scale=1.0,
        prune_threshold=prune_threshold,
    )

    print(f"  FC1: {len(fc1_exc_base)} exc + {len(fc1_inh_base)} inh connections "
          f"(total {len(fc1_exc_base)+len(fc1_inh_base)})")
    print(f"  FC2: {len(fc2_exc_base)} exc + {len(fc2_inh_base)} inh connections "
          f"(total {len(fc2_exc_base)+len(fc2_inh_base)})")
    total_conns = (len(fc1_exc_base) + len(fc1_inh_base) +
                   len(fc2_exc_base) + len(fc2_inh_base))
    print(f"  Total connections: {total_conns}")

    if total_conns > 500000:
        print(f"  WARNING: {total_conns} connections is very large. "
              f"May hit UDP buffer limits. Consider --prune-threshold 0.05")

    # IF_cond_exp parameters
    # Using biological-style parameters where IF_cond_exp is designed to work.
    # The shunting inhibition prevents the cancellation problem.
    #
    # Key dynamics:
    #   dV/dt = (g_L*(E_L - V) + g_E*(E_rev_E - V) + g_I*(E_rev_I - V)) / cm
    #   At rest (V = v_rest = -65mV):
    #     Exc driving force = E_rev_E - V = 0 - (-65) = +65 mV
    #     Inh driving force = E_rev_I - V = -80 - (-65) = -15 mV
    #   So excitation is ~4.3x more effective per unit conductance.
    #
    #   This asymmetry actually HELPS us: the original cancellation problem
    #   was that exc and inh currents perfectly cancelled. With conductance-based
    #   synapses and this asymmetry, the balance is broken in favour of excitation,
    #   making neurons more likely to fire.
    #
    #   The weight_scale needs to compensate for:
    #   - snnTorch weights are current amplitudes, not conductances
    #   - driving force changes with membrane potential (nonlinear)
    #
    cond_params = {
        "cm": 1.0,           # nF
        "tau_m": 20.0,       # ms (matches snnTorch beta=0.95: 1-dt/tau=0.95)
        "tau_refrac": 0.1,   # ms (minimal, snnTorch has none)
        "v_reset": -65.0,    # mV
        "v_rest": -65.0,     # mV
        "v_thresh": -50.0,   # mV (15mV above rest)
        "tau_syn_E": 5.0,    # ms
        "tau_syn_I": 5.0,    # ms
        "e_rev_E": 0.0,      # mV (excitatory reversal)
        "e_rev_I": -80.0,    # mV (inhibitory reversal)
    }

    def _scale_connections(conn_list, scale):
        """Apply a weight scale to a connection list."""
        return [
            (pre, post, w * scale, delay)
            for pre, post, w, delay in conn_list
        ]

    def _run_one_sample(sample_features, scale, sample_idx=0):
        """Run a single sample through SpiNNaker FC1+FC2."""

        # Convert binary features to spike times
        # features[t, j] == 1.0 means neuron j fires at time t
        # SpikeSourceArray times are in ms; we use t (0-indexed) so
        # spike times range from 0.0 to 24.0, all within sim.run(25).
        spike_times = {}
        for j in range(2304):
            # Vectorized: find timesteps where this neuron is active
            active_steps = np.where(sample_features[:, j] > 0.5)[0]
            spike_times[j] = active_steps.astype(float).tolist()

        total_input_spikes = sum(len(v) for v in spike_times.values())
        active_inputs = sum(1 for v in spike_times.values() if len(v) > 0)
        spikes_per_step = (sample_features > 0.5).sum(axis=1)

        print(f"    Input: {active_inputs}/2304 neurons active, "
              f"{total_input_spikes} total spikes, "
              f"mean={spikes_per_step.mean():.0f}/step, "
              f"max={spikes_per_step.max()}/step")

        # Scale connections
        fc1_exc = _scale_connections(fc1_exc_base, scale)
        fc1_inh = _scale_connections(fc1_inh_base, scale)
        fc2_exc = _scale_connections(fc2_exc_base, scale)
        fc2_inh = _scale_connections(fc2_inh_base, scale)

        result = {
            "sample": sample_idx,
            "weight_scale": scale,
            "total_input_spikes": total_input_spikes,
            "active_inputs": active_inputs,
        }

        try:
            sim.setup(timestep=1.0)

            try:
                # Input population (2304 neurons = flattened MaxPool features)
                input_pop = sim.Population(
                    2304,
                    sim.SpikeSourceArray,
                    {"spike_times": [spike_times[j] for j in range(2304)]},
                    label="input"
                )
                input_pop.record("spikes")

                # Hidden population (256 neurons, IF_cond_exp)
                hidden_pop = sim.Population(
                    256,
                    sim.IF_cond_exp(**cond_params),
                    label="hidden"
                )
                hidden_pop.initialize(v=cond_params["v_rest"])
                hidden_pop.record(["spikes", "v"])

                # Output population (50 neurons, IF_cond_exp)
                output_pop = sim.Population(
                    50,
                    sim.IF_cond_exp(**cond_params),
                    label="output"
                )
                output_pop.initialize(v=cond_params["v_rest"])
                output_pop.record(["spikes", "v"])

                # FC1 projections: input -> hidden
                if fc1_exc:
                    sim.Projection(
                        input_pop, hidden_pop,
                        sim.FromListConnector(fc1_exc),
                        receptor_type="excitatory"
                    )
                if fc1_inh:
                    sim.Projection(
                        input_pop, hidden_pop,
                        sim.FromListConnector(fc1_inh),
                        receptor_type="inhibitory"
                    )

                # FC2 projections: hidden -> output
                if fc2_exc:
                    sim.Projection(
                        hidden_pop, output_pop,
                        sim.FromListConnector(fc2_exc),
                        receptor_type="excitatory"
                    )
                if fc2_inh:
                    sim.Projection(
                        hidden_pop, output_pop,
                        sim.FromListConnector(fc2_inh),
                        receptor_type="inhibitory"
                    )

                sim.run(NUM_STEPS)

                # --- Extract results ---

                # Verify input spikes delivered
                in_data = input_pop.get_data("spikes")
                in_actual = sum(
                    len(st) for st in in_data.segments[0].spiketrains
                )

                # Hidden layer spikes
                hidden_data = hidden_pop.get_data(["spikes", "v"])
                hidden_spiketrains = hidden_data.segments[0].spiketrains
                hidden_spike_counts = np.array([
                    len(st) for st in hidden_spiketrains
                ])
                total_hidden_spikes = int(hidden_spike_counts.sum())
                hidden_neurons_fired = int((hidden_spike_counts > 0).sum())

                # Hidden voltage (for diagnostics)
                hidden_v_signals = hidden_data.segments[0].filter(name="v")
                max_hidden_v = -999.0
                if hidden_v_signals:
                    h_v = hidden_v_signals[0].magnitude
                    max_hidden_v = float(h_v.max())

                # Output layer spikes
                out_data = output_pop.get_data(["spikes", "v"])
                out_spiketrains = out_data.segments[0].spiketrains
                output_spike_counts = [0] * 50
                for neuron_id, st in enumerate(out_spiketrains):
                    output_spike_counts[neuron_id] = len(st)
                total_output_spikes = sum(output_spike_counts)

                # Output membrane voltages
                out_v_signals = out_data.segments[0].filter(name="v")
                output_v_final = [0.0] * 50
                output_v_sum = [0.0] * 50
                if out_v_signals:
                    v_arr = out_v_signals[0].magnitude  # (T, 50)
                    for n in range(min(50, v_arr.shape[1])):
                        output_v_final[n] = float(v_arr[-1, n])
                        output_v_sum[n] = float(v_arr[:, n].sum())

                # Classification: prefer spike count, fall back to voltage
                if total_output_spikes > 0:
                    predicted = int(np.argmax(output_spike_counts))
                    basis = "spike_count"
                else:
                    # Fall back to membrane voltage sum (like snnTorch
                    # mem_out.sum(dim=0).argmax())
                    predicted = int(np.argmax(output_v_sum))
                    basis = "voltage_sum"

                neurons_fired_output = [
                    i for i, c in enumerate(output_spike_counts) if c > 0
                ]

                print(f"    Input delivered: expected={total_input_spikes} "
                      f"actual={in_actual}")
                print(f"    Hidden: {total_hidden_spikes} spikes, "
                      f"{hidden_neurons_fired}/256 neurons fired, "
                      f"max_v={max_hidden_v:.4f}")
                print(f"    Output: {total_output_spikes} spikes, "
                      f"neurons fired: {neurons_fired_output[:10]}")
                print(f"    Predicted: {predicted} (via {basis})")

                result.update({
                    "predicted": predicted,
                    "classification_basis": basis,
                    "hidden_spikes_total": total_hidden_spikes,
                    "hidden_neurons_fired": hidden_neurons_fired,
                    "max_hidden_v": max_hidden_v,
                    "output_spikes_total": total_output_spikes,
                    "output_spike_counts": output_spike_counts,
                    "output_v_final": output_v_final,
                    "output_v_sum": output_v_sum,
                    "neurons_fired_output": neurons_fired_output,
                    "input_delivered": in_actual,
                })

            finally:
                sim.end()

        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {str(e)[:200]}")
            result.update({
                "predicted": -1,
                "error": str(e),
                "hidden_spikes_total": 0,
            })

        return result

    # ---- Scale sweep mode ----
    if scale_sweep:
        if sweep_scales is None:
            sweep_scales = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
                            0.2, 0.5, 1.0]

        print(f"\n  === SCALE SWEEP on sample 0 ===")
        sample_features = features[0]
        true_label = int(labels[0])
        print(f"  True label: {true_label}")

        sweep_results = []
        for scale in sweep_scales:
            print(f"\n  --- Scale {scale} ---")
            r = _run_one_sample(sample_features, scale, sample_idx=0)
            r["true"] = true_label
            r["correct"] = int(r.get("predicted", -1) == true_label)
            sweep_results.append(r)

            hidden_spk = r.get("hidden_spikes_total", 0)
            out_spk = r.get("output_spikes_total", 0)
            print(f"    => hidden_spk={hidden_spk} out_spk={out_spk} "
                  f"pred={r.get('predicted', -1)} "
                  f"{'CORRECT' if r.get('correct') else 'WRONG'}")

        print(f"\n  === Scale Sweep Summary ===")
        print(f"  {'Scale':>10}  {'HidSpk':>8}  {'HidFired':>8}  "
              f"{'OutSpk':>8}  {'Pred':>5}  {'Correct':>7}")
        print("  " + "-" * 60)
        for r in sweep_results:
            print(f"  {r['weight_scale']:>10.4f}  "
                  f"{r.get('hidden_spikes_total', 0):>8}  "
                  f"{r.get('hidden_neurons_fired', 0):>8}  "
                  f"{r.get('output_spikes_total', 0):>8}  "
                  f"{r.get('predicted', -1):>5}  "
                  f"{'YES' if r.get('correct') else 'NO':>7}")

        return sweep_results, 0.0

    # ---- Full inference mode ----
    n_samples = features.shape[0]
    results = []

    for idx in range(n_samples):
        sample_features = features[idx]  # (25, 2304)
        true_label = int(labels[idx])

        print(f"\n  [{idx+1}/{n_samples}] True label: {true_label}")
        r = _run_one_sample(sample_features, weight_scale, sample_idx=idx)
        r["true"] = true_label
        r["correct"] = int(r.get("predicted", -1) == true_label)
        results.append(r)

        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"    => {status}")

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    return results, accuracy


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full SpiNNaker deploy with IF_cond_exp + MaxPool model"
    )
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="LIF threshold used when training MaxPool model "
                             "(must match spinnaker_option_a.py setting)")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--weight-scale", type=float, default=0.005,
                        help="Conductance scale for IF_cond_exp weights. "
                             "Needs calibration -- use --scale-sweep first.")
    parser.add_argument("--prune-threshold", type=float, default=0.01,
                        help="Minimum |weight| to keep (default: 0.01)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract features (use with .venv, not "
                             ".venv-spinnaker)")
    parser.add_argument("--scale-sweep", action="store_true",
                        help="Run scale sweep on sample 0 to calibrate "
                             "weight_scale")
    parser.add_argument("--sweep-scales", type=str, default=None,
                        help="Comma-separated list of scales for sweep "
                             "(e.g., '0.001,0.01,0.1')")
    parser.add_argument("--use-cached-features", action="store_true",
                        help="Load previously extracted features from disk "
                             "instead of re-extracting")
    args = parser.parse_args()

    save_dir = RESULTS_DIR / "spinnaker_results" / "full_deploy_cond"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Full SpiNNaker Deployment (IF_cond_exp + MaxPool)")
    print("=" * 60)
    print(f"  Fold:             {args.fold}")
    print(f"  LIF threshold:    {args.threshold}")
    print(f"  Num samples:      {args.num_samples}")
    print(f"  Weight scale:     {args.weight_scale}")
    print(f"  Prune threshold:  {args.prune_threshold}")
    print(f"  Scale sweep:      {args.scale_sweep}")
    print()

    # Step 1: Load weights from MaxPool model
    print("[1/3] Loading MaxPool model weights...")
    fc1_w_path = save_dir / f"fc1_weight_fold{args.fold}.npy"
    fc2_w_path = save_dir / f"fc2_weight_fold{args.fold}.npy"

    if fc1_w_path.exists() and fc2_w_path.exists():
        # Load pre-extracted numpy weights (no torch needed)
        print("  Loading cached numpy weights...")
        weights = {
            "fc1_weight": np.load(fc1_w_path),
            "fc1_bias": None,
            "fc2_weight": np.load(fc2_w_path),
            "fc2_bias": None,
        }
    else:
        # Extract weights (requires torch)
        weights = load_model_weights(args.fold, threshold=args.threshold)
        np.save(fc1_w_path, weights["fc1_weight"])
        np.save(fc2_w_path, weights["fc2_weight"])
        print("  Saved weights as numpy for future SpiNNaker runs")

    print(f"  FC1: {weights['fc1_weight'].shape}")
    print(f"  FC2: {weights['fc2_weight'].shape}")

    # Step 2: Extract or load features
    features_path = save_dir / f"fc1_input_features_fold{args.fold}.npy"
    labels_path = save_dir / f"labels_fold{args.fold}.npy"
    preds_path = save_dir / f"snntorch_preds_fold{args.fold}.npy"

    if args.use_cached_features and features_path.exists():
        print(f"\n[2/3] Loading cached features from {save_dir}...")
        features = np.load(features_path)
        labels = np.load(labels_path)
        snntorch_preds = np.load(preds_path)
        print(f"  Features shape: {features.shape}")
    else:
        print(f"\n[2/3] Extracting conv features via snnTorch (MaxPool model)...")
        features, labels, snntorch_preds = extract_input_features(
            args.fold, args.num_samples, threshold=args.threshold
        )
        print(f"  Features shape: {features.shape}")
        print(f"  Feature stats: min={features.min():.4f} "
              f"max={features.max():.4f} "
              f"mean={features.mean():.4f}")

        # Save features for reuse
        np.save(features_path, features)
        np.save(labels_path, labels)
        np.save(preds_path, snntorch_preds)
        print(f"  Saved features to {save_dir}")

    snntorch_acc = (snntorch_preds == labels).mean()
    print(f"  snnTorch MaxPool model accuracy: {snntorch_acc:.4f} "
          f"({int(snntorch_acc * len(labels))}/{len(labels)})")

    if args.extract_only:
        print("\n--extract-only: stopping before SpiNNaker.")
        print("Next: activate .venv-spinnaker and run without --extract-only")
        return

    # Step 3: SpiNNaker deployment
    sweep_scales = None
    if args.sweep_scales:
        sweep_scales = [float(x) for x in args.sweep_scales.split(",")]

    if args.scale_sweep:
        print(f"\n[3/3] Running scale sweep on SpiNNaker...")
        results, _ = run_spinnaker_full(
            features, labels, weights,
            weight_scale=args.weight_scale,
            prune_threshold=args.prune_threshold,
            scale_sweep=True,
            sweep_scales=sweep_scales,
        )

        if results:
            sweep_path = save_dir / f"scale_sweep_fold{args.fold}.json"
            with open(sweep_path, "w") as f:
                json.dump({
                    "fold": args.fold,
                    "threshold": args.threshold,
                    "neuron_model": "IF_cond_exp",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "results": results,
                }, f, indent=2, default=str)
            print(f"\n  Sweep saved: {sweep_path}")
            print("\n  Use the best scale from the sweep with:")
            print(f"  python experiments/full_spinnaker_deploy_cond.py "
                  f"--fold {args.fold} --weight-scale <BEST_SCALE> "
                  f"--num-samples {args.num_samples} --use-cached-features")
    else:
        print(f"\n[3/3] Running full inference on SpiNNaker "
              f"(weight_scale={args.weight_scale})...")
        results, accuracy = run_spinnaker_full(
            features, labels, weights,
            weight_scale=args.weight_scale,
            prune_threshold=args.prune_threshold,
        )

        if results:
            # Add true labels to results
            for r in results:
                r["snntorch_pred"] = int(
                    snntorch_preds[r["sample"]]
                ) if r["sample"] < len(snntorch_preds) else -1

            summary = {
                "fold": args.fold,
                "threshold": args.threshold,
                "encoding": "direct",
                "num_samples": len(results),
                "weight_scale": args.weight_scale,
                "prune_threshold": args.prune_threshold,
                "neuron_model": "IF_cond_exp",
                "model_source": f"results/snn/maxpool/best_fold{args.fold}.pt",
                "spinnaker_accuracy": accuracy,
                "snntorch_accuracy": float(snntorch_acc),
                "hardware_gap": float(snntorch_acc - accuracy),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            }

            result_path = save_dir / f"full_deploy_fold{args.fold}.json"
            with open(result_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"\n{'='*60}")
            print(f"Full Deploy Results (Fold {args.fold})")
            print(f"{'='*60}")
            print(f"  Model:                  MaxPool (threshold={args.threshold})")
            print(f"  Neuron model:           IF_cond_exp")
            print(f"  Weight scale:           {args.weight_scale}")
            print(f"  SpiNNaker accuracy:     {accuracy:.4f} "
                  f"({sum(r['correct'] for r in results)}/{len(results)})")
            print(f"  snnTorch accuracy:      {snntorch_acc:.4f}")
            print(f"  Hardware gap:           "
                  f"{snntorch_acc - accuracy:.4f}")
            print(f"\n  Per-sample:")
            print(f"  {'Idx':>4}  {'True':>5}  {'Pred':>5}  {'HidSpk':>7}  "
                  f"{'OutSpk':>7}  {'Result':>8}")
            print("  " + "-" * 50)
            for r in results:
                status = "OK" if r.get("correct") else "WRONG"
                if r.get("error"):
                    status = "ERROR"
                print(f"  {r['sample']:>4}  {r.get('true', '?'):>5}  "
                      f"{r.get('predicted', -1):>5}  "
                      f"{r.get('hidden_spikes_total', 0):>7}  "
                      f"{r.get('output_spikes_total', 0):>7}  "
                      f"{status:>8}")
            print(f"\n  Saved: {result_path}")


if __name__ == "__main__":
    main()
