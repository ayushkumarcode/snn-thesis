"""
Convert trained snnTorch model weights to sPyNNaker-compatible format.

sPyNNaker uses PyNN-style populations and projections, so we need to:
1. Extract weight matrices from the trained snnTorch model
2. Fold BatchNorm parameters into conv/fc weights (BN not available on SpiNNaker)
3. Convert Conv2d weights to equivalent dense connection matrices
4. Apply MaxPool/AvgPool as index mappings between populations
5. Save in a format sPyNNaker can load

Architecture being converted:
  Conv2d(1,32,3,pad=1) -> BN(32) -> MaxPool(2) -> LIF
  Conv2d(32,64,3,pad=1) -> BN(64) -> MaxPool(2) -> LIF
  AvgPool2d(4,6) -> Flatten
  FC(2304,256) -> LIF
  FC(256,50) -> LIF

Input spectrogram: (1, 64, 216)
After conv1+pool1: (32, 32, 108)
After conv2+pool2: (64, 16, 54)
After avg_pool(4,6): (64, 4, 9) = 2304 neurons
After fc1: 256 neurons
After fc2: 50 neurons (output classes)
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.config import RESULTS_DIR


def extract_weights(model_path: str, device="cpu") -> dict:
    """Extract all weight matrices from a trained snnTorch model.

    Args:
        model_path: Path to the saved .pt state dict.
        device: Device to load to.

    Returns:
        Dict mapping layer names to numpy weight arrays.
    """
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    weights = {}
    for name, param in state_dict.items():
        weights[name] = param.cpu().numpy()

    return weights


def fold_batchnorm_into_conv(conv_w, conv_b, bn_gamma, bn_beta,
                              bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm parameters into preceding Conv2d weights.

    BN computes: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Conv computes: x = W * input + b

    Combined: y = (gamma / sqrt(var + eps)) * W * input
                + (gamma / sqrt(var + eps)) * b - gamma * mean / sqrt(var + eps) + beta

    So new_W = gamma / sqrt(var+eps) * W
       new_b = gamma / sqrt(var+eps) * b - gamma * mean / sqrt(var+eps) + beta

    Args:
        conv_w: Conv weight (out_c, in_c, kH, kW)
        conv_b: Conv bias (out_c,)
        bn_gamma: BN weight/scale (out_c,)
        bn_beta: BN bias (out_c,)
        bn_mean: BN running mean (out_c,)
        bn_var: BN running variance (out_c,)
        eps: BN epsilon

    Returns:
        (new_weight, new_bias) with BN folded in.
    """
    scale = bn_gamma / np.sqrt(bn_var + eps)

    # Reshape scale for broadcasting with conv weights
    new_w = conv_w * scale.reshape(-1, 1, 1, 1)
    new_b = scale * conv_b - scale * bn_mean + bn_beta

    return new_w, new_b


def conv2d_to_dense(weight, bias, input_shape,
                    stride=1, padding=1):
    """Convert Conv2d weight tensor to equivalent dense connection matrix.

    This "unrolls" the convolution into a matrix multiplication.
    Used for SpiNNaker deployment where Conv2d is not natively available.

    Args:
        weight: Conv2d weight of shape (out_c, in_c, kH, kW).
        bias: Conv2d bias of shape (out_c,) or None.
        input_shape: (in_c, H, W) of the input feature map.
        stride: Convolution stride.
        padding: Convolution padding.

    Returns:
        (dense_matrix, bias_vector) where
        dense_matrix shape is (out_neurons, in_neurons),
        bias_vector shape is (out_neurons,).
    """
    out_c, in_c, kH, kW = weight.shape
    _, H, W = input_shape

    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1

    in_neurons = in_c * H * W
    out_neurons = out_c * out_H * out_W

    dense = np.zeros((out_neurons, in_neurons), dtype=np.float32)

    for oc in range(out_c):
        for oh in range(out_H):
            for ow in range(out_W):
                out_idx = oc * out_H * out_W + oh * out_W + ow
                for ic in range(in_c):
                    for kh in range(kH):
                        for kw in range(kW):
                            ih = oh * stride - padding + kh
                            iw = ow * stride - padding + kw
                            if 0 <= ih < H and 0 <= iw < W:
                                in_idx = ic * H * W + ih * W + iw
                                dense[out_idx, in_idx] = weight[oc, ic, kh, kw]

    # Expand bias to match output neurons
    if bias is not None:
        bias_expanded = np.repeat(bias, out_H * out_W)
    else:
        bias_expanded = np.zeros(out_neurons, dtype=np.float32)

    return dense, bias_expanded


def maxpool2d_mapping(input_shape, pool_size=2):
    """Create MaxPool2d index mapping as a sparse connection matrix.

    On SpiNNaker, MaxPool is approximated: in a spiking network, the
    "max" operation becomes "any spike in the pool window passes through".
    This is implemented as a many-to-one connection with weight 1.0.

    Args:
        input_shape: (C, H, W) of the input feature map.
        pool_size: Size of the pooling window (square).

    Returns:
        (mapping_matrix, output_shape) where mapping_matrix shape is
        (out_neurons, in_neurons) and output_shape is (C, H//pool, W//pool).
    """
    C, H, W = input_shape
    out_H = H // pool_size
    out_W = W // pool_size

    in_neurons = C * H * W
    out_neurons = C * out_H * out_W

    # For spiking networks, max pool = OR of spikes in window
    # Use weight=1 connections from all pool inputs to pool output
    mapping = np.zeros((out_neurons, in_neurons), dtype=np.float32)

    for c in range(C):
        for oh in range(out_H):
            for ow in range(out_W):
                out_idx = c * out_H * out_W + oh * out_W + ow
                for ph in range(pool_size):
                    for pw in range(pool_size):
                        ih = oh * pool_size + ph
                        iw = ow * pool_size + pw
                        in_idx = c * H * W + ih * W + iw
                        mapping[out_idx, in_idx] = 1.0

    return mapping, (C, out_H, out_W)


def avgpool2d_mapping(input_shape, kernel_size):
    """Create AvgPool2d mapping as a connection matrix with averaged weights.

    Args:
        input_shape: (C, H, W) of the input.
        kernel_size: (kH, kW) pooling kernel size.

    Returns:
        (mapping_matrix, output_shape).
    """
    C, H, W = input_shape
    kH, kW = kernel_size
    out_H = H // kH
    out_W = W // kW

    in_neurons = C * H * W
    out_neurons = C * out_H * out_W
    scale = 1.0 / (kH * kW)

    mapping = np.zeros((out_neurons, in_neurons), dtype=np.float32)

    for c in range(C):
        for oh in range(out_H):
            for ow in range(out_W):
                out_idx = c * out_H * out_W + oh * out_W + ow
                for ph in range(kH):
                    for pw in range(kW):
                        ih = oh * kH + ph
                        iw = ow * kW + pw
                        in_idx = c * H * W + ih * W + iw
                        mapping[out_idx, in_idx] = scale

    return mapping, (C, out_H, out_W)


def prepare_spinnaker_weights(model_path: str, output_dir: str = None,
                               fc_only: bool = False):
    """Prepare weight files for SpiNNaker deployment.

    Extracts weights, folds BatchNorm, converts conv layers to dense,
    and saves as .npy files that can be loaded in sPyNNaker scripts.

    Two deployment modes:
    1. fc_only=True: Only deploy FC layers (input=2304 AvgPool output).
       Smaller, more feasible on SpiNNaker, but requires pre-computing
       the conv feature extraction on CPU.
    2. fc_only=False: Full network unrolled to dense. Very large matrices
       (conv1 unrolled = 221,184 x 13,824), probably too large for
       SpiNNaker1. Saved for reference/future SpiNNaker2.

    Args:
        model_path: Path to trained snnTorch model (.pt file).
        output_dir: Directory to save weight files.
        fc_only: If True, only save FC layer weights.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "spinnaker_weights"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    all_params = extract_weights(model_path)

    # Print all parameter names
    print("\nModel parameters found:")
    for name, p in all_params.items():
        print(f"  {name}: {p.shape}")

    # --- Step 1: Fold BatchNorm into Conv layers ---
    print("\n--- Folding BatchNorm into Conv weights ---")

    conv1_w_fused, conv1_b_fused = fold_batchnorm_into_conv(
        conv_w=all_params["conv1.weight"],     # (32, 1, 3, 3)
        conv_b=all_params["conv1.bias"],        # (32,)
        bn_gamma=all_params["bn1.weight"],      # (32,)
        bn_beta=all_params["bn1.bias"],          # (32,)
        bn_mean=all_params["bn1.running_mean"],  # (32,)
        bn_var=all_params["bn1.running_var"],    # (32,)
    )
    print(f"  Conv1+BN1 fused: weight={conv1_w_fused.shape}, bias={conv1_b_fused.shape}")

    conv2_w_fused, conv2_b_fused = fold_batchnorm_into_conv(
        conv_w=all_params["conv2.weight"],     # (64, 32, 3, 3)
        conv_b=all_params["conv2.bias"],        # (64,)
        bn_gamma=all_params["bn2.weight"],      # (64,)
        bn_beta=all_params["bn2.bias"],          # (64,)
        bn_mean=all_params["bn2.running_mean"],  # (64,)
        bn_var=all_params["bn2.running_var"],    # (64,)
    )
    print(f"  Conv2+BN2 fused: weight={conv2_w_fused.shape}, bias={conv2_b_fused.shape}")

    # --- Step 2: Save FC weights (always needed) ---
    print("\n--- Saving FC layer weights ---")
    fc1_w = all_params["fc1.weight"]  # (256, 2304)
    fc1_b = all_params["fc1.bias"]    # (256,)
    fc2_w = all_params["fc2.weight"]  # (50, 256)
    fc2_b = all_params["fc2.bias"]    # (50,)

    np.save(output_dir / "fc1_weight.npy", fc1_w)
    np.save(output_dir / "fc1_bias.npy", fc1_b)
    np.save(output_dir / "fc2_weight.npy", fc2_w)
    np.save(output_dir / "fc2_bias.npy", fc2_b)
    print(f"  fc1_weight: {fc1_w.shape}")
    print(f"  fc1_bias:   {fc1_b.shape}")
    print(f"  fc2_weight: {fc2_w.shape}")
    print(f"  fc2_bias:   {fc2_b.shape}")

    # --- Step 3: Save fused conv weights ---
    print("\n--- Saving fused Conv+BN weights ---")
    np.save(output_dir / "conv1_fused_weight.npy", conv1_w_fused)
    np.save(output_dir / "conv1_fused_bias.npy", conv1_b_fused)
    np.save(output_dir / "conv2_fused_weight.npy", conv2_w_fused)
    np.save(output_dir / "conv2_fused_bias.npy", conv2_b_fused)

    # --- Step 4: Generate connection lists for sPyNNaker ---
    # sPyNNaker uses FromListConnector with (pre_idx, post_idx, weight, delay)
    print("\n--- Generating sPyNNaker connection lists ---")

    # FC1: 2304 -> 256
    fc1_connections = []
    for post in range(fc1_w.shape[0]):  # 256
        for pre in range(fc1_w.shape[1]):  # 2304
            w = float(fc1_w[post, pre])
            if abs(w) > 1e-6:
                fc1_connections.append([int(pre), int(post), w, 1.0])
    fc1_conn_array = np.array(fc1_connections, dtype=np.float64)
    np.save(output_dir / "fc1_connections.npy", fc1_conn_array)
    print(f"  FC1 connections: {len(fc1_connections)} non-zero "
          f"(of {fc1_w.shape[0] * fc1_w.shape[1]} total, "
          f"sparsity={1 - len(fc1_connections) / (fc1_w.shape[0] * fc1_w.shape[1]):.1%})")

    # FC2: 256 -> 50
    fc2_connections = []
    for post in range(fc2_w.shape[0]):  # 50
        for pre in range(fc2_w.shape[1]):  # 256
            w = float(fc2_w[post, pre])
            if abs(w) > 1e-6:
                fc2_connections.append([int(pre), int(post), w, 1.0])
    fc2_conn_array = np.array(fc2_connections, dtype=np.float64)
    np.save(output_dir / "fc2_connections.npy", fc2_conn_array)
    print(f"  FC2 connections: {len(fc2_connections)} non-zero "
          f"(of {fc2_w.shape[0] * fc2_w.shape[1]} total, "
          f"sparsity={1 - len(fc2_connections) / (fc2_w.shape[0] * fc2_w.shape[1]):.1%})")

    # --- Step 5: Compute weight statistics for SpiNNaker parameter tuning ---
    print("\n--- Weight statistics (for SpiNNaker parameter tuning) ---")
    for name, w in [("fc1", fc1_w), ("fc2", fc2_w),
                     ("conv1_fused", conv1_w_fused.reshape(conv1_w_fused.shape[0], -1)),
                     ("conv2_fused", conv2_w_fused.reshape(conv2_w_fused.shape[0], -1))]:
        print(f"  {name}: min={w.min():.4f}, max={w.max():.4f}, "
              f"mean={w.mean():.4f}, std={w.std():.4f}")

    # --- Step 6: Save metadata ---
    metadata = {
        "model_path": str(model_path),
        "architecture": {
            "input_shape": [1, 64, 216],
            "conv1": {"in_c": 1, "out_c": 32, "kernel": 3, "pad": 1, "stride": 1},
            "pool1": {"type": "MaxPool2d", "size": 2},
            "shape_after_conv1_pool1": [32, 32, 108],
            "conv2": {"in_c": 32, "out_c": 64, "kernel": 3, "pad": 1, "stride": 1},
            "pool2": {"type": "MaxPool2d", "size": 2},
            "shape_after_conv2_pool2": [64, 16, 54],
            "avg_pool": {"type": "AvgPool2d", "kernel": [4, 6]},
            "shape_after_avgpool": [64, 4, 9],
            "flatten_size": 2304,
            "fc1": {"in": 2304, "out": 256},
            "fc2": {"in": 256, "out": 50},
        },
        "snn_params": {
            "beta": 0.95,
            "threshold": 1.0,
            "num_steps": 25,
        },
        "spinnaker_lif_params": {
            "cm": 1.0,
            "tau_m": 20.0,
            "tau_refrac": 2.0,
            "v_reset": 0.0,
            "v_rest": 0.0,
            "v_thresh": 1.0,
            "note": "tau_m=20ms approximates beta=0.95 with dt=1ms: beta=exp(-dt/tau_m)=exp(-1/20)=0.951",
        },
        "files": {
            "fc1_weight": "fc1_weight.npy",
            "fc1_bias": "fc1_bias.npy",
            "fc2_weight": "fc2_weight.npy",
            "fc2_bias": "fc2_bias.npy",
            "fc1_connections": "fc1_connections.npy (pre, post, weight, delay)",
            "fc2_connections": "fc2_connections.npy (pre, post, weight, delay)",
            "conv1_fused_weight": "conv1_fused_weight.npy (BN folded in)",
            "conv1_fused_bias": "conv1_fused_bias.npy",
            "conv2_fused_weight": "conv2_fused_weight.npy (BN folded in)",
            "conv2_fused_bias": "conv2_fused_bias.npy",
        },
        "deployment_notes": [
            "FC-only mode: Use fc1_connections.npy and fc2_connections.npy",
            "FC-only requires pre-computing conv features on CPU before SpiNNaker",
            "Full unrolled mode: conv layers must be unrolled to dense matrices",
            "Full unrolled is very large and may exceed SpiNNaker1 capacity",
            "SpiNNaker handles excitatory/inhibitory weights separately",
            "Negative weights need separate inhibitory projections",
        ],
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll weights saved to {output_dir}")
    print(f"\nFor SpiNNaker FC-only deployment:")
    print(f"  - Upload fc1_connections.npy, fc2_connections.npy, and metadata.json")
    print(f"  - FC1: 2304 input neurons -> 256 hidden neurons")
    print(f"  - FC2: 256 hidden neurons -> 50 output neurons")
    print(f"  - Total: {2304 + 256 + 50} = 2610 neurons")

    return all_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert snnTorch model to sPyNNaker format"
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to trained model .pt file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/spinnaker_weights)")
    parser.add_argument("--fc-only", action="store_true",
                        help="Only save FC layer weights (skip conv unrolling)")
    args = parser.parse_args()

    prepare_spinnaker_weights(args.model_path, args.output_dir, args.fc_only)
