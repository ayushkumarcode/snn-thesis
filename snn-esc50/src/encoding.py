"""
Spike encoding methods for converting mel-spectrograms to spike trains.

Four encoding methods are implemented:
  - Rate coding: spike probability proportional to input intensity
  - Delta coding: spikes on temporal changes (increases)
  - Latency coding: higher intensity → earlier spike
  - Direct coding: no explicit encoding; feed normalised values directly
    to the SNN at each timestep (the network learns its own representation)
"""

import torch
import snntorch.spikegen as spikegen

from src.config import NUM_STEPS


def encode_rate(data: torch.Tensor, num_steps: int = NUM_STEPS) -> torch.Tensor:
    """Rate coding: spike probability proportional to pixel intensity.

    Args:
        data: Tensor of shape (batch, channels, height, width) in [0, 1].
        num_steps: Number of timesteps.

    Returns:
        Spike tensor of shape (num_steps, batch, channels, height, width).
    """
    return spikegen.rate(data, num_steps=num_steps)


def encode_delta(data: torch.Tensor, num_steps: int = NUM_STEPS,
                 threshold: float = 0.1) -> torch.Tensor:
    """Delta coding: spikes generated from temporal differences.

    Since our input is a static spectrogram (no inherent time axis for
    encoding), we tile it across timesteps and let delta detect changes.
    In practice, delta encoding is more meaningful when the input varies
    over time. Here we apply small Gaussian noise per timestep to create
    temporal variation, then use delta modulation.

    Args:
        data: Tensor of shape (batch, channels, height, width) in [0, 1].
        num_steps: Number of timesteps.
        threshold: Change threshold for spike generation.

    Returns:
        Spike tensor of shape (num_steps, batch, channels, height, width).
    """
    # Create a time-varying version by adding small noise per step
    tiled = data.unsqueeze(0).repeat(num_steps, *([1] * data.dim()))
    noise = torch.randn_like(tiled) * 0.02
    tiled = torch.clamp(tiled + noise, 0, 1)

    # Use snntorch delta encoding on the time-varying data
    spikes = spikegen.delta(tiled, threshold=threshold, off_spike=False)
    return spikes


def encode_latency(data: torch.Tensor, num_steps: int = NUM_STEPS,
                   tau: float = 5.0, normalize: bool = True) -> torch.Tensor:
    """Latency coding: higher intensity → earlier spike time.

    Each input value produces exactly one spike; the spike time is
    inversely related to the input magnitude.

    Args:
        data: Tensor of shape (batch, channels, height, width) in [0, 1].
        num_steps: Number of timesteps.
        tau: Time constant controlling spike time spread.
        normalize: Whether to normalize the input first.

    Returns:
        Spike tensor of shape (num_steps, batch, channels, height, width).
    """
    # Clamp to avoid log(0) issues in latency encoding
    data_clamped = torch.clamp(data, 1e-6, 1.0)
    return spikegen.latency(
        data_clamped, num_steps=num_steps, tau=tau,
        threshold=0.01, normalize=normalize, linear=True,
    )


def encode_direct(data: torch.Tensor, num_steps: int = NUM_STEPS) -> torch.Tensor:
    """Direct encoding: repeat the normalised input at each timestep.

    No explicit spike conversion -- the SNN's LIF neurons handle
    the conversion internally through membrane potential dynamics.
    This is the simplest approach and often competitive.

    Args:
        data: Tensor of shape (batch, channels, height, width) in [0, 1].
        num_steps: Number of timesteps.

    Returns:
        Tensor of shape (num_steps, batch, channels, height, width).
        (Not binary spikes -- continuous values fed to spiking neurons.)
    """
    return data.unsqueeze(0).repeat(num_steps, *([1] * data.dim()))


def encode_burst(data: torch.Tensor, num_steps: int = NUM_STEPS,
                 max_spikes: int = 5) -> torch.Tensor:
    """Burst coding: spike count ∝ intensity, all spikes in first N timesteps.

    Higher intensity → more spikes, concentrated at the beginning of the
    simulation window. This produces a dense, early burst for bright pixels
    and silence for dark pixels. Biologically motivated by bursting neurons
    in auditory cortex.

    Args:
        data: Tensor of shape (batch, channels, height, width) in [0, 1].
        num_steps: Number of timesteps (must be >= max_spikes).
        max_spikes: Maximum number of spikes per neuron (default: 5).

    Returns:
        Spike tensor of shape (num_steps, batch, channels, height, width).
    """
    # n_spikes[i] = round(data[i] * max_spikes), clamped to [0, max_spikes]
    n_spikes = (data * max_spikes).round().long().clamp(0, max_spikes)
    # neuron fires at timestep t if t < n_spikes[neuron]
    spikes = torch.zeros(num_steps, *data.shape, device=data.device)
    for t in range(min(num_steps, max_spikes)):
        spikes[t] = (t < n_spikes).float()
    return spikes


def encode_phase(data: torch.Tensor, num_steps: int = NUM_STEPS) -> torch.Tensor:
    """Phase coding: spike timing relative to a global oscillation cycle.

    Encodes intensity as spike time within a single oscillation period.
    High intensity → early spike (small phase offset).
    Low intensity → late spike or no spike.

    Biologically motivated by theta-phase precession in hippocampus and
    phase-of-firing codes in auditory cortex.

    Args:
        data: Tensor of shape (batch, channels, height, width) in [0, 1].
        num_steps: Number of timesteps (= one oscillation period).

    Returns:
        Spike tensor of shape (num_steps, batch, channels, height, width).
        Each neuron fires exactly once at timestep t = floor((1 - data) * num_steps),
        unless intensity == 0 (silent neuron).
    """
    # spike_time[i] = floor((1 - data[i]) * (num_steps - 1))
    # High intensity (≈1) → spike_time ≈ 0 (fires early)
    # Low intensity (≈0) → spike_time ≈ num_steps-1 (fires late)
    spike_time = ((1.0 - data) * (num_steps - 1)).long().clamp(0, num_steps - 1)
    spikes = torch.zeros(num_steps, *data.shape, device=data.device)
    for t in range(num_steps):
        spikes[t] = (spike_time == t).float()
    # Zero out silent neurons (data == 0 exactly — no spike at all)
    silent = (data == 0.0)
    for t in range(num_steps):
        spikes[t] = spikes[t] * (~silent).float()
    return spikes


# Registry for easy access by name
ENCODERS = {
    "rate": encode_rate,
    "delta": encode_delta,
    "latency": encode_latency,
    "direct": encode_direct,
    "burst": encode_burst,
    "phase": encode_phase,
}


def get_encoder(name: str):
    """Get encoding function by name."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoding '{name}'. Choose from {list(ENCODERS.keys())}")
    return ENCODERS[name]
