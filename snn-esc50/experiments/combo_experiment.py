"""
Combinatorial experiment runner: stack multiple SNN techniques via flags.

Usage:
    # Rhythm + KD:
    python -m experiments.combo_experiment --rhythm --kd --device cuda

    # Rhythm + hybrid init + TET:
    python -m experiments.combo_experiment --rhythm --hybrid-init --tet --device cuda

    # Kitchen sink:
    python -m experiments.combo_experiment --rhythm --dendritic --delays --kd --tet --cochleagram --device cuda

    # Single fold test:
    python -m experiments.combo_experiment --rhythm --kd --fold 1 --device cuda

Techniques available:
    --learn-beta       Learnable beta per neuron (learn_beta=True)
    --learn-threshold  Learnable threshold per neuron
    --dropout          Dropout(0.3) before fc2
    --sre              spike_rate_escape surrogate (default: fast_sigmoid)
    --rhythm           Oscillatory modulation (Nature Comms 2025)
    --dendritic        Multi-compartment dendritic neurons (K=3 branches)
    --delays           Learnable synaptic delays on FC layers
    --kd               Knowledge distillation from ANN teacher
    --hybrid-init      Initialize SNN weights from trained ANN
    --tet              Temporal Efficient Training loss
    --cochleagram      Gammatone filterbank instead of mel spectrogram
    --l1-reg LAMBDA    L1 spike rate regularization

All techniques are composable. Results saved to results/experiments/combo_<hash>/
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    NUM_CLASSES, NUM_STEPS, BETA, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device, SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH,
)
from src.dataset import download_esc50, get_fold_dataloaders, ESC50Dataset
from src.encoding import encode_direct
from src.models.ann_model import ConvANN


# ============================================================
# Custom neuron modules
# ============================================================

class RhythmLIF(nn.Module):
    """LIF neuron with learnable oscillatory modulation."""

    def __init__(self, size, beta=BETA, spike_grad=None, learn_beta=True):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.spike_grad = spike_grad
        self.size = size

        # Learnable beta via sigmoid
        beta_init = math.log(beta / (1 - beta))
        self.beta_raw = nn.Parameter(torch.full((size,), beta_init))

        # Oscillation parameters
        self.amplitude = nn.Parameter(torch.full((size,), 0.1))
        self.frequency = nn.Parameter(torch.empty(size).uniform_(0.5, 5.0))
        self.phase = nn.Parameter(torch.zeros(size))

        self.threshold = nn.Parameter(torch.ones(size)) if learn_beta else None
        self._threshold_val = 1.0

    def init_mem(self, device=None):
        if device is None:
            device = self.beta_raw.device
        return torch.zeros(1, device=device)

    def forward(self, x, mem, step):
        if mem.device != x.device:
            mem = mem.to(x.device)
        beta = torch.sigmoid(self.beta_raw)
        thresh = self.threshold if self.threshold is not None else self._threshold_val

        # Reshape for broadcasting
        shape = [1] * (x.dim() - 1) + [-1]
        if x.dim() == 4:  # conv: (batch, channels, h, w)
            shape = [1, -1, 1, 1]
        elif x.dim() == 2:  # fc: (batch, features)
            shape = [1, -1]

        b = beta.view(shape)
        t = thresh.view(shape) if isinstance(thresh, nn.Parameter) else thresh
        osc = (self.amplitude * torch.sin(
            2 * math.pi * self.frequency * step / NUM_STEPS + self.phase
        )).view(shape)

        mem = b * mem + x + osc
        spk = self.spike_grad((mem - t) / t)
        mem = mem * (1 - spk.detach())
        return spk, mem


class DendriticLIF(nn.Module):
    """Multi-compartment dendritic neuron with K branches."""

    def __init__(self, size, K=3, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.spike_grad = spike_grad
        self.K = K
        self.size = size

        # Branch betas (fast, medium, slow)
        init_betas = [0.7, 0.9, 0.99] if K == 3 else [0.7 + 0.3 * i / (K - 1) for i in range(K)]
        self.branch_beta_raw = nn.ParameterList([
            nn.Parameter(torch.full((size,), math.log(b / (1 - b)))) for b in init_betas
        ])
        # Branch gating weights
        self.gate_logits = nn.Parameter(torch.zeros(K, size))
        self.threshold = nn.Parameter(torch.ones(size))

    def init_mem(self, device=None):
        if device is None:
            device = self.branch_beta_raw[0].device
        return [torch.zeros(1, device=device) for _ in range(self.K)]

    def forward(self, x, branch_mems, step=None):
        branch_mems = [m.to(x.device) if m.device != x.device else m for m in branch_mems]
        shape = [1] * (x.dim() - 1) + [-1]
        if x.dim() == 4:
            shape = [1, -1, 1, 1]
        elif x.dim() == 2:
            shape = [1, -1]

        gates = F.softmax(self.gate_logits, dim=0)  # (K, size)
        soma = torch.zeros_like(x)
        new_mems = []

        for k in range(self.K):
            beta_k = torch.sigmoid(self.branch_beta_raw[k]).view(shape)
            gate_k = gates[k].view(shape)
            mem_k = beta_k * branch_mems[k] + gate_k * x
            new_mems.append(mem_k)
            soma = soma + mem_k

        thresh = self.threshold.view(shape)
        spk = self.spike_grad((soma - thresh) / thresh)

        # Reset all branches on spike
        reset = spk.detach()
        new_mems = [m * (1 - reset) for m in new_mems]

        return spk, new_mems


class DelayedLinear(nn.Module):
    """Linear layer with per-output-neuron learnable delays."""

    def __init__(self, in_features, out_features, max_delay=5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.max_delay = max_delay
        self.delay_raw = nn.Parameter(torch.empty(out_features).uniform_(0.1, 0.9))
        self.buffer = None

    def init_buffer(self, batch_size, in_features, device):
        self.buffer = torch.zeros(self.max_delay + 1, batch_size, in_features, device=device)

    def forward(self, x):
        if self.buffer is None or self.buffer.shape[1] != x.shape[0]:
            self.init_buffer(x.shape[0], x.shape[1], x.device)

        # Shift buffer and add new input
        self.buffer = torch.roll(self.buffer, 1, dims=0)
        self.buffer[0] = x

        # Soft delay lookup (differentiable) — fully vectorized
        delays = torch.sigmoid(self.delay_raw) * self.max_delay  # (out_features,)
        delay_floor = delays.long().clamp(0, self.max_delay - 1)
        delay_ceil = (delay_floor + 1).clamp(0, self.max_delay)
        delay_frac = (delays - delay_floor.float()).view(-1, 1, 1)  # (out_features, 1, 1)

        # Gather delayed inputs: buffer is (max_delay+1, batch, in_features)
        buf_lo = self.buffer[delay_floor]  # (out_features, batch, in_features)
        buf_hi = self.buffer[delay_ceil]   # (out_features, batch, in_features)
        # Interpolate: (out_features, batch, in_features) — frac broadcasts over batch & in_features
        delayed = (1 - delay_frac) * buf_lo + delay_frac * buf_hi
        # delayed[j] is the delayed input for output neuron j, shape (batch, in_features)
        # Apply weight matrix: out[batch, j] = sum_i(W[j,i] * delayed[j, batch, i]) + bias[j]
        # This is: (batch, out, in) * (out, in) summed over in
        delayed = delayed.permute(1, 0, 2)  # (batch, out_features, in_features)
        out = (delayed * self.linear.weight.unsqueeze(0)).sum(-1) + self.linear.bias

        return out

    def reset(self):
        self.buffer = None


# ============================================================
# Cochleagram preprocessing
# ============================================================

def gammatone_filterbank(sr, n_fft, n_filters=64, fmin=50):
    """Create gammatone filterbank matrix."""
    import librosa
    fmax = sr / 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # ERB-spaced center frequencies
    ear_q = 9.26449
    min_bw = 24.7
    cf_low = -(ear_q * min_bw) + np.exp(np.log(fmin + ear_q * min_bw) +
              np.arange(0, n_filters) * (-np.log(fmax + ear_q * min_bw) +
              np.log(fmin + ear_q * min_bw)) / (n_filters - 1)) * 0
    cfs = np.exp(np.linspace(np.log(fmin + ear_q * min_bw),
                              np.log(fmax + ear_q * min_bw), n_filters)) - ear_q * min_bw

    fb = np.zeros((n_filters, len(freqs)))
    for i, cf in enumerate(cfs):
        erb = 24.7 * (4.37 * cf / 1000 + 1)
        b = 1.019 * 2 * np.pi * erb
        resp = ((2 * np.pi * np.abs(freqs - cf)) ** 2 + b ** 2) ** (-2)
        fb[i] = resp / (resp.max() + 1e-10)

    return fb.astype(np.float32)


def wav_to_cochleagram(filepath, fb_matrix):
    """Convert audio to gammatone cochleagram."""
    import librosa
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    expected_len = SAMPLE_RATE * DURATION
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    cochleagram = fb_matrix @ S
    cochleagram = librosa.power_to_db(cochleagram + 1e-10, ref=np.max)

    # Normalize to [0, 1]
    mn, mx = cochleagram.min(), cochleagram.max()
    if mx - mn > 0:
        cochleagram = (cochleagram - mn) / (mx - mn)
    else:
        cochleagram = np.zeros_like(cochleagram)

    return cochleagram.astype(np.float32)


# ============================================================
# Combo model builder
# ============================================================

class ComboSpikingCNN(nn.Module):
    """Configurable SNN that can combine any techniques."""

    def __init__(self, args):
        super().__init__()
        self.num_steps = NUM_STEPS
        self.args = args
        self.use_delays = args.delays
        self.use_dendritic = args.dendritic
        self.use_rhythm = args.rhythm

        sg = surrogate.spike_rate_escape(beta=1, slope=25) if args.sre else surrogate.fast_sigmoid(slope=25)

        # Conv layers (shared across all configs)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC layers
        if args.delays:
            self.fc1 = DelayedLinear(64 * 4 * 9, 256, max_delay=args.max_delay)
            self.fc2 = DelayedLinear(256, NUM_CLASSES, max_delay=args.max_delay)
        else:
            self.fc1 = nn.Linear(64 * 4 * 9, 256)
            self.fc2 = nn.Linear(256, NUM_CLASSES)

        # Dropout
        self.dropout = nn.Dropout(0.3) if args.dropout else nn.Identity()

        # Neuron layers
        if args.dendritic:
            self.n1 = DendriticLIF(32, K=args.branches, spike_grad=sg)
            self.n2 = DendriticLIF(64, K=args.branches, spike_grad=sg)
            self.n3 = DendriticLIF(256, K=args.branches, spike_grad=sg)
            self.n4 = DendriticLIF(NUM_CLASSES, K=args.branches, spike_grad=sg)
        elif args.rhythm:
            self.n1 = RhythmLIF(32, BETA, sg, args.learn_beta)
            self.n2 = RhythmLIF(64, BETA, sg, args.learn_beta)
            self.n3 = RhythmLIF(256, BETA, sg, args.learn_beta)
            self.n4 = RhythmLIF(NUM_CLASSES, BETA, sg, args.learn_beta)
        else:
            self.n1 = snn.Leaky(beta=BETA, spike_grad=sg,
                                 learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
            self.n2 = snn.Leaky(beta=BETA, spike_grad=sg,
                                 learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
            self.n3 = snn.Leaky(beta=BETA, spike_grad=sg,
                                 learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
            self.n4 = snn.Leaky(beta=BETA, spike_grad=sg,
                                 learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)

    def _init_states(self, device):
        if self.use_dendritic:
            return [n.init_mem(device) for n in [self.n1, self.n2, self.n3, self.n4]]
        elif self.use_rhythm:
            return [n.init_mem(device) for n in [self.n1, self.n2, self.n3, self.n4]]
        else:
            return [n.init_leaky() for n in [self.n1, self.n2, self.n3, self.n4]]

    def forward(self, x):
        device = x.device
        states = self._init_states(device)
        m1, m2, m3, m4 = states

        if self.use_delays:
            self.fc1.reset()
            self.fc2.reset()

        spk_rec, mem_rec = [], []
        total_spikes = 0

        for step in range(self.num_steps):
            x_t = x[step]

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            if self.use_dendritic:
                spk1, m1 = self.n1(cur1, m1, step)
            elif self.use_rhythm:
                spk1, m1 = self.n1(cur1, m1, step)
            else:
                spk1, m1 = self.n1(cur1, m1)

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            if self.use_dendritic:
                spk2, m2 = self.n2(cur2, m2, step)
            elif self.use_rhythm:
                spk2, m2 = self.n2(cur2, m2, step)
            else:
                spk2, m2 = self.n2(cur2, m2)

            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = self.fc1(flat)
            if self.use_dendritic:
                spk3, m3 = self.n3(cur3, m3, step)
            elif self.use_rhythm:
                spk3, m3 = self.n3(cur3, m3, step)
            else:
                spk3, m3 = self.n3(cur3, m3)

            spk3 = self.dropout(spk3)

            cur4 = self.fc2(spk3)
            if self.use_dendritic:
                spk4, m4 = self.n4(cur4, m4, step)
                # For dendritic, mem = soma potential (sum of branches)
                mem4_scalar = sum(m4)
            elif self.use_rhythm:
                spk4, m4 = self.n4(cur4, m4, step)
                mem4_scalar = m4
            else:
                spk4, m4 = self.n4(cur4, m4)
                mem4_scalar = m4

            spk_rec.append(spk4)
            mem_rec.append(mem4_scalar)
            total_spikes += spk4.sum().item()

        return torch.stack(spk_rec), torch.stack(mem_rec), total_spikes


# ============================================================
# Training functions
# ============================================================

def compute_loss(mem_out, targets, args, spk_total=0, teacher_logits=None):
    """Compute loss based on selected techniques."""
    criterion = nn.CrossEntropyLoss()
    T_steps = mem_out.shape[0]
    device = mem_out.device

    if args.tet:
        # TET loss: mean CE + lambda * temporal variance
        per_step_losses = []
        for step in range(T_steps):
            per_step_losses.append(criterion(mem_out[step], targets))
        per_step = torch.stack(per_step_losses)
        mean_loss = per_step.mean()
        var_loss = ((per_step - mean_loss) ** 2).mean()
        loss = mean_loss + args.lambda_tet * var_loss
    else:
        # Standard per-timestep CE
        loss = torch.zeros(1, device=device)
        for step in range(T_steps):
            loss += criterion(mem_out[step], targets)
        loss = loss / T_steps

    # Knowledge distillation
    if args.kd and teacher_logits is not None:
        student_logits = mem_out.sum(dim=0)
        T = args.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        loss = args.alpha * loss + (1 - args.alpha) * kd_loss

    # L1 spike rate regularization
    if args.l1_reg > 0 and spk_total > 0:
        loss = loss + args.l1_reg * spk_total

    return loss


def load_teacher(fold, device):
    """Load trained ANN teacher."""
    weight_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"
    teacher = ConvANN().to(device)
    teacher.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    teacher.eval()
    return teacher


def load_ann_weights_into_snn(model, fold, device):
    """Map ANN weights to SNN for hybrid init."""
    ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"
    ann_state = torch.load(ann_path, map_location=device, weights_only=True)

    mapping = {
        "features.0.weight": "conv1.weight", "features.0.bias": "conv1.bias",
        "features.1.weight": "bn1.weight", "features.1.bias": "bn1.bias",
        "features.1.running_mean": "bn1.running_mean", "features.1.running_var": "bn1.running_var",
        "features.1.num_batches_tracked": "bn1.num_batches_tracked",
        "features.4.weight": "conv2.weight", "features.4.bias": "conv2.bias",
        "features.5.weight": "bn2.weight", "features.5.bias": "bn2.bias",
        "features.5.running_mean": "bn2.running_mean", "features.5.running_var": "bn2.running_var",
        "features.5.num_batches_tracked": "bn2.num_batches_tracked",
    }
    # FC layers - handle DelayedLinear wrapper
    fc1_prefix = "fc1.linear" if hasattr(model.fc1, 'linear') else "fc1"
    fc2_prefix = "fc2.linear" if hasattr(model.fc2, 'linear') else "fc2"
    mapping["classifier.0.weight"] = f"{fc1_prefix}.weight"
    mapping["classifier.0.bias"] = f"{fc1_prefix}.bias"
    mapping["classifier.3.weight"] = f"{fc2_prefix}.weight"
    mapping["classifier.3.bias"] = f"{fc2_prefix}.bias"

    snn_state = model.state_dict()
    transferred = 0
    for ann_key, snn_key in mapping.items():
        if ann_key in ann_state and snn_key in snn_state:
            snn_state[snn_key] = ann_state[ann_key]
            transferred += 1

    model.load_state_dict(snn_state)
    print(f"  Transferred {transferred} weight tensors from ANN")
    return model


def train_epoch(model, loader, optimizer, device, args, teacher=None, encoder_fn=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder_fn(data).to(device)

        optimizer.zero_grad()

        teacher_logits = None
        if args.kd and teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(data)

        spk_out, mem_out, spk_total = model(spk_input)
        loss = compute_loss(mem_out, targets, args, spk_total, teacher_logits)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_model(model, loader, device, encoder_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder_fn(data).to(device)
        spk_out, mem_out, _ = model(spk_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


# ============================================================
# Main
# ============================================================

def get_experiment_name(args):
    """Generate a descriptive name from active flags."""
    parts = []
    if args.rhythm: parts.append("rhythm")
    if args.dendritic: parts.append("dendritic")
    if args.delays: parts.append("delays")
    if args.kd: parts.append("kd")
    if args.hybrid_init: parts.append("hybrid")
    if args.tet: parts.append("tet")
    if args.cochleagram: parts.append("cochleagram")
    if args.learn_beta: parts.append("lbeta")
    if args.learn_threshold: parts.append("lthresh")
    if args.dropout: parts.append("drop")
    if args.sre: parts.append("sre")
    if args.l1_reg > 0: parts.append(f"l1_{args.l1_reg}")
    return "combo_" + "_".join(parts) if parts else "combo_baseline"


def run_fold(fold, args, device):
    exp_name = get_experiment_name(args)
    print(f"\n{'='*60}")
    print(f"  {exp_name} | Fold {fold}/5 | Device: {device}")
    print(f"  Techniques: {[p for p in exp_name.split('_')[1:]]}")
    print(f"{'='*60}")

    # Data loading
    if args.cochleagram:
        # Use cochleagram dataset
        from src.dataset import ESC50_AUDIO_DIR, ESC50_META_PATH
        import pandas as pd
        fb = gammatone_filterbank(SAMPLE_RATE, N_FFT, 64)

        train_folds = [f for f in range(1, 6) if f != fold]
        meta = pd.read_csv(ESC50_META_PATH)

        print(f"  Loading cochleagram data...")
        train_data, train_labels = [], []
        for _, row in meta[meta["fold"].isin(train_folds)].iterrows():
            cg = wav_to_cochleagram(str(ESC50_AUDIO_DIR / row["filename"]), fb)
            train_data.append(cg)
            train_labels.append(row["target"])

        test_data, test_labels = [], []
        for _, row in meta[meta["fold"] == fold].iterrows():
            cg = wav_to_cochleagram(str(ESC50_AUDIO_DIR / row["filename"]), fb)
            test_data.append(cg)
            test_labels.append(row["target"])

        train_data = torch.tensor(np.array(train_data)).unsqueeze(1)
        test_data = torch.tensor(np.array(test_data)).unsqueeze(1)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        train_ds = torch.utils.data.TensorDataset(train_data, train_labels)
        test_ds = torch.utils.data.TensorDataset(test_data, test_labels)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    encoder_fn = encode_direct

    # Model
    model = ComboSpikingCNN(args).to(device)

    if args.hybrid_init:
        model = load_ann_weights_into_snn(model, fold, device)

    teacher = load_teacher(fold, device) if args.kd else None

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {param_count:,}")

    lr = 1e-4 if args.hybrid_init else LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    epochs = args.epochs
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, args, teacher, encoder_fn)
        test_loss, test_acc = eval_model(model, test_loader, device, encoder_fn)
        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            save_dir = RESULTS_DIR / "experiments" / exp_name
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Best: {best_acc:.4f} (ep{best_epoch}) | {time.time()-start:.0f}s")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"  Fold {fold}: {best_acc:.4f} (ep{best_epoch}) in {elapsed:.1f}s")

    result = {
        "fold": fold, "experiment": exp_name,
        "best_acc": best_acc, "best_epoch": best_epoch,
        "total_epochs": epoch, "time_seconds": elapsed,
        "param_count": param_count,
        "history": history,
    }

    save_dir = RESULTS_DIR / "experiments" / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"result_fold{fold}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Combo SNN experiment")
    # Fold/device
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)

    # Techniques
    parser.add_argument("--learn-beta", action="store_true")
    parser.add_argument("--learn-threshold", action="store_true")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--sre", action="store_true")
    parser.add_argument("--rhythm", action="store_true")
    parser.add_argument("--dendritic", action="store_true")
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--delays", action="store_true")
    parser.add_argument("--max-delay", type=int, default=5)
    parser.add_argument("--kd", action="store_true")
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--hybrid-init", action="store_true")
    parser.add_argument("--tet", action="store_true")
    parser.add_argument("--lambda-tet", type=float, default=1.0)
    parser.add_argument("--cochleagram", action="store_true")
    parser.add_argument("--l1-reg", type=float, default=0.0)

    args = parser.parse_args()
    device = torch.device(args.device) if args.device else get_device()
    download_esc50()

    exp_name = get_experiment_name(args)
    folds = [args.fold] if args.fold else list(range(1, 6))
    results = []

    for fold in folds:
        result = run_fold(fold, args, device)
        results.append(result)

    if len(results) == 5:
        accs = [r["best_acc"] for r in results]
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs))**0.5
        print(f"\n{'='*60}")
        print(f"  {exp_name} 5-Fold: {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  Per-fold: {[f'{a:.4f}' for a in accs]}")
        print(f"  Baseline SNN: 47.15% | ANN: 63.85%")
        print(f"{'='*60}")

        summary = {
            "experiment": exp_name,
            "fold_accuracies": accs,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
        }
        save_dir = RESULTS_DIR / "experiments" / exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
