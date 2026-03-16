"""
Central configuration for SNN-ESC50 project.
All hyperparameters and paths in one place.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ESC50_DIR = DATA_DIR / "ESC-50-master"
ESC50_AUDIO_DIR = ESC50_DIR / "audio"
ESC50_META_PATH = ESC50_DIR / "meta" / "esc50.csv"

# --- Audio Processing ---
SAMPLE_RATE = 22050
DURATION = 5  # seconds (ESC-50 clips are 5s)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 0
F_MAX = None  # Nyquist

# --- Dataset ---
NUM_CLASSES = 50
NUM_FOLDS = 5
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

# --- SNN Parameters ---
NUM_STEPS = 25  # number of timesteps for spike simulation
BETA = 0.95  # membrane potential decay rate for LIF neurons
THRESHOLD = 1.0  # spike threshold

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # early stopping patience

# --- Encoding ---
ENCODING_METHODS = ["rate", "delta", "latency", "direct"]

# --- Device ---
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
