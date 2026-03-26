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

