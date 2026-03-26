"""
info_bottleneck_snn.py -- Information Bottleneck SNN experiment.

Compress spike representations via variational information bottleneck.
Based on "Learning to Time-Decode via Information Bottleneck" (NeurIPS, 2024).

Adds a variational information bottleneck (VIB) after the hidden layer (lif3):
  - mu = Linear(256, 256)
  - logvar = Linear(256, 256)
  - z = mu + exp(0.5 * logvar) * N(0,1)  (reparameterisation trick)
  - KL_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
  - Total loss = CE_loss + beta_ib * KL_loss

During training, z replaces spk3 as input to FC2. The KL term encourages
the 256-dim hidden spike representation to be maximally compressed while
retaining task-relevant information. This acts as a powerful regulariser
on the small 1600-sample ESC-50 dataset.

During evaluation: use mu directly (no sampling, no stochasticity).

Also uses: learn_beta=True, Dropout(0.3), spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/info_bottleneck_snn.py --fold 1
    python experiments/info_bottleneck_snn.py --beta-ib 1e-3    # all 5 folds
"""
