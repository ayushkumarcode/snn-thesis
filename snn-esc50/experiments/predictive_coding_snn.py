"""
predictive_coding_snn.py -- Predictive Coding SNN experiment.

Suppress predictable spikes and transmit only prediction errors.
Based on "Predictive Coding Light" (Nature Communications, 2025).

Architecture: Same conv backbone, but after the hidden FC layer (lif3),
a prediction module tries to predict the *next* timestep's hidden spikes
from the *current* timestep's spikes. Only the prediction ERROR (residual)
is transmitted to FC2. This naturally reduces redundant spikes and acts
as a temporal regulariser on the small ESC-50 dataset.

Implementation details:
  - predict_fc = Linear(256, 256) predicts spk3[t] from spk3[t-1]
  - error[t] = spk3[t] - predict_fc(spk3[t-1])  (for t >= 1)
  - error[0] = spk3[0]  (no prediction available at first step)
  - FC2 receives error instead of spk3
  - Loss = CE_loss + lambda_pred * MSE(predict_fc(spk3[t-1]), spk3[t])
  - The MSE term encourages the network to produce predictable hidden
    representations, compressing redundancy across timesteps.

Also uses: learn_beta=True, Dropout(0.3), spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/predictive_coding_snn.py --fold 1
    python experiments/predictive_coding_snn.py              # all 5 folds
