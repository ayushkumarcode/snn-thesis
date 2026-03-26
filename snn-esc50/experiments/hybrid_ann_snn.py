"""
hybrid_ann_snn.py -- ANN-to-SNN hybrid initialization with fine-tuning.

Research question: Can transferring learned ANN weights to an enhanced SNN
(with learnable beta, threshold, dropout, and spike_rate_escape surrogate)
close the ANN-SNN accuracy gap?

Method:
  - Load trained ANN weights from results/ann/none/best_fold{fold}.pt
  - Map ANN conv/bn/fc layers to corresponding SNN layers
  - Create EnhancedSpikingCNN with learn_beta=True, learn_threshold=True,
    Dropout(0.3), spike_rate_escape surrogate gradient
  - Fine-tune for 20 epochs at lower learning rate (1e-4)
  - 5-fold CV, save results to results/experiments/hybrid_ann_snn/

Weight mapping (ConvANN -> SpikingCNN):
  features.0  (Conv2d)     -> conv1
  features.1  (BatchNorm)  -> bn1
  features.4  (Conv2d)     -> conv2
  features.5  (BatchNorm)  -> bn2
  classifier.0 (Linear)    -> fc1
  classifier.3 (Linear)    -> fc2

Usage:
  cd snn-esc50
  source .venv/bin/activate
  python experiments/hybrid_ann_snn.py
  python experiments/hybrid_ann_snn.py --fold 1 --epochs 20
