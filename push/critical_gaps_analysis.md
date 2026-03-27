# Critical Gaps Analysis

going through our weakest points honestly.

## Top 5 Weakest Links

### 1. Energy argument has no real hardware numbers
SNN uses 2.1x MORE energy in software. the "on neuromorphic hardware ACs cost less" argument is theoretical -- zero wall-clock or power measurements. for ICONS (hardware conference), this is the biggest hole.
**fix:** run spinnaker_latency_energy.py on SpiNNaker (needs .venv-spinnaker)

### 2. Adversarial robustness was single-fold
14.9x robustness claim from fold 4 only. everything else is 5-fold.
**fix:** run adversarial on all 5 folds -- CSF3 job submitted.

### 3. PANNs+Linear beats PANNs+SNN
Linear: 93.80%, ANN: 93.45%, SNN: 92.50%. SNN is worst of the three. reviewer will ask: "why bother with spiking?"
**defense:** hardware compatibility -- only SNN can run on neuromorphic hardware. but really need to actually deploy PANNs+SNN on SpiNNaker to make this concrete.

### 4. SpiNNaker variance too high
fold 5: 25.2% (nearly random for 50 classes). std = 6.9%.
**defense:** first quantified hardware gap. document root cause (weight quantization).

### 5. "First on ESC-50" is novelty, not quality
reviewer could say: "nobody did this because it doesn't work well."
**defense:** PANNs gap-collapse and adversarial findings are the real contributions. the "first" claim is the hook, not the substance.

## Code Verification
- 12/14 scripts PASS
