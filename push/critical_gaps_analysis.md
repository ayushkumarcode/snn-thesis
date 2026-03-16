# Critical Gaps Analysis — Ralph Wiggum Verification Loop
*Generated: 15 March 2026 by creative gap analysis agent*

## Top 5 Weakest Links (in priority order)

### 1. Energy argument has no real hardware numbers
SNN uses 2.1x MORE energy in software. The "on neuromorphic hardware ACs cost less" is theoretical — zero wall-clock or power measurements. For ICONS (hardware conference), this is the biggest hole.
**Fix:** Run spinnaker_latency_energy.py on SpiNNaker (needs .venv-spinnaker)

### 2. Adversarial robustness is single-fold
14.9x robustness claim from fold 4 only. Everything else is 5-fold.
**Fix:** Run adversarial on all 5 folds — CSF3 job submitted.

### 3. PANNs+Linear beats PANNs+SNN
Linear: 93.80%, ANN: 93.45%, SNN: 92.50%. SNN is worst. Reviewer: "why bother with spiking?"
**Defense:** Hardware compatibility — only SNN can run on neuromorphic hardware. But need to actually deploy PANNs+SNN on SpiNNaker to make this argument concrete.

### 4. SpiNNaker variance too high
Fold 5: 25.2% (nearly random for 50 classes). Std = 6.9%.
**Defense:** First quantified hardware gap. Document root cause (weight quantization).

### 5. "First on ESC-50" is novelty, not quality
Reviewer: "Nobody did this because it doesn't work well."
**Defense:** PANNs gap-collapse and adversarial findings are the real contributions.

## Code Verification Results
- 12/14 scripts PASS
- 2 FAIL: stochastic_resonance (MPS generator — FIXED), spinnaker_latency_energy (wrong file path — FIXED)

## Quick Wins Remaining
- [ ] Run 5-fold adversarial (CSF3) — SUBMITTED
- [x] Run temporal ablation — DONE (90% at T=7, 72% energy saving)
- [x] Run encoding transfer matrix — DONE (ratio=0.27, encoding-specific)
- [x] Run pruning resilience — DONE (SNN: 93.2% at 90% pruning, ANN: 36.8%)
- [x] Run weight distribution — DONE (ANN sparser: 38.8% vs 21.0%)
- [ ] Run SpiNNaker latency (needs SpiNNaker access)
- [ ] Wilcoxon test on PANNs results
- [ ] Run noise robustness (CSF3)
