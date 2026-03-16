# Final Completeness Audit (Updated 16 March 2026)

## 5-Fold Validated (DONE — 21 experiments)
All core training (7 encodings + ANN + 2 augmented), adversarial, noise,
temporal ablation, pruning, neuron ablation, stochastic resonance,
encoding transfer, PANNs, SpiNNaker FC2 5-fold, statistical tests,
**NeuroBench energy (NEW)**, **continual learning (NEW)**.

## Single-Fold — Should Extend to 5-Fold
| Experiment | Current | Action |
|-----------|---------|--------|
| ~~Continual learning~~ | ~~fold 4 only~~ | **DONE 16 March** — 5-fold: SNN 69.9%±4.3%, ANN 74.7%±2.4% |
| Surrogate ablation | fold 1, seed 42 | Check CSF3 for 3-seed run (lower priority) |
| ~~NeuroBench energy~~ | ~~fold 4 only~~ | **DONE 16 March** — 5-fold: SNN 968±37 nJ, ANN 454±11 nJ |
| Few-shot learning | fold 1 only | **RUNNING on CSF3** (job 12175826, folds 2-5) |
| Spike Pareto | fold 1 only | **RUNNING on CSF3** (job 12175826, folds 2-5) |

## Single-Fold — OK to Leave as Single-Fold
| Experiment | Why OK |
|-----------|--------|
| Saliency maps | Qualitative, IoU confirmed at 0.083 on 100 samples |
| Weight distribution | Descriptive analysis |
| Temporal analysis | Rate-vs-first-spike, separate from temporal ablation (which IS 5-fold) |
| Spike drop robustness | SpiNNaker-specific analysis |
| SpiNNaker Option A/C | Hardware calibration experiments |

## ZERO Results (Scripts Exist, Never Run)
| Script | What it does | Priority |
|--------|-------------|----------|
| urbansound8k_1fold.py | Cross-dataset validation | LOW (listed as future work in ICONS paper) |
| spinnaker_latency_energy.py | Wall-clock hardware timing | MEDIUM (would strengthen energy claims) |

## SpiNNaker Full Deployment (BREAKTHROUGH 16 March)
- ROOT CAUSE of zero hidden spikes: missing `population.initialize(v=0.0)`
- Step 3a verified: 231/256 hidden neurons fired after fix
- Step 4 (FC1+FC2 end-to-end) RUNNING — 20 samples, exc-only FC1
- If step 4 accuracy > 33.1% (FC2-only), this is a major improvement

## CSF3 Retrieval Gap
3-seed surrogate ablation was submitted to CSF3 but results were never downloaded.
Check: `ssh csf3 "ls ~/scratch/snn-esc50/results/snn/surrogate_ablation/"` for multi-seed files.

## Empty Directories
- `results/snn/saliency_5fold/` — empty, 5-fold saliency run never completed
- `saliency_sample4.png` missing from fold-1 saliency (samples 0,1,2,3,5 exist)
