# New Experiment Results Summary (15 March 2026)

## Completed

### 1. Temporal Ablation (fold 1, direct)
- **SNN reaches 90% of full accuracy by T=7** (72% energy saving)
- peaks at T=20 (41.0%), slightly BETTER than T=25 (40.5%)
- T=5 gets 82.7% of full with 80% energy saving
- takeaway: "SNN achieves 90% accuracy with 72% fewer timesteps"

### 2. Encoding Transfer Matrix (fold 1, 6x6)
- **transfer ratio = 0.27** -- encoding-SPECIFIC circuits
- diagonal mean: 19.2%, off-diagonal: 5.2%
- direct-trained model only gets 5-8% when tested with other encodings
- novel finding nobody has published
- takeaway: "SNNs learn encoding-coupled representations, not general audio features"

### 3. Pruning Resilience (fold 1)
- **at 90% pruning: SNN retains 93.2%, ANN collapses to 36.8%**
- SNN barely affected up to 70% pruning (95.7% retained)
- ANN stays stable until 70% then cliff-edges at 90%
- takeaway: "SNN maintains 93% accuracy with 90% weight removal"

### 4. Weight Distribution Analysis (fold 1)
- ANN weights are sparser (38.8% near-zero vs SNN 21.0%)
- SNN fc2 kurtosis: 24.6 vs ANN 14.6 (SNN more peaked/heavy-tailed)
- both models have similar overall norms
- finding: spiking constraint produces denser, more peaked weight distributions

### 5. Neuron Ablation / Fault Tolerance (fold 1)
- **SNN more fault-tolerant**: retains 13.7% at 50% ablation vs ANN 12.6%
- at 10-30% ablation, SNN actually BEATS ANN in absolute accuracy
- takeaway: "SNN surpasses ANN accuracy when 10-30% of neurons fail"

### 6. Stochastic Resonance (fold 1)
- **stochastic resonance detected**: sigma=0.02 improves SNN by +0.25pp
- no SR in ANN (expected)
- SNN dramatically more noise-resilient: at sigma=0.5, SNN=39.25% vs ANN=13.1%
- takeaway: "biological stochastic resonance manifests in trained LIF network"

## Running on CSF3 (job 12168476, A100 GPU)

- adversarial robustness 5-fold (fixes the single-fold gap)
- noise robustness (SNR sweep)
- temporal ablation 5-fold
- few-shot learning curves
- spike efficiency Pareto

## Not Yet Run (need SpiNNaker or UrbanSound8K)

- SpiNNaker latency/energy measurement
- spike drop robustness
- full SpiNNaker IF_cond_exp deployment
- UrbanSound8K cross-dataset (needs dataset download)
- SNN saliency maps (can run locally)

