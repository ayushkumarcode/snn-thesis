# SNN research gaps -- achievable thesis opportunities

the SNN field is in a weird spot right now: mature enough that good tools and datasets exist, but immature enough that huge gaps remain in basic empirical coverage. most SNN papers focus on image classification (MNIST, CIFAR-10, ImageNet) with surrogate gradient training. entire application domains, datasets, and framework comparisons remain untouched or have like 1-2 papers. that's actually great for an undergrad thesis because there's plenty of room to contribute something genuinely new without needing PhD-level ambition.

the single easiest path to a real contribution is: **take an existing SNN architecture and apply it to a dataset or domain where nobody has tried it.** second easiest: **run the same experiment across multiple frameworks and report the differences.** both are basically "engineering" contributions -- running experiments and reporting results -- not "invention" contributions. but they're genuinely useful to the community and count as novel work.

---

## 1. domains where SNNs haven't been tried (or barely tried)

### completely untouched or near-untouched

| Domain | Status | Why SNNs could work | Effort |
|--------|--------|-------------------|--------|
| **plant disease detection from leaf images** | zero SNN papers found. whole agricultural CV field uses CNNs/transformers. | standard image classification; direct transfer of existing SNN architectures. | LOW |
| **wildlife camera trap classification** | nothing found. | sparse, event-like data (animals appear briefly). SNNs could exploit temporal sparsity. | LOW-MEDIUM |
| **satellite/remote sensing land cover** | one paper (SNN4Space, ESA) on EuroSAT and UC Merced. no follow-ups. | standard image classification with big datasets. energy efficiency argument strong for satellite edge computing. | LOW |
| **document/OCR classification** | nothing beyond MNIST digits. | character recognition is a natural extension of digit recognition. | LOW |
| **food recognition/calorie estimation** | nothing found. | standard image classification. Food-101, Food-2K datasets exist. | LOW |
| **weather/climate prediction from sensor data** | nothing found. | time-series data naturally maps to temporal spike encoding. | MEDIUM |
| **music genre classification** | one undergrad thesis (mrahtz, 2016) on musical pattern recognition. no genre classification. | audio temporal patterns are a natural fit. | LOW-MEDIUM |
| **sports action recognition** | no SNN papers on standard sports datasets (UCF-101, HMDB-51). | temporal dynamics of actions suit SNNs. | MEDIUM |

### barely explored (1-3 papers)

| Domain | What exists | What's missing | Effort |
|--------|-----------|---------------|--------|
| **fraud/anomaly detection on tabular data** | one paper: Bayesian Optimization 1D-CSNN for BAF dataset (EPIA 2024). | no comparison with standard ML baselines (XGBoost, RF) on common fraud datasets. nothing on credit card fraud (Kaggle dataset). | LOW |
