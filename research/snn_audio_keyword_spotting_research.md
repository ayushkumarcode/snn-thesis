# SNNs for Audio Processing: Keyword Spotting & Speech Command Recognition

Looking into whether SNN-based keyword spotting / speech command recognition could work as a thesis topic. Short answer: yes, this has gotten pretty mature in 2024-2025. The accuracy gap between SNNs and ANNs has narrowed a lot -- state-of-the-art SNNs now hit 96.9% on Google Speech Commands V2 (35-class), which is close to the ANN ceiling of ~97-98%. There are multiple open-source frameworks (snnTorch, SpikingJelly, sparch) with good documentation, and several complete implementations on GitHub in 300-600 lines of core Python. The energy efficiency argument holds up with hardware benchmarks showing 10-200x lower energy per inference on neuromorphic hardware (Intel Loihi) vs conventional processors.

---

## SNN vs ANN Accuracy on Google Speech Commands

### Current state of the art (early 2025)

| Model | Type | Dataset (Task) | Accuracy | Parameters | Year | Code? |
|-------|------|----------------|----------|------------|------|-------|
| **SpikCommander** | SNN (Spiking Transformer) | GSC V2 (35-class) | **96.92%** | 2.13M | 2025 | Yes |
| **SpikCommander** | SNN (Spiking Transformer) | GSC V2 (35-class) | **97.08%** (T=200) | 2.13M | 2025 | Yes |
| **SpikeSCR** | SNN (Hybrid Attention) | GSC V2 (35-class) | **95.60%** | ~3.3M | 2024 | Pending |
| **SIDC-KWS** | SNN (Conformer) | GSC V2 (12-class) | **96.8%** | -- | 2025 | -- |
| **Spiking LMUFormer** | SNN | GSC V2 (35-class) | **96.12%** | -- | 2024 | -- |
| **RadLIF (sparch)** | SNN (Recurrent) | GSC V2 (35-class) | **96.60%** | ~1M | 2022 | Yes |
| **adLIF (sparch)** | SNN (Non-recurrent) | GSC V2 (35-class) | **95.50%** | ~1M | 2022 | Yes |
| **LSNN** | SNN (Spiking RNN) | GSC V1 (12-class) | **91.2%** | -- | 2020 | Yes |
| **ED-sKWS** | SNN (Early Decision) | GSC V2 (35-class) | **93.04%** | 27.6K | 2024 | No |
| LMUFormer | ANN | GSC V2 (35-class) | 96.53% | -- | 2024 | -- |
| Attention-RNN | ANN | GSC V2 (20-class) | 94.5% | 202K | 2019 | -- |
| LSTM | ANN (Baseline) | GSC V1 (12-class) | 94.4% | -- | 2020 | Yes |
| CNN (Baseline) | ANN | GSC V1 (12-class) | 87.6% | -- | 2020 | Yes |

The gap is basically closed. In 2020, best SNN (LSNN at 91.2%) trailed best ANN (LSTM at 94.4%) by ~3.2 points on GSC 12-class. By 2025, SpikCommander gets 96.92% (35-class), beating many ANN baselines. On the 12-class task, SNNs routinely hit 95-97%, matching or exceeding ANNs. On the harder 35-class task, best SNNs get ~96.9%, within 1-2 points of ANN ceiling.

Parameter efficiency is interesting too: SpikCommander gets 96.71% with only 1.12M params. ED-sKWS gets 93% with just 27.6K parameters -- orders of magnitude fewer than typical ANNs.

### SHD (Spiking Heidelberg Digits) Benchmark

| Model | Type | SHD Accuracy | Parameters | Year |
|-------|------|-------------|------------|------|
| **SpikCommander** | SNN | **96.41%** | 0.19M | 2025 |
| **SpikeSCR** | SNN | **95.70%** | -- | 2024 |
| **SE-adLIF** | SNN | **95.81%** | 0.45M | 2024 |
| **RadLIF (sparch)** | SNN | **97.60%** | ~1M | 2022 |
| **adLIF (sparch)** | SNN | **97.40%** | ~1M | 2022 |
| Hardware deployment | SNN | **93.4%** | -- | 2024 |

### SSC (Spiking Speech Commands) Benchmark

| Model | Type | SSC Accuracy | Parameters | Year |
|-------|------|-------------|------------|------|
| **SpikCommander** | SNN | **83.49%** | 2.13M | 2025 |
| **SpikeSCR** | SNN | **82.79%** | -- | 2024 |
| **RadLIF (sparch)** | SNN | **93.40%** | ~1M | 2022 |
| CNN (Cramer et al.) | ANN | 77.7% | -- | 2020 |
| GRU | ANN | 79.05% | -- | 2020 |

---

## Frameworks and Tools

### Framework Comparison
