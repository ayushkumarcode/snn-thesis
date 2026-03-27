# Spiking Neural Networks for ECG / Heartbeat Anomaly Detection

Looking into whether SNNs for ECG classification could work as a thesis topic. Covering what exists, datasets, performance vs conventional DL, open-source tools, novelty angles, and whether it's actually feasible for an undergrad project.

---

## Has Anyone Done This Before?

Yeah, it's an active but still emerging field. Between 2020 and 2025, roughly 15-20 peer-reviewed papers have directly addressed SNN-based ECG classification. That's not a lot compared to the hundreds using conventional deep learning (CNNs, LSTMs, Transformers). Most of the work is motivated by energy efficiency -- SNNs consume orders of magnitude less power than traditional DNNs, which is great for wearable and edge-deployed cardiac monitors.

Best SNN accuracy on MIT-BIH reaches 98.29% (SparrowSNN, 2024) at 31.39 nanojoules per inference, which is competitive with CNN baselines (97-99%). But most SNN-ECG work focuses narrowly on MIT-BIH with single-lead signals and 5-class AAMI classification. Big gaps exist in 12-lead classification (PTB-XL), spike encoding comparison, interpretability, and continual/few-shot learning -- all potentially viable novelty angles.

The natural fit here is actually a strong argument. ECG signals are temporal, quasi-periodic, and have spike-like QRS complexes. R-peaks and QRS complexes map naturally to spike trains, and delta modulation encoding can convert ECG signals into sparse spike representations with minimal information loss.

### Key papers and results:

| Paper / System | Year | Dataset | Classes | Accuracy | Energy | Key Innovation |
|---|---|---|---|---|---|---|
| Energy Efficient ECG (Corradi et al.) | 2020 | MIT-BIH | 5 (AAMI) | ~95% | Low (estimated) | First dedicated SNN-ECG work; delta modulation encoding |
| SNN + Attention (Deng et al.) | 2022 | MIT-BIH | 5 (AAMI) | 98.26% | 346.33 uJ/beat | Channel-wise attentional module in SNN |
| Deep SNN from CNN Conversion (Hu et al.) | 2022 | MIT-BIH | 4 | 84.41% | -- | DNN-to-SNN conversion with ReLU, 14-layer deep SNN |
| SNN + STDP Learning (various) | 2023 | MIT-BIH | 4 | 97.9% | 1.78 uJ/beat | Unsupervised STDP training; real-time inference |
| sCCfC (Spiking ConvLSTM + CfC) | 2024 | PTB-XL / CPSC | Multiple | Competitive | 4.68 uJ/Inf (neuromorphic) vs 450 uJ/Inf (CPU) | On-device edge learning |
| SparrowSNN | 2024 | MIT-BIH | 5 (AAMI) | 98.29% | 31.39 nJ/inference | SOTA SNN accuracy; ASIC co-design |
| LIF-based ANN-Inspired SNN | 2024 | MIT-BIH | 5 | ~93.8% | -- | LIF neurons within ANN-inspired framework |
| Neuromorphic Arrhythmia Detection (Kolhar) | 2025 | MIT-BIH | Multiple | 94.4% overall | <8ms inference, 1.28M FLOPs, 2.59 MB model | Lightweight for wearable deployment |
| AF Detection on Wearable Edge | 2024 | PhysioNet AF | 2 (AF/Normal) | High (>95%) | Minimal | Feed-forward SNN with custom encoder |

Things to notice:
- Accuracy range for SNNs on MIT-BIH: 84% to 98.29%, depending on architecture and training
- Three main training paradigms: (1) ANN-to-SNN conversion, (2) surrogate gradient backprop, (3) unsupervised STDP
- ANN-to-SNN conversion suffers ~1-15% accuracy drop vs original ANN
- Surrogate gradient training (direct SNN training) gives the best results

Sources:
- SparrowSNN (arXiv 2024): https://arxiv.org/html/2406.06543
- SNN + Attention (MDPI Electronics 2022): https://www.mdpi.com/2079-9292/11/12/1889
- sCCfC On-device Edge Learning (APL Machine Learning 2024): https://pubs.aip.org/aip/aml/article/2/2/026109/3282738
- Review on SNN-based ECG Classification (Biomedical Engineering Letters 2024): https://link.springer.com/article/10.1007/s13534-024-00391-2
- Neuromorphic Arrhythmia Detection (Scientific Reports 2025): https://www.nature.com/articles/s41598-025-23248-9
- LIF-based SNN Framework (Sensors 2024): https://www.mdpi.com/1424-8220/24/11/3426

---

## Available Datasets

### Tier 1: Primary Benchmarks (most used in SNN-ECG research)

| Dataset | Records | Leads | Sampling Rate | Classes | Size | Access |
|---|---|---|---|---|---|---|
| **MIT-BIH Arrhythmia Database** | 48 recordings (47 subjects) | 2-lead | 360 Hz | 5 AAMI classes (N,S,V,F,Q) | ~100 MB | Free on PhysioNet |
| **PTB-XL** | 21,799 ECGs (18,869 patients) | 12-lead | 500 Hz (+ 100 Hz) | 71 SCP-ECG statements, 5 super-classes | ~7.7 GB | Free on PhysioNet |
| **CPSC 2018 (ICBEB)** | 6,877 training + 2,954 test | 12-lead | 500 Hz | 9 classes (1 normal + 8 abnormal) | ~1 GB | Free on PhysioNet |

### Tier 2: Supplementary

| Dataset | Description | Access |
|---|---|---|
| **Chapman-Shaoxing** | 45,152 patients, 12-lead, 500 Hz | Free on PhysioNet |
| **St Petersburg INCART** | 32 Holter records, 12-lead, annotated | Free on PhysioNet |
| **PhysioNet/CinC Challenge 2020** | Multi-database 12-lead ECG classification | Free on PhysioNet |
| **QTDB** | QT interval annotations, used in some SNN studies | Free on PhysioNet |
| **Icentia11k** | 11,000 patients, single-lead, 7 days continuous | Free (large download) |
| **Kaggle MIT-BIH (CSV format)** | Pre-processed MIT-BIH in accessible CSV format | Free on Kaggle |

All major datasets are freely available through PhysioNet (https://physionet.org). MIT-BIH is the de facto standard for SNN-ECG work (~90% of papers use it). PTB-XL is the gold standard for 12-lead but has NOT been used with SNNs -- that's a major gap. Kaggle versions of MIT-BIH give you ready-to-use CSV/numpy arrays.

---

## SNN Performance vs Conventional Deep Learning

### Accuracy Comparison (MIT-BIH, 5-class AAMI)

| Method | Architecture | Accuracy | F1 Score | Energy per Inference |
|---|---|---|---|---|
| CNN (conventional) | 1D-CNN | 97.4-99.5% | 95-98% | ~450 uJ (CPU) |
| CNN-LSTM hybrid | CNN + BiLSTM + Attention | 99.2% | 98.3% | High (GPU) |
| CNN-LSTM-SE | CNN + LSTM + Squeeze-Excite | 98.5% | >97% | High (GPU) |
| SNN (SparrowSNN) | Co-designed SNN + ASIC | 98.29% | ~97% | **31.39 nJ** |
| SNN + Attention | SNN + Channel-wise Attention | 98.26% | 89.09% | 346.33 uJ |
| SNN (STDP) | Unsupervised STDP | 97.9% | -- | 1.78 uJ |
| SNN (ANN-to-SNN) | Converted 14-layer CNN | 84.41% | -- | Low |
| SNN (Neuromorphic 2025) | Lightweight SNN | 94.4% | >88% | 1.28M FLOPs |

### What this tells us

1. **Accuracy gap is narrowing**: best SNNs (98.29%) are within 1% of best CNNs (99.5%) on MIT-BIH
2. **Energy advantage is massive**: SNNs are 100x to 10,000x more efficient
   - SparrowSNN: 31.39 nJ vs CNN on CPU: ~450 uJ (factor of ~14,000x)
   - SNN on Loihi: ~30 mW vs LSTM on GPU: ~15W (factor of 500x)
3. **F1 score gap exists**: SNN F1 scores (89-97%) trail CNNs (95-98%), especially on minority classes (S, F)
4. **Model size advantage**: SNNs typically 2-10 MB vs 50-200+ MB for CNN/Transformers
5. **Latency advantage**: SNN inference in <8ms enables true real-time classification
6. **The trade-off**: SNNs sacrifice 1-5% accuracy for 100-10,000x energy savings

SNNs win for edge/wearable deployment, real-time continuous monitoring, battery-powered devices, always-on cardiac monitoring. CNNs/Transformers win for server-side batch processing, when max accuracy is the only priority, and 12-lead analysis (SNNs haven't been validated there yet).

---

## Why ECG and SNNs are a Natural Match

This is actually one of the strongest arguments for this research direction.

1. **Temporal/Event-Driven Nature**
   - ECG signals are inherently temporal and quasi-periodic
   - QRS complexes are sharp, spike-like events that naturally map to SNN spikes
   - The R-peak is the dominant "event" in each heartbeat cycle
   - P-waves, T-waves are secondary events with precise timing
   - SNNs process information through precisely timed spikes -- direct analogy

2. **Sparse Representation**
   - Most of the ECG signal is baseline (isoelectric segments)
   - Only ~20-30% of each heartbeat cycle has diagnostically relevant morphology
   - SNNs naturally exploit this sparsity (neurons only fire during events)
   - Delta modulation encoding converts ECG to sparse spike trains efficiently

3. **Temporal Feature Importance**
   - Arrhythmia diagnosis depends on timing: R-R intervals, P-R intervals, QT duration
   - SNNs inherently encode timing info through spike timing
   - Unlike CNNs which treat time as "just another dimension," SNNs process time natively

4. **Biological Plausibility**
   - The cardiac conduction system itself runs on electrical impulses (spikes)
   - Biological neurons in the brainstem process cardiac signals as spike trains
   - SNNs provide a biologically grounded model for cardiac signal processing

### Spike Encoding Methods for ECG

| Encoding Method | Description | Accuracy | Robustness | Firing Rate | Best For |
|---|---|---|---|---|---|
| **Rate Encoding** | Maps ECG amplitude to spike frequency | 91.7% | Moderate | High | General purpose |
| **Time-to-First-Spike (TTFS)** | Maps amplitude to spike timing | 89% | Low (noise sensitive) | 2% (very sparse) | Energy-critical apps |
| **Delta Modulation** | Encodes value changes as ON/OFF spikes | ~90% | Best (0.7% drop at 0.1 noise) | Low | Noisy real-world ECG |
| **Peak Encoding** | Uses P/QRS/T peak timing as spike events | Novel approach | -- | Very sparse | Clinical interpretability |
| **Gaussian Encoding** | One value -> time-magnified spike train | -- | -- | Medium | Time series tasks |

Delta modulation deserves special attention. It takes the difference between consecutive ECG samples and generates ON spike (positive change > threshold) or OFF spike (negative change > threshold). The threshold controls the sparsity vs fidelity trade-off. Maps naturally to event-driven neuromorphic hardware. `snntorch.delta` implements this directly.

---

## Open-Source Implementations

### ECG-Specific SNN Repos

| Repository | Language/Framework | Description | Stars |
|---|---|---|---|
| [federicohyo/ecgheartbeat](https://github.com/federicohyo/ecgheartbeat) | Python | ECG to spike conversion using delta modulation; spike train visualization | ~30 |
| [alirezaamir/HDL-SpikingNet-ECG](https://github.com/alirezaamir/HDL-SpikingNet-ECG) | Verilog/FPGA | FPGA implementation of SNN for ECG inference | ~15 |
| [byin-cwi/Efficient-spiking-networks](https://github.com/byin-cwi/Efficient-spiking-networks) | PyTorch | Adaptive spiking recurrent networks for ECG (+ SHD, SSC) | ~50 |
| [VELIDIPRADEEPKUMAR/SPIKING-NEURAL-NETWORK](https://github.com/VELIDIPRADEEPKUMAR/SPIKING-NEURAL-NETWORK) | Verilog | AdEx neuron model for ECG spike encoding and arrhythmia detection | ~5 |

### General SNN Frameworks

| Framework | GitHub | ECG Support | Best For |
|---|---|---|---|
| **snnTorch** | [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) | Built-in delta encoding | **Recommended for thesis** |
| **SpikingJelly** | [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) | Via publications | Large-scale SNN research |
| **Norse** | [norse/norse](https://github.com/norse/norse) | Not direct | Bio-plausible models |
| **BindsNET** | [BindsNET/bindsnet](https://github.com/BindsNET/bindsnet) | Not direct | STDP/unsupervised learning |

snnTorch seems like the best option for what i'd need: best tutorials (9 covering encoding, training, deployment), PyTorch integration, built-in `snntorch.delta` for ECG-to-spike encoding, active community, surrogate gradient training out of the box.

---

## Feasibility for an Undergrad Thesis

i think this would work really well. Here's why:

**Strengths:**
1. Well-defined problem -- ECG classification is standard and well-benchmarked
2. Accessible datasets -- MIT-BIH is small (~100 MB), well-documented, freely available, pre-processed versions on Kaggle
3. Mature tooling -- snnTorch provides everything needed with good tutorials
4. Clear evaluation -- standard metrics (accuracy, F1, sensitivity, specificity) and AAMI standards
5. Reproducible baselines -- multiple CNN baselines exist for comparison
6. Manageable scope -- single-lead, 5-class classification is tractable
7. Strong narrative -- energy-efficient cardiac monitoring for wearables is compelling
8. Publication potential -- under-explored enough that novel contributions are achievable

**Challenges (but manageable):**
1. SNN training is less mature than CNN training (more hyperparameter tuning)
2. Surrogate gradient methods require understanding (but snnTorch abstracts most complexity)
3. Limited existing code specifically for ECG (need to adapt general SNN code)
4. Reproducing exact results from papers is tricky (preprocessing details often missing)

### Suggested timeline

```
Phase 1 (Weeks 1-3): Foundation
  - Literature review
  - Set up Python environment (PyTorch + snnTorch)
  - Download and preprocess MIT-BIH
  - Do snnTorch tutorials 1-5

Phase 2 (Weeks 4-6): Baseline Implementation
  - Implement CNN baseline for ECG (1D-CNN)
  - Implement basic SNN with snnTorch (LIF neurons, surrogate gradient)
