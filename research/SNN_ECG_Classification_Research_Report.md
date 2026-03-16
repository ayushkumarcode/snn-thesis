# Spiking Neural Networks for ECG / Heartbeat Anomaly Detection
## Comprehensive Research Report

**Date:** 2026-02-25
**Research Scope:** SNN-based ECG classification, datasets, performance benchmarks, open-source tools, novelty angles, and undergraduate thesis feasibility.

---

## 1. Executive Summary

Spiking Neural Networks (SNNs) applied to ECG classification and heartbeat anomaly detection represent a rapidly growing but still under-explored research area with significant untapped potential. Between 2020 and 2025, approximately 15-20 peer-reviewed papers have directly addressed SNN-based ECG classification, compared to hundreds using conventional deep learning (CNNs, LSTMs, Transformers). The field is dominated by energy-efficiency motivations -- SNNs consume orders of magnitude less power than traditional DNNs, making them ideal for wearable and edge-deployed cardiac monitors. State-of-the-art SNN accuracy on the MIT-BIH benchmark reaches 98.29% (SparrowSNN, 2024) at 31.39 nanojoules per inference, competitive with CNN baselines (97-99%). However, most SNN-ECG work focuses narrowly on the MIT-BIH dataset with single-lead signals and 5-class AAMI classification. Major gaps exist in 12-lead classification (PTB-XL), spike encoding method comparison, interpretability, and continual/few-shot learning -- all of which represent viable novelty angles for an undergraduate thesis.

The natural fit between ECG signals (temporal, quasi-periodic, spike-like QRS complexes) and SNNs (event-driven, temporal processing) is a core strength of this research direction. ECG R-peaks and QRS complexes map naturally to spike trains, and delta modulation encoding can convert ECG signals into sparse spike representations with minimal information loss.

---

## 2. Has Anyone Done SNN-Based ECG Classification? What Results Did They Get?

### Yes -- this is an active but still emerging field. Key papers and results:

#### Landmark Papers (Chronological)

| Paper / System | Year | Dataset | Classes | Accuracy | Energy | Key Innovation |
|---|---|---|---|---|---|---|
| Energy Efficient ECG (Corradi et al.) | 2020 | MIT-BIH | 5 (AAMI) | ~95% | Low (estimated) | First dedicated SNN-ECG work; delta modulation encoding |
| SNN + Attention (Deng et al.) | 2022 | MIT-BIH | 5 (AAMI) | 98.26% | 346.33 uJ/beat | Channel-wise attentional module in SNN |
| Deep SNN from CNN Conversion (Hu et al.) | 2022 | MIT-BIH | 4 | 84.41% | -- | DNN-to-SNN conversion with ReLU, 14-layer deep SNN |
| SNN + STDP Learning (various) | 2023 | MIT-BIH | 4 | 97.9% | 1.78 uJ/beat | Unsupervised STDP training; real-time inference |
| sCCfC (Spiking ConvLSTM + CfC) | 2024 | PTB-XL / CPSC | Multiple | Competitive | 4.68 uJ/Inf (neuromorphic) vs 450 uJ/Inf (CPU) | On-device edge learning; bio-inspired architecture |
| SparrowSNN (Hardware/Software Co-design) | 2024 | MIT-BIH | 5 (AAMI) | 98.29% | 31.39 nJ/inference | SOTA SNN accuracy; ASIC co-design; minimal timesteps |
| LIF-based ANN-Inspired SNN | 2024 | MIT-BIH | 5 | ~93.8% | -- | LIF neurons within ANN-inspired framework |
| Neuromorphic Arrhythmia Detection (Kolhar) | 2025 | MIT-BIH | Multiple | 94.4% overall | <8ms inference, 1.28M FLOPs, 2.59 MB model | Lightweight for real-time wearable deployment |
| AF Detection on Wearable Edge | 2024 | PhysioNet AF | 2 (AF/Normal) | High (>95%) | Minimal | Feed-forward SNN with custom encoder |

#### Key Observations
- **Accuracy range for SNNs on MIT-BIH**: 84% to 98.29%, depending on architecture and training method
- **Best performing**: SparrowSNN (98.29%) and SNN+Attention (98.26%) are near-SOTA
- **Training approaches**: Three main paradigms -- (1) ANN-to-SNN conversion, (2) surrogate gradient backpropagation, (3) unsupervised STDP
- **ANN-to-SNN conversion** suffers ~1-15% accuracy drop vs. original ANN
- **Surrogate gradient training** (direct SNN training) yields the best results

### Sources
- [SparrowSNN (arXiv 2024)](https://arxiv.org/html/2406.06543)
- [SNN + Attention (MDPI Electronics 2022)](https://www.mdpi.com/2079-9292/11/12/1889)
- [sCCfC On-device Edge Learning (APL Machine Learning 2024)](https://pubs.aip.org/aip/aml/article/2/2/026109/3282738/On-device-edge-learning-for-cardiac-abnormality)
- [Review on SNN-based ECG Classification (Biomedical Engineering Letters 2024)](https://link.springer.com/article/10.1007/s13534-024-00391-2)
- [Neuromorphic Arrhythmia Detection (Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-23248-9)
- [LIF-based SNN Framework (Sensors 2024)](https://www.mdpi.com/1424-8220/24/11/3426)

---

## 3. Available Datasets

### Tier 1: Primary Benchmark Datasets (Most Used in SNN-ECG Research)

| Dataset | Records | Leads | Sampling Rate | Classes | Size | Access |
|---|---|---|---|---|---|---|
| **MIT-BIH Arrhythmia Database** | 48 recordings (47 subjects) | 2-lead | 360 Hz | 5 AAMI classes (N,S,V,F,Q) | ~100 MB | Free on PhysioNet |
| **PTB-XL** | 21,799 ECGs (18,869 patients) | 12-lead | 500 Hz (+ 100 Hz) | 71 SCP-ECG statements, 5 super-classes | ~7.7 GB | Free on PhysioNet |
| **CPSC 2018 (ICBEB)** | 6,877 training + 2,954 test | 12-lead | 500 Hz | 9 classes (1 normal + 8 abnormal) | ~1 GB | Free on PhysioNet |

### Tier 2: Supplementary Datasets

| Dataset | Description | Access |
|---|---|---|
| **Large Scale 12-Lead ECG for Arrhythmia (Chapman-Shaoxing)** | 45,152 patients, 12-lead, 500 Hz | Free on PhysioNet |
| **St Petersburg INCART** | 32 Holter records, 12-lead, annotated | Free on PhysioNet |
| **PhysioNet/CinC Challenge 2020** | Multi-database 12-lead ECG classification | Free on PhysioNet |
| **QTDB** | QT interval annotations, used in some SNN studies | Free on PhysioNet |
| **Icentia11k** | 11,000 patients, single-lead, 7 days continuous | Free (large download) |
| **Kaggle MIT-BIH (CSV format)** | Pre-processed MIT-BIH in accessible CSV format | Free on Kaggle |

### Access Notes
- **All major datasets are freely available** through PhysioNet (https://physionet.org)
- MIT-BIH is the de facto standard for SNN-ECG work (used in ~90% of papers)
- PTB-XL is the gold standard for 12-lead classification but has NOT been used with SNNs (major gap)
- CPSC 2018 is used by some SNN papers (sCCfC) but under-explored
- Kaggle versions of MIT-BIH provide ready-to-use CSV/numpy arrays

### Sources
- [MIT-BIH Arrhythmia Database (PhysioNet)](https://physionet.org/content/mitdb/1.0.0/)
- [PTB-XL Dataset (PhysioNet)](https://physionet.org/content/ptb-xl/1.0.3/)
- [CPSC 2018 Challenge](http://2018.icbeb.org/Challenge.html)
- [PhysioNet Database List](https://physionet.org/about/database/)

---

## 4. SNN Performance vs. Conventional Deep Learning

### Accuracy Comparison (MIT-BIH, 5-class AAMI)

| Method | Architecture | Accuracy | F1 Score | Energy per Inference |
|---|---|---|---|---|
| **CNN (conventional)** | 1D-CNN | 97.4-99.5% | 95-98% | ~450 uJ (CPU) |
| **CNN-LSTM hybrid** | CNN + BiLSTM + Attention | 99.2% | 98.3% | High (GPU) |
| **CNN-LSTM-SE** | CNN + LSTM + Squeeze-Excite | 98.5% | >97% | High (GPU) |
| **SNN (SparrowSNN)** | Co-designed SNN + ASIC | 98.29% | ~97% | **31.39 nJ** |
| **SNN + Attention** | SNN + Channel-wise Attention | 98.26% | 89.09% | 346.33 uJ |
| **SNN (STDP)** | Unsupervised STDP | 97.9% | -- | 1.78 uJ |
| **SNN (ANN-to-SNN)** | Converted 14-layer CNN | 84.41% | -- | Low |
| **SNN (Neuromorphic 2025)** | Lightweight SNN | 94.4% | >88% | 1.28M FLOPs |

### Key Takeaways

1. **Accuracy gap is narrowing**: Best SNNs (98.29%) are within 1% of best CNNs (99.5%) on MIT-BIH
2. **Energy advantage is massive**: SNNs are 100x to 10,000x more energy efficient
   - SparrowSNN: 31.39 nJ vs. CNN on CPU: ~450 uJ (a factor of ~14,000x)
   - SNN on Loihi: ~30 mW vs. LSTM on GPU: ~15W (a factor of 500x)
3. **F1 score gap exists**: SNN F1 scores (89-97%) trail CNNs (95-98%), especially on minority classes (S, F)
4. **Model size advantage**: SNN models are typically 2-10 MB vs. 50-200+ MB for CNN/Transformers
5. **Latency advantage**: SNN inference in <8ms enables true real-time classification
6. **The trade-off is clear**: SNNs sacrifice 1-5% accuracy for 100-10,000x energy savings

### When SNNs Win
- Edge/wearable deployment where power is constrained
- Real-time continuous monitoring (latency matters)
- Battery-powered devices (smartwatches, patches)
- Always-on cardiac monitoring

### When CNNs/Transformers Win
- Server-side batch processing where power is unlimited
- Maximum accuracy is the only priority
- 12-lead ECG analysis (SNNs have not been validated here)

### Sources
- [SparrowSNN Energy Results (arXiv 2024)](https://arxiv.org/html/2406.06543)
- [CNN-LSTM-SE Comparison (MDPI Sensors 2024)](https://www.mdpi.com/1424-8220/24/19/6306)
- [Systematic Review: ECG Arrhythmia Classification (arXiv 2025)](https://arxiv.org/abs/2503.07276)

---

## 5. Natural Fit Between ECG Signals and SNNs

### This is one of the strongest arguments for this research direction.

#### Why ECG and SNNs are a Natural Match

1. **Temporal/Event-Driven Nature**
   - ECG signals are inherently temporal and quasi-periodic
   - QRS complexes are sharp, spike-like events (naturally map to SNN spikes)
   - The R-peak is the dominant "event" in each heartbeat cycle
   - P-waves, T-waves are secondary events with precise timing
   - SNNs process information through precisely timed spikes -- a direct analogy

2. **Sparse Representation**
   - Most of the ECG signal is baseline (isoelectric segments between complexes)
   - Only ~20-30% of each heartbeat cycle contains diagnostically relevant morphology
   - SNNs naturally exploit this sparsity (neurons only fire during events)
   - Delta modulation encoding converts ECG to sparse spike trains efficiently

3. **Temporal Feature Importance**
   - Arrhythmia diagnosis depends on timing: R-R intervals, P-R intervals, QT duration
   - SNNs inherently encode timing information (spike timing)
   - Unlike CNNs which treat time as "just another dimension," SNNs process time natively

4. **Biological Plausibility**
   - The cardiac conduction system itself operates via electrical impulses (spikes)
   - Biological neurons in the brainstem process cardiac signals as spike trains
   - SNNs provide a biologically grounded model for cardiac signal processing

#### Spike Encoding Methods for ECG

| Encoding Method | Description | Accuracy | Robustness | Firing Rate | Best For |
|---|---|---|---|---|---|
| **Rate Encoding** | Maps ECG amplitude to spike frequency | 91.7% | Moderate | High | General purpose |
| **Time-to-First-Spike (TTFS)** | Maps amplitude to spike timing | 89% | Low (noise sensitive) | 2% (very sparse) | Energy-critical apps |
| **Delta Modulation** | Encodes value changes as ON/OFF spikes | ~90% | Best (0.7% drop at 0.1 noise) | Low | Noisy real-world ECG |
| **Peak Encoding** | Uses P/QRS/T peak timing as spike events | Novel approach | -- | Very sparse | Clinical interpretability |
| **Gaussian Encoding** | One value -> time-magnified spike train | -- | -- | Medium | Time series tasks |

#### The Delta Modulation Encoding Deserves Special Attention
- Takes the difference between consecutive ECG samples
- Generates ON spike (positive change > threshold) or OFF spike (negative change > threshold)
- The threshold controls sparsity vs. fidelity trade-off
- Maps naturally to event-driven neuromorphic hardware
- The `snntorch.delta` function implements this directly

### Sources
- [Spike Encoding Evaluation (arXiv 2024)](https://arxiv.org/html/2407.09260v1)
- [snntorch Spike Encoding Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
- [SNN for Biomedical Signal Analysis (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/)
- [SNN for Physiological Signals Review (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362433/)

---

## 6. Open-Source Implementations

### ECG-Specific SNN Repositories

| Repository | Language/Framework | Description | Stars | Status |
|---|---|---|---|---|
| [federicohyo/ecgheartbeat](https://github.com/federicohyo/ecgheartbeat) | Python | ECG to spike conversion using delta modulation; spike train visualization | ~30 | Maintained |
| [alirezaamir/HDL-SpikingNet-ECG](https://github.com/alirezaamir/HDL-SpikingNet-ECG) | Verilog/FPGA | FPGA implementation of SNN for ECG classification inference | ~15 | Research code |
| [byin-cwi/Efficient-spiking-networks](https://github.com/byin-cwi/Efficient-spiking-networks) | PyTorch | Adaptive spiking recurrent networks for ECG (+ SHD, SSC, etc.) | ~50 | Active, pre-trained models included |
| [VELIDIPRADEEPKUMAR/SPIKING-NEURAL-NETWORK](https://github.com/VELIDIPRADEEPKUMAR/SPIKING-NEURAL-NETWORK) | Verilog | AdEx neuron model for ECG spike encoding and arrhythmia detection | ~5 | Research code |

### General SNN Frameworks (Use These to Build Your Own)

| Framework | GitHub | PyTorch-based | Documentation | ECG Support | Best For |
|---|---|---|---|---|---|
| **snnTorch** | [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) | Yes | Excellent (9 tutorials) | Built-in delta encoding | **Recommended for thesis** |
| **SpikingJelly** | [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) | Yes | Good (Chinese + English) | Via publications | Large-scale SNN research |
| **Norse** | [norse/norse](https://github.com/norse/norse) | Yes | Good | Not direct | Bio-plausible models |
| **BindsNET** | [BindsNET/bindsnet](https://github.com/BindsNET/bindsnet) | Yes | Good | Not direct | STDP/unsupervised learning |

### Why snnTorch is Recommended for an Undergraduate Thesis
1. Best-in-class tutorials (9 comprehensive tutorials covering encoding, training, deployment)
2. PyTorch integration means familiar development workflow
3. Built-in `snntorch.delta` for ECG-to-spike encoding
4. Active community and recent updates
5. Surrogate gradient training out of the box
6. Regression tutorials applicable to time-series tasks

### Sources
- [snnTorch Documentation](https://snntorch.readthedocs.io/en/latest/)
- [SpikingJelly Publications](https://github.com/fangwei123456/spikingjelly/blob/master/publications.md)
- [Efficient Spiking Networks Repository](https://github.com/byin-cwi/Efficient-spiking-networks)

---

## 7. Feasibility for an Undergraduate Thesis

### Assessment: HIGHLY FEASIBLE -- this is an excellent undergraduate thesis topic.

#### Why This Works for Undergrad Level

**Strengths:**
1. **Well-defined problem**: ECG classification is a standard, well-benchmarked task
2. **Accessible datasets**: MIT-BIH is small (~100 MB), well-documented, freely available, and pre-processed versions exist on Kaggle
3. **Mature tooling**: snnTorch provides a PyTorch-based framework with excellent tutorials
4. **Clear evaluation**: Standard metrics (accuracy, F1, sensitivity, specificity) and AAMI classification standards
5. **Reproducible baselines**: Multiple CNN baselines exist for comparison
6. **Manageable scope**: A single-lead, 5-class classification task is tractable
7. **Strong narrative**: Energy-efficient cardiac monitoring for wearables is compelling and timely
8. **Publication potential**: Under-explored enough that novel contributions are achievable

**Challenges (Manageable):**
1. SNN training is less mature than CNN training (more hyperparameter tuning needed)
2. Surrogate gradient methods require understanding (but snnTorch abstracts most complexity)
3. Limited existing code specifically for ECG (will need to adapt general SNN code)
4. Reproducing exact results from papers can be tricky (preprocessing details often missing)

#### Suggested Thesis Architecture

```
Phase 1 (Weeks 1-3): Foundation
  - Literature review (this report provides the foundation)
  - Set up Python environment (PyTorch + snnTorch)
  - Download and preprocess MIT-BIH dataset
  - Complete snnTorch tutorials 1-5

Phase 2 (Weeks 4-6): Baseline Implementation
  - Implement CNN baseline for ECG classification (1D-CNN)
  - Implement basic SNN using snnTorch (LIF neurons, surrogate gradient)
  - Implement ECG-to-spike encoding (delta modulation)
  - Evaluate on MIT-BIH with AAMI 5-class split

Phase 3 (Weeks 7-10): Novel Contribution
  - Choose novelty angle (see Section 8 below)
  - Implement and experiment with chosen angle
  - Compare SNN vs CNN on accuracy AND energy metrics
  - Iterate on architecture/encoding

Phase 4 (Weeks 11-14): Analysis and Writing
  - Comprehensive evaluation and comparison tables
  - Ablation studies (encoding methods, neuron types, network depth)
  - Write thesis document
  - Create visualizations (spike raster plots, confusion matrices, energy charts)
```

#### Minimum Viable Thesis
At minimum, an undergrad thesis could:
1. Implement an SNN for MIT-BIH 5-class classification using snnTorch
2. Compare it against a CNN baseline
3. Analyze accuracy vs. energy trade-off
4. Experiment with 2-3 spike encoding methods
5. Provide spike raster visualizations showing how ECG maps to spikes

This alone would be a solid contribution given the limited existing implementations.

---

## 8. Novelty Angles -- What Makes This Under-Explored?

### The Field is Under-Explored in Several Critical Dimensions

#### Gap Analysis

| Research Area | Explored? | Number of Papers | Opportunity Level |
|---|---|---|---|
| SNN + MIT-BIH (5-class, single-lead) | Moderately | ~15-20 | LOW -- well-covered |
| SNN + PTB-XL (12-lead, multi-label) | Barely | 1-2 (sCCfC only) | **VERY HIGH** |
| Spike encoding method comparison for ECG | Partially | ~3-5 | **HIGH** |
| SNN + Explainability/Interpretability for ECG | Almost none | ~1-2 | **VERY HIGH** |
| SNN + AF-specific detection | Emerging | ~3-4 | MODERATE |
| SNN + Continual/Online learning for ECG | None found | 0 | **VERY HIGH** |
| SNN + Few-shot learning for rare arrhythmias | None found | 0 | **VERY HIGH** |
| SNN energy benchmarking across hardware | Limited | ~3-5 | HIGH |
| SNN + ECG on actual neuromorphic hardware (Loihi) | Very limited | ~2-3 | HIGH (but needs hardware) |

### Top 5 Recommended Novelty Angles (Ranked by Feasibility for Undergrad)

#### 1. RECOMMENDED: Systematic Spike Encoding Comparison for ECG Classification
**What**: Compare all major encoding methods (rate, TTFS, delta modulation, peak encoding, Gaussian) on the same SNN architecture and MIT-BIH dataset.
**Why novel**: No single paper systematically compares all encoding methods specifically for ECG. The 2024 paper by arxiv:2407.09260 does a general comparison but not ECG-focused.
**Feasibility**: HIGH -- uses existing snnTorch encoding functions; clear methodology.
**Impact**: Would become a reference paper for anyone doing SNN-ECG work.

#### 2. RECOMMENDED: SNN for 12-Lead ECG Classification (PTB-XL)
**What**: First systematic application of SNNs to the PTB-XL 12-lead dataset with multi-label classification.
**Why novel**: Almost all SNN-ECG papers use MIT-BIH (single-lead, 5 classes). PTB-XL is the modern gold-standard but is virtually untouched by SNN research.
**Feasibility**: MODERATE-HIGH -- larger dataset and multi-label adds complexity, but the framework is the same.
**Impact**: HIGH -- fills a glaring gap in the literature.

#### 3. SNN Interpretability via Spike Pattern Analysis
**What**: Analyze what the SNN "sees" by visualizing spike patterns corresponding to different arrhythmia classes. Map spike timing to clinical ECG features (P-wave, QRS, T-wave).
**Why novel**: SNN interpretability for ECG is almost completely unexplored. Spike timing patterns could map to clinically meaningful ECG intervals.
**Feasibility**: MODERATE -- requires creativity in visualization and analysis.
**Impact**: HIGH -- bridges gap between computational neuroscience and clinical cardiology.

#### 4. Energy-Accuracy Pareto Analysis Across SNN Architectures
**What**: Systematically map the energy-accuracy trade-off curve for multiple SNN architectures (feed-forward, recurrent, convolutional) with different encoding methods.
**Why novel**: Most papers report a single accuracy/energy point. No comprehensive Pareto frontier analysis exists.
**Feasibility**: HIGH -- straightforward to implement multiple architectures and measure.
**Impact**: MODERATE-HIGH -- practical guidance for deployment decisions.

#### 5. Continual Learning SNN for ECG (Ambitious)
**What**: Train an SNN that can learn new arrhythmia classes over time without forgetting old ones, leveraging STDP's natural online learning capabilities.
**Why novel**: Zero papers found combining SNN continual learning with ECG. STDP is inherently suited to online learning.
**Feasibility**: MODERATE -- requires understanding of continual learning; more research-heavy.
**Impact**: VERY HIGH -- would be genuinely novel.

### Combined Approach (Recommended for Maximum Impact)
A strong thesis could combine angles 1 + 4:
- **Title**: "Energy-Efficient ECG Classification with Spiking Neural Networks: A Systematic Study of Encoding Methods and Architecture Trade-offs"
- This provides: (a) clear methodology, (b) multiple experiments, (c) practical conclusions, (d) fills a gap in the literature.

---

## 9. Research Gaps and Limitations

### What I Could Not Find
1. **Exact code for SparrowSNN** -- the paper does not appear to have a public GitHub repository
2. **Direct head-to-head SNN vs. Transformer comparisons** on ECG -- this comparison does not appear to exist yet
3. **SNN-ECG papers on arXiv from 2026** -- the field may have new publications I could not access
4. **Neuromorphic hardware benchmark suites** specifically for ECG (Loihi/TrueNorth results are scattered)
5. **Patient-specific SNN adaptation** studies -- adapting a trained SNN to individual patient ECG morphology

### Information Confidence Levels
- **HIGH confidence**: Dataset accessibility, SNN-ECG paper existence, accuracy numbers from published papers
- **MODERATE confidence**: Energy consumption comparisons (different measurement methodologies across papers)
- **LOW confidence**: Exact count of all SNN-ECG papers (some may exist in non-indexed venues)

---

## 10. Recommended Follow-Up Actions

1. **Read the 2024 review paper** in full: "Review on spiking neural network-based ECG classification methods for low-power environments" (Biomedical Engineering Letters, 2024) -- this is the most comprehensive existing survey
2. **Read the 2025 systematic review**: arXiv:2503.07276 for the latest benchmarking standards
3. **Install snnTorch** and work through tutorials 1-5 (spike encoding, LIF neurons, training)
4. **Download MIT-BIH from Kaggle** (CSV format) for quick prototyping
5. **Replicate federicohyo/ecgheartbeat** to understand ECG-to-spike conversion
6. **Replicate byin-cwi/Efficient-spiking-networks** ECG folder for a working SNN baseline
7. **Read the SparrowSNN paper** (arXiv:2406.06543) for the current SOTA architecture details
8. **Discuss with supervisor**: Which novelty angle aligns with thesis requirements and timeline

---

## 11. Key References (Organized by Relevance)

### Must-Read Papers
1. Review on SNN-based ECG classification (2024): https://link.springer.com/article/10.1007/s13534-024-00391-2
2. SparrowSNN (2024): https://arxiv.org/html/2406.06543
3. Systematic Review of ECG Arrhythmia Classification (2025): https://arxiv.org/abs/2503.07276
4. Neuromorphic Arrhythmia Detection (2025): https://www.nature.com/articles/s41598-025-23248-9
5. SNN + Attention for ECG (2022): https://www.mdpi.com/2079-9292/11/12/1889
6. sCCfC On-device Edge Learning (2024): https://pubs.aip.org/aip/aml/article/2/2/026109/3282738
7. Spike Encoding Evaluation (2024): https://arxiv.org/html/2407.09260v1

### Background and Context
8. SNN for Biomedical Signals (2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/
9. SNN Potential in Biomedical Applications (2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11362408/
10. Energy Efficient ECG with SNN (2020): https://www.sciencedirect.com/science/article/abs/pii/S1746809420303098

### Tools and Frameworks
11. snnTorch: https://snntorch.readthedocs.io/en/latest/
12. SpikingJelly: https://github.com/fangwei123456/spikingjelly
13. Norse: https://github.com/norse/norse

### Datasets
14. MIT-BIH: https://physionet.org/content/mitdb/1.0.0/
15. PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
16. CPSC 2018: http://2018.icbeb.org/Challenge.html

---

*Report generated through systematic multi-platform research across Google Scholar, arXiv, PubMed/PMC, GitHub, PhysioNet, IEEE Xplore, MDPI, Springer, Nature, and ResearchGate.*
