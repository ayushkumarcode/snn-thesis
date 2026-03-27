# SNNs for Audio and Environmental Sound Classification -- What Exists (2024-2026)

went through a ton of papers on SNNs for audio, covering 2024-2026 with some older context. searched arXiv, IEEE Xplore, NeurIPS proceedings, OpenReview, Semantic Scholar, Google Scholar, ACM DL, Frontiers, Nature, and university repositories.

**the big takeaway: nobody has applied a full SNN to the complete ESC-50 (50-class) dataset before us.** closest is Larroza et al. (2025, arXiv:2503.11206) who used a 4-layer FC-only SNN on ESC-10 (10-class subset) and got 69.0% F1-score with their best encoding (TAE). our thesis (47.15% on full ESC-50 with convolutional SNN, 92.50% with PANNs+SNN head) is genuinely first.

the field is moving fast though -- spiking Transformers for speech commands are getting impressive (SpikeSCR: 95.70% SHD; SpikCommander: 96.71% GSC), multimodal audio-visual SNNs are hitting 98%+ on UrbanSound8K-AV, and neuromorphic hardware deployment is getting real (SpiNNaker2 91.12%, Loihi 2 with 200x energy reduction). but environmental sound classification specifically? barely explored.

---

## 2. Papers Applying SNNs to Audio/Sound Classification

### Environmental Sound Classification (most relevant to us)

#### Larroza et al. (2025) -- the closest competitor
- "Spike Encoding for Environmental Sound: A Comparative Benchmark"
- arXiv:2503.11206v3 (submitted to ICASSP 2026)
- Datasets: ESC-10, UrbanSound8K, TAU Urban Acoustic Scenes (3-class). **NOT tested on ESC-50**
- Architecture: 4 fully-connected layers, 128 LIF neurons per hidden layer, snnTorch v0.9.1. **no convolutional layers**
- Training: 100 epochs, batch 32, LR 0.01
- Encodings compared: Threshold Adaptive Encoding (TAE), Step Forward (SF), Moving Window (MW)

| Encoder | ESC-10 | UrbanSound8K | TAU-3Class |
|---------|--------|--------------|------------|
| TAE     | 0.690  | 0.535        | 0.690      |
| SF      | 0.598  | 0.564        | 0.640      |
| MW      | 0.620  | 0.530        | 0.550      |
| **Baseline (non-spiking)** | **0.727** | **0.730** | **0.873** |

they explicitly state: "To our knowledge, no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods and performed classification with a spiking neural network (SNN)."

their best ESC-10 result (69.0% with TAE) uses only FC layers and only 10 classes. our ConvSNN gets 47.15% on the full ESC-50 (50 classes) -- fundamentally harder task. spike firing rates: TAE has 38.44% (ESC-10), lowest = most energy efficient.

#### Guo et al. (2024) -- Multimodal Audio-Visual SNN
- "Transformer-Based Spiking Neural Networks for Multimodal Audiovisual Classification"
- IEEE TCDS, Vol. 16(3), June 2024
- UrbanSound8K-AV Accuracy: 96.85% (timesteps=4)
- but this is MULTIMODAL (audio+visual), not audio-only. not directly comparable.

#### S-CMRL (2025) -- Semantic-Alignment Audio-Visual SNN
- arXiv:2502.12488, February 2025
- Transformer-based multimodal SNN with cross-modal residual learning

| Dataset | S-CMRL | CMCI | SMMT (Guo) | WeightAttention |
|---------|--------|------|------------|-----------------|
| UrbanSound8K-AV | **98.13%** | 97.90% | 96.85% | 97.60% |
| CREMA-D | **73.25%** | 70.02% | -- | 64.78% |

again multimodal, not audio-only.

### Speech Command Recognition (Keyword Spotting)

#### SpikeSCR (Wang et al., 2024)
- "Efficient Speech Command Recognition Leveraging SNN and Curriculum Learning-based Knowledge Distillation"
- arXiv:2412.12858, published in Neural Networks 2025
- fully spike-driven, Global-Local Hybrid Encoder (Spiking Self-Attention + Separable Gated Convolution)

| Dataset | SpikeSCR (100 steps) | With KDCL (40 steps) | Previous SOTA |
|---------|---------------------|---------------------|---------------|
| SHD (20 classes) | **95.70%** | 93.60% | 95.07% (DCLS) |
| SSC (35 classes) | **82.79%** | 80.25% | 80.69% (DCLS) |
| GSC v2 (35 classes) | **95.60%** | 95.01% | 95.35% (DCLS) |

KDCL reduces timesteps by 60%, energy by 54.8% (0.0314mJ to 0.0142mJ on SSC). uses AC=0.9pJ, MAC=4.6pJ energy model -- same as what we use for NeuroBench.

#### SpikCommander (2025/2026)
- arXiv:2511.07883, January 2026
- Multi-view Spiking Spatio-Temporal Attention + Spiking Contextual Refinement MLP

| Dataset | SpikCommander | SpikeSCR | DCLS |
|---------|--------------|----------|------|
| SHD     | **96.41%**   | 95.70%   | 95.07% |
| SSC     | **83.26%**   | 82.79%   | 80.69% |
| GSC v2  | **96.71%**   | 95.60%   | 95.35% |

0.19M params (SHD), 1.12M (SSC/GSC). current SOTA for SNN speech command recognition.

#### SIDC-KWS (Interspeech 2025)
- Spiking Inception-Dilated Conformer
- GSC v2 12-class: 96.8%
- 75.59% less energy than ANN counterpart

#### E-prop on SpiNNaker 2 (Yan et al., 2022)
- Frontiers in Neuroscience, 2022
- spiking RNN with e-prop learning rule, trained ONLINE on SpiNNaker 2
- 91.12% on Google Speech Commands
- 680 KB for 25K weights, 12x less energy than V100 GPU

### Speech Enhancement

#### Spiking-FullSubNet (Hao et al., 2024)
- arXiv:2410.04785, IEEE TNNLS 2025
- **1st Place Winner, Intel N-DNS Challenge (Track 1: Algorithmic)**
- full-band + sub-band SNN with Gated Spiking Neurons
- nearly 3 orders of magnitude less energy than best ANN (CMGAN)
- first SNN to win a major speech processing competition

#### Three-Stage Hybrid SNN Fine-Tuning (Abuhajar et al., 2025)
- Frontiers in Neuroscience, April 2025
- ANN train -> ANN-to-SNN conversion -> hybrid fine-tuning (spiking forward, ANN backward)
- spiking Wave-U-Net and spiking Conv-TasNet
- operates in temporal domain (no FFT needed)

### Sound Source Localization

#### RF-PLC SSL (Zhang et al., NeurIPS 2024)
- "Spike-based Neuromorphic Model for Sound Source Localization"
- NeurIPS 2024 Poster
- Resonate-and-Fire neurons with Phase-Locking Coding + Multi-Auditory Attention
- SOTA accuracy in SSL tasks, strong noise robustness
- first SNN at NeurIPS for audio processing

#### Hilbert Transform SNN Localization (Haghighatshoar & Muir, 2025)
- Communications Engineering (Nature), 2025
- Hilbert transform avoids dense band-pass filters; event-based encoding captures analytic signal phase
- MAE: 0.25-0.65 degrees (1.6-2.6 kHz)
- deployed on SynSense Xylo
- GitHub: https://github.com/synsense/HaghighatshoarMuir2024

### Audio Fidelity / Fake Audio Detection

#### SAFE (2024, Withdrawn)
- "Spiking Neural Network-based Audio Fidelity Evaluation"
- submitted to ICLR 2025, withdrawn
- first attempt at using SNNs for deepfake audio detection

### Other Audio SNN Work

**SOM-Associated-SNN (2025):** Neurocomputing, May 2025. 3-layer SNN with SOM clustering + STDP + associative learning on Spoken-MNIST and SHD. no backpropagation needed -- unsupervised/biologically plausible.

**Ternary Spike System (2024/2025):** arXiv:2407.05310, Neural Networks 2025. TAE encoding produces ternary spikes {-1, 0, 1}; QT-SNN quantizes membrane potentials and weights. 94% memory reduction, 7.5x energy savings. speech recognition and EEG.

**Cochlear Encoding Comparison (Meunier et al., 2025):** IEEE AICAS 2025. hardware-friendly cochlear encoders can outperform bio-mimetic ones in accuracy and energy. tested on Heidelberg Digits and Google Speech Commands.

**Spiking-LEAF (ICASSP 2024):** learnable filter bank + IHC-LIF neuron model inspired by inner hair cells. keyword spotting, speaker ID. outperforms SOTA spike encodings and conventional fbank features.

**Spike Time Difference Encoders (2025):** arXiv:2503.15402. TDE feedforward (89%) vs CuBa-LIF feedforward (71%) vs recurrent CuBa-LIF (91%) on TIdigits. TDE achieves 92% fewer synaptic operations than recurrent.

---

## 3. Best Reported SNN Accuracies on Environmental Sound Benchmarks

### ESC-50 (50 classes)

| Method | Architecture | Accuracy | Year | Reference |
|--------|-------------|----------|------|-----------|
| **our thesis** | **Conv SNN (LIF, surrogate gradients)** | **47.15% +/- 4.50%** | **2026** | **this thesis** |
| **our thesis + PANNs** | **PANNs CNN14 + SNN head** | **92.50% +/- 1.30%** | **2026** | **this thesis** |
| No other SNN work exists | -- | -- | -- | -- |

we are literally the first and only work on full ESC-50 with an SNN. confirmed by:
1. Larroza et al. (2025) explicitly stating "no state-of-the-art solution has yet encoded environmental sound datasets using spike-based methods"
2. Basu et al. (2025) survey finding no ESC-50 SNN results
3. Baek & Lee (2024) review finding no ESC-50 SNN results

### ESC-10 (10 classes)

| Method | Architecture | Accuracy/F1 | Year |
|--------|-------------|-------------|------|
| Larroza et al. (TAE) | 4-layer FC SNN, 128 LIF | 69.0% | 2025 |
| Larroza et al. (MW) | Same | 62.0% | 2025 |
| Larroza et al. (SF) | Same | 59.8% | 2025 |
| Non-spiking baseline | -- | 72.7% | 2025 |

### UrbanSound8K

| Method | Architecture | Accuracy | Notes | Year |
|--------|-------------|----------|-------|------|
| S-CMRL | Transformer SNN | 98.13% | Multimodal (AV) | 2025 |
| SMMT (Guo) | Transformer SNN | 96.85% | Multimodal (AV) | 2024 |
| Larroza et al. (SF) | FC SNN | 56.4% | Audio-only | 2025 |
| Larroza et al. (TAE) | FC SNN | 53.5% | Audio-only | 2025 |
| Non-spiking baseline | -- | 73.0% | Audio-only | 2025 |

important: the high UrbanSound8K numbers (96-98%) are MULTIMODAL audio-visual, not audio-only. audio-only SNN on UrbanSound8K peaks at only 56.4%.

---

## 4. Spike Encoding Methods Used in Audio SNNs

| Encoding Method | Used In | Domain | Notes |
|----------------|---------|--------|-------|
| **Direct (learnable)** | SpikeSCR, SpikCommander, our thesis | Speech, ESC | Most common for surrogate gradient training |
| **Rate coding** | Wu et al. 2018, our thesis | Speech, ESC | Straightforward but needs many timesteps |
| **Threshold Adaptive (TAE)** | Larroza 2025, Ternary Spike 2024 | ESC, speech | Best for environmental sound in Larroza study |
| **Step Forward (SF)** | Larroza 2025 | ESC | Second-best on UrbanSound8K |
| **Moving Window (MW)** | Larroza 2025 | ESC | Worst overall in their study |
| **Latency (time-to-first-spike)** | our thesis, TTFS literature | ESC | 4-7.5x fewer operations than rate |
| **Phase coding** | our thesis | ESC | Tied with rate in our study |
| **Population coding** | our thesis | ESC | Underperformed in our study |
| **Delta (temporal difference)** | our thesis | ESC | Very poor for static spectrograms |
| **Burst coding** | our thesis | ESC | Worst in our study (6.50%) |
| **Hilbert Transform** | Haghighatshoar 2025 | SSL | Event-based encoding of analytic signal phase |
| **RF-PLC** | Zhang 2024 (NeurIPS) | SSL | Phase-locking with Resonate-and-Fire neurons |
| **Speech2Spikes** | Orchard et al. 2023 | KWS | Delta-based; 88.5% on GSC |
| **Cochlear/IHC-LIF** | Spiking-LEAF 2024 | KWS | Learnable auditory frontend |
| **Mel spectrogram + LIF embedding** | SpikeSCR, SpikCommander | Speech | Standard in Spiking Transformer literature |
| **ANN-to-SNN conversion** | Spiking-FullSubNet 2024 | SE | Post-training conversion |

the most successful recent audio SNNs (SpikeSCR, SpikCommander) use **direct encoding** where mel spectrograms go through learnable linear layers into spike embeddings via LIF neurons. this is basically what we call "direct encoding" -- it bypasses handcrafted encoding entirely and lets the network learn optimal spike representations.

our thesis is the ONLY work to compare 7 encoding methods on the same architecture and dataset for environmental sound. the encoding hierarchy we found (direct >> rate = phase > population > latency >> delta = burst) is a novel contribution with no precedent.

---

## 5. Neuromorphic Hardware for Audio SNNs

| Hardware | Audio Task | Accuracy | Energy | Reference |
|----------|-----------|----------|--------|-----------|
| **SpiNNaker 1** | Pure tone classification (8 classes) | >85% (SNR>3dB) | -- | Dominguez-Morales et al. 2016 |
| **SpiNNaker 1** | ESC-50 FC2-only (ours) | 33.1% +/- 6.9% | 86 nJ/sample | our thesis |
| **SpiNNaker 2** | Keyword spotting (GSC) | 91.12% | 12x < V100 GPU | Yan et al. 2022 |
| **Intel Loihi** | Keyword spotting (GSC) | ~88.5% | 109x < GPU, 23x < CPU | Speech2Spikes 2023 |
| **Intel Loihi 2** | Keyword spotting | ~comparable | 200x < embedded GPU | 2024 demos |
| **Synsense Xylo** | Sound source localization | MAE 0.25-0.65 deg | Ultra-low-power | Haghighatshoar 2025 |
| **FPGA** | Event-graph audio classification | SOTA for FPGA | Low latency/power | 2025 |

our thesis is one of very few works deploying SNN for environmental sound (not just speech) on neuromorphic hardware. only prior work is Dominguez-Morales et al. (2016) with 8 pure tones on SpiNNaker. our work is a massive step up in complexity.

---

## 6. PANNs / Pretrained Audio Features with SNNs

**found nothing.** no prior work combines PANNs (or any pretrained audio features) with an SNN classifier head for environmental sound classification.

our PANNs CNN14 frozen embeddings + 3-layer SNN head achieving 92.50% on ESC-50 is novel. the insight that the SNN-ANN gap collapses from 16.7pp (scratch) to 0.95pp (PANNs features) is an original contribution.

related but different:
- ANN-to-SNN conversion (Bu et al. CVPR 2025; ICLR 2024) converts pretrained ANNs to SNNs, but vision/NLP focused
- SpikeSCR KD transfers between SNN teachers/students, but doesn't use pretrained audio features
- Spiking-LEAF uses learnable auditory frontend but trains from scratch

the PANNs+SNN experiment answers: "is the SNN accuracy gap a feature-learning problem or a spiking computation problem?" answer: feature-learning. first time this has been shown in audio.

---

## 7. Survey Papers and Reviews

**Basu et al. (2025):** "Fundamental Survey on Neuromorphic Audio Classification," arXiv:2502.15056. 24-page survey covering SNNs, memristors, neuromorphic hardware for audio. reviews ESC-50 as a dataset but reports no SNN results on it.

**Baek & Lee (2024):** "SNN and Sound: A Comprehensive Review," Biomedical Engineering Letters, Vol. 14(5):981-991. covers speech recognition and classification. reviews Wu et al. (2018, SOM-SNN 99.60% RWCP), Dong et al. (2018, Conv SNN 97.5% TIDIGITS), etc. notable gap: focuses on speech datasets, environmental sound barely covered.

---

## 8. Theses on SNNs for Audio

found a few but nothing on environmental sound:

| Thesis | University | Year | Topic |
|--------|-----------|------|-------|
| Daddinounou | Grenoble Alpes | 2024 | Neuromorphic SNNs with Spintronic Synapses (hardware) |
| Rios-Navarro | Seville | 2022 | Neuromorphic Auditory Computing for Robotics |
| Various masters | Human Brain Project | 2022-2024 | Gradient estimation for analog neuromorphic hardware |

**no thesis from Edinburgh, UCL, Imperial, ETH Zurich, MIT, TU Munich, TU Delft, or KU Leuven specifically addresses SNN-based environmental sound classification.** their SNN work focuses on vision, robotics, or theory.

---

## 9. Results Tables

### Speech Command Datasets

| Method | SHD | SSC | GSC v2 | Params | Timesteps | Year |
|--------|-----|-----|--------|--------|-----------|------|
| SpikCommander | **96.41%** | **83.26%** | **96.71%** | 0.19-1.12M | 100 | 2026 |
| SpikeSCR | 95.70% | 82.79% | 95.60% | 1.63M | 100 | 2024 |
| DCLS-Delays | 95.07% | 80.69% | 95.35% | 2.50M | 100 | 2024 |
| SpikeSCR+KDCL | 93.60% | 80.25% | 95.01% | 1.63M | **40** | 2024 |
| SIDC-KWS | -- | -- | 96.8% (12-class) | -- | -- | 2025 |
| Speech2Spikes+SNN | -- | -- | 88.5% | -- | -- | 2023 |
| E-prop (SpiNNaker2) | -- | -- | 91.12% | 25K weights | online | 2022 |

### Environmental Sound Datasets

| Method | Dataset | Classes | Accuracy | Architecture | Year |
|--------|---------|---------|----------|-------------|------|
| **our thesis** | **ESC-50** | **50** | **47.15%** | **Conv SNN (LIF)** | **2026** |
| **our thesis + PANNs** | **ESC-50** | **50** | **92.50%** | **PANNs+SNN head** | **2026** |
| Larroza (TAE) | ESC-10 | 10 | 69.0% | FC-only SNN | 2025 |
| Larroza (TAE) | UrbanSound8K | 10 | 53.5% | FC-only SNN | 2025 |
| Larroza (SF) | UrbanSound8K | 10 | 56.4% | FC-only SNN | 2025 |
| S-CMRL | US8K-AV | 10 | 98.13% | Transformer SNN (multimodal) | 2025 |
| SMMT (Guo) | US8K-AV | 10 | 96.85% | Transformer SNN (multimodal) | 2024 |

### Sound Classification (Non-Environmental)

| Method | Dataset | Accuracy | Architecture | Year |
|--------|---------|----------|-------------|------|
| Wu et al. (SOM-SNN) | RWCP | 99.60% | SOM+SNN | 2018 |
| Wu et al. (SOM-SNN) | TIDIGITS | 97.4% | SOM+SNN | 2018 |
| Amin (ATM-SNN) | TIDIGITS | 97.64% | Adaptive threshold SNN | 2021 |
| Dong et al. | TIDIGITS | 97.5% | Conv SNN (STDP) | 2018 |
| Bensimon et al. | RWCP | 98.73% | SCTN-SNN | 2021 |
| Yang & Chang | TIMIT | PER 22.6% | RSNN (71.2 uW) | 2024 |

---

## 10. Adversarial Robustness of Audio SNNs

nobody has studied this before us. our finding -- SNN retains 26% accuracy at FGSM eps=0.1 while ANN drops to 1.75% -- is novel in the audio domain.

relevant work from vision: Wang et al. (2025, arXiv:2512.22522) warns that SNN robustness may be overestimated due to gradient estimation issues and proposes Adaptive Sharpness Surrogate Gradient. also FEEL-SNN (NeurIPS 2024) on robust SNNs with sparse connections, and Sharmin et al. (ECCV 2020) as the original study.

we should cite Wang et al. (2025) and acknowledge that FGSM/PGD attacks may underestimate SNN vulnerability due to surrogate gradient inaccuracies. but our qualitative finding of different robustness behavior still holds and is first in audio.

---

## 11. Continual Learning with Audio SNNs

also nobody. our experiment (SNN forgetting: 74.4% vs ANN forgetting: 81.3%, SNN forgets 6.9pp less on ESC-50 super-categories) is novel.

---

## 12. Where Our Work Fits

### what's novel about our work

1. **first SNN on full ESC-50 (50 classes)** -- confirmed, nothing else exists
2. **most extensive encoding comparison** -- 7 encodings on same architecture. Larroza compared only 3 on ESC-10
3. **first PANNs+SNN hybrid for audio** -- no prior work
4. **first SNN adversarial robustness study for audio** -- prior work all vision
5. **first continual learning study for audio SNNs** -- no prior work
6. **SpiNNaker deployment for ESC-50** -- first environmental sound SNN on neuromorphic hardware beyond trivial pure tones
7. **surrogate gradient ablation for audio SNNs** -- no prior comparison of 8 surrogates for audio SNN training

### complexity spectrum

```
                    AUDIO SNN COMPLEXITY SPECTRUM

Simple                                              Complex
|----|----|----|----|----|----|----|----|----|----|
Pure     Digits    KWS      ESC-10   ESC-50   Full
tones                                         AudioSet

Dominguez  Wu/Dong  SpikeSCR Larroza  [US]     [None]
-Morales   2018     2024     2025
2016                         (FC-only)

                             69.0%    47.15%
                                      92.50%
                                      (PANNs)
```

we tackle the most complex audio classification task ever attempted with an SNN and get competitive results with pretrained features.

---

## 13. Gaps and Limitations of This Search

things i couldn't fully resolve:
1. exact audio-only accuracy from Guo et al. (2024) -- they don't report it separately from multimodal
2. full details from Basu et al. (2025) survey -- couldn't extract full text
3. some institutional thesis repositories were limited to web-accessible metadata
4. Chinese-language SNN audio papers may exist but aren't indexed in English
5. can't search upcoming ICONS 2026 submissions obviously

---

## 14. Citations

### must-cite (directly relevant)
1. Larroza et al. (2025) arXiv:2503.11206 -- closest competitor, ESC-10 only
2. Baek & Lee (2024) Biomedical Eng. Letters -- SNN+sound review
3. Basu et al. (2025) arXiv:2502.15056 -- neuromorphic audio survey
4. Dominguez-Morales et al. (2016) ICANN -- SpiNNaker audio predecessor
5. Wu et al. (2018) Frontiers -- SOM-SNN framework
6. Wang et al. (2025) arXiv:2512.22522 -- adversarial robustness evaluation warning

### should-cite (context)
7. SpikeSCR (Wang et al. 2024) -- SOTA speech command SNN
8. Spiking-FullSubNet (Hao et al. 2024) -- SNN competition winner
9. Zhang et al. (NeurIPS 2024) -- RF-PLC sound localization
10. Guo et al. (2024) IEEE TCDS -- multimodal audio SNN
11. Meunier et al. (2025) IEEE AICAS -- cochlear encoding comparison
12. Haghighatshoar & Muir (2025) Comm. Eng. -- SNN audio localization
13. Speech2Spikes (2023) NICE -- audio encoding pipeline
14. Spiking-LEAF (ICASSP 2024) -- learnable auditory frontend

---

## 15. Confidence

| Finding | Confidence | Basis |
|---------|-----------|-------|
| No prior SNN on full ESC-50 | Very high (95%+) | Multiple surveys confirm; Larroza et al. explicitly claim it |
| No prior PANNs+SNN for audio | High (90%) | searched everywhere, found nothing |
| No prior adversarial robustness for audio SNNs | High (90%) | all adversarial SNN papers are vision-domain |
| No prior continual learning for audio SNNs | High (90%) | found nothing |
| SpikCommander is current SOTA on SHD/SSC/GSC | High (85%) | arXiv Jan 2026, most recent |
| Larroza best ESC-10 result is 69.0% | Very high (95%) | directly from paper |
| Our 47.15% is competitive given ESC-50 difficulty | Very high (95%) | 50 classes vs 10, CNN arch vs FC-only |
