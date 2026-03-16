# ANN-to-SNN Conversion: Comprehensive Research Report for Undergraduate Thesis Direction

**Research Date:** 2026-02-25
**Scope:** Evaluating ANN-to-SNN conversion as a practical and contributory undergraduate thesis direction

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [State of the Art (2024-2026)](#2-state-of-the-art-2024-2026)
3. [Available Tools and Frameworks](#3-available-tools-and-frameworks)
4. [Accuracy Loss During Conversion](#4-accuracy-loss-during-conversion)
5. [Which Architectures Convert Best](#5-which-architectures-convert-best)
6. [Timestep Requirements](#6-timestep-requirements)
7. [Undergraduate Contribution Opportunities](#7-undergraduate-contribution-opportunities)
8. [Recent Papers with Reproducible Code](#8-recent-papers-with-reproducible-code)
9. [Time to Get a Working Pipeline](#9-time-to-get-a-working-pipeline)
10. [Thesis Framing Recommendations](#10-thesis-framing-recommendations)
11. [Consolidated Accuracy Tables](#11-consolidated-accuracy-tables)
12. [Research Gaps and Open Problems](#12-research-gaps-and-open-problems)
13. [Risk Assessment](#13-risk-assessment)
14. [Sources](#14-sources)

---

## 1. Executive Summary

ANN-to-SNN conversion is one of the two dominant methods for building deep spiking neural networks (the other being direct training with surrogate gradients). The core idea is straightforward: take a pre-trained artificial neural network, replace ReLU activations with integrate-and-fire spiking neurons, normalize thresholds, and run inference where spike rates encode activation values. This is the **most cost-effective** method for obtaining high-accuracy SNNs because it leverages the mature ANN training ecosystem.

**Key finding for thesis viability:** This is an excellent undergraduate thesis direction. The conversion pipeline is well-supported by existing tools (SpikingJelly, snn_toolbox, snntorch, and standalone paper implementations), the core experiments are reproducible within weeks, and there are clear contribution opportunities in under-explored domains and architecture comparisons. The field is actively producing top-venue publications (ICML 2024/2025, CVPR 2025, NeurIPS 2023, ECCV 2024) with open-source code, making it both current and accessible.

**The strongest thesis framing would be:** "Evaluating the Practicality of ANN-to-SNN Conversion for [Specific Domain/Architecture]" -- where the specific domain is chosen to be something not yet comprehensively studied (medical imaging, audio classification, lightweight architectures like MobileNet/EfficientNet, or a head-to-head tool comparison).

---

## 2. State of the Art (2024-2026)

### 2.1 The Evolution of ANN-to-SNN Conversion

ANN-to-SNN conversion has evolved through three major phases:

**Phase 1 (2015-2019): Basic Rate Coding**
- Replace ReLU with IF neurons, normalize weights/thresholds
- Required 500-2500+ timesteps for competitive accuracy
- Limited to VGG-like architectures on CIFAR-10/MNIST
- Key papers: Diehl et al. 2015, Sengupta et al. 2019

**Phase 2 (2020-2023): Optimized Conversion with Reduced Latency**
- Introduction of threshold balancing, weight normalization, calibration
- Reduction to 32-256 timesteps while maintaining accuracy
- Extension to ResNets, deeper architectures, ImageNet-scale
- Key papers: SNN Calibration (ICML 2021), QCFS (ICLR 2022), unified framework (ICML 2023)

**Phase 3 (2024-2026): Ultra-Low Latency and Beyond-CNN Architectures**
- Conversion with 1-8 timesteps achieving near-ANN accuracy
- First successful Transformer-to-SNN conversions
- First conversion of non-ReLU architectures (ConvNeXt, MLP-Mixer, ResMLP)
- Training-free conversion methods eliminating retraining requirements
- Extension to object detection, semantic segmentation, video classification

### 2.2 Landmark Papers (2024-2026)

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Sign Gradient Descent-based Neuronal Dynamics | ICML 2024 | First to convert ConvNeXt, MLP-Mixer, ResMLP (beyond ReLU) |
| Optimal ANN-SNN Conversion with Group Neurons | ICASSP 2024 | ResNet-34 on ImageNet: 73.61% at T=2 |
| SpikeYOLO | ECCV 2024 (Best Paper Candidate) | Integer-valued training + spike-driven inference for detection |
| Inference-Scale Complexity in ANN-SNN Conversion | CVPR 2025 | Training-free conversion at inference-scale; classification, segmentation, detection, video |
| Differential Coding for Training-Free ANN-to-SNN | ICML 2025 | Novel differential coding reduces spike counts and energy |
| Towards High-performance Spiking Transformers | ICLR 2025 | Spiking Transformer: 88.60% top-1, only 1% loss at T=4 |
| One-Timestep is Enough | 2025 | Scale-and-Fire neurons achieve high accuracy at T=1 |

### 2.3 Current Research Frontiers

1. **Ultra-low latency (T=1-4)**: The holy grail -- achieving ANN-level accuracy with minimal timesteps
2. **Transformer conversion**: Moving beyond CNNs to convert attention-based architectures
3. **Beyond-ReLU activation conversion**: Converting architectures using GELU, SiLU, Swish
4. **Training-free conversion**: No retraining needed -- just convert and run
5. **Domain expansion**: Object detection, segmentation, video, NLP tasks
6. **Energy-accuracy co-optimization**: Jointly optimizing for accuracy and neuromorphic energy efficiency
7. **Adaptive inference**: Dynamically adjusting timesteps per input for efficiency

---

## 3. Available Tools and Frameworks

### 3.1 Tool Comparison Matrix

| Feature | snn_toolbox | SpikingJelly (ann2snn) | snnTorch | Custom Paper Code |
|---------|------------|----------------------|---------|-------------------|
| **Input framework** | Keras, PyTorch, Caffe, Lasagne | PyTorch only | PyTorch only | Varies (usually PyTorch) |
| **Conversion method** | Weight normalization + threshold balancing | MaxNorm / RobustNorm / Scaling | Basic IF neuron replacement | Method-specific (QCFS, calibration, etc.) |
| **Backend simulators** | INIsim (built-in), Brian2, pyNN, SpiNNaker, Loihi | SpikingJelly's own simulator | snnTorch simulator | Usually custom |
| **Hardware deployment** | SpiNNaker, Loihi support | Limited | No | Varies |
| **Documentation quality** | Good (ReadTheDocs) | Good (English + Chinese) | Excellent (tutorials, Colab) | Varies (often minimal) |
| **Learning curve** | Moderate | Low (PyTorch-native) | Low (PyTorch-native) | High (read the paper) |
| **Maintenance (2024-25)** | Low activity; ~387 stars, some stale dependencies | Active; published in Science Advances; high activity | Active; good tutorial ecosystem | Depends on authors |
| **Best for** | Multi-backend deployment, SpiNNaker/Loihi | Fast prototyping, research | Learning, education, visualization | State-of-the-art results |
| **GitHub** | NeuromorphicProcessorProject/snn_toolbox | fangwei123456/spikingjelly | jeshraghian/snntorch | Various |

### 3.2 snn_toolbox (Detailed)

- **Repository:** https://github.com/NeuromorphicProcessorProject/snn_toolbox
- **Stars:** ~387 | **Forks:** ~103 | **Open issues:** 6
- **Pipeline:** Load ANN -> Parse layers -> Normalize parameters -> Convert to SNN -> Simulate
- **Supported layers:** Conv2D, Dense, BatchNorm (absorbed into preceding layer), Pooling, Flatten
- **Known limitations:**
  - Conv1D has normalization issues (GitHub issue #129)
  - Keras version compatibility problems (issue #87)
  - Lower maintenance activity in 2024-2025
  - ResNet accuracy degradation during conversion (issue #66)
- **Configuration:** Extensive config file system documented at ReadTheDocs
- **Best use case:** If you need SpiNNaker or Loihi deployment, or multi-backend comparison

### 3.3 SpikingJelly ann2snn (Detailed)

- **Repository:** https://github.com/fangwei123456/spikingjelly
- **Published:** Science Advances (2023) -- high credibility
- **Conversion modes:**
  - `MaxNorm`: Uses max activation values for threshold setting
  - `RobustNorm`: Uses 99.9% activation quantile (more robust to outliers)
  - Scaling mode: User-specified scaling parameters
- **Process:** ReLU -> IF neurons; AvgPool -> spatial downsampling
- **Pre-built examples:** `resnet18_cifar10.py`, `cnn_mnist.py`
- **Speed advantage:** Up to 11x speedup over other frameworks when T=32
- **Full-stack:** Supports neuromorphic datasets, ANN2SNN, surrogate gradients, and bio-plausible rules
- **Best use case:** Research prototyping, fast iteration, if already using PyTorch

### 3.4 snnTorch (Detailed)

- **Repository:** https://github.com/jeshraghian/snntorch
- **Tutorials:** Comprehensive series (Tutorial 1 through 7+), Colab notebooks
- **Conversion approach:** Simpler, more educational -- demonstrates basic ANN-to-SNN concepts
- **Example:** `Saad-data/ANN-to-SNN-Conversion-with-snnTorch` demonstrates a simple FC + ReLU -> SNN conversion
- **Best use case:** Learning the fundamentals, educational projects, clear visualization of spike dynamics

### 3.5 Standalone Research Code (Most Recommended for State-of-the-Art)

For an undergraduate thesis aiming at current results, the standalone paper implementations are often the best starting point:

| Repository | Paper | Venue | Ease of Use |
|-----------|-------|-------|-------------|
| `putshua/ANN_SNN_QCFS` | QCFS: Optimal conversion | ICLR 2022 | High -- fixed bugs, shared weights, clear commands |
| `yhhhli/SNN_Calibration` | SNN Calibration | ICML 2021 | High -- clear CLI, supports VGG16/ResNet on CIFAR |
| `snuhcs/snn_signgd` | SignGD beyond ReLU | ICML 2024 | Medium -- advanced, converts ConvNeXt/MLP-Mixer |
| `Lyu6PosHao/ANN2SNN_GN` | Group Neurons | ICASSP 2024 | Medium -- recent, good results |
| `IGITUGraz/RobustSNNConversion` | Adversarial robustness | TMLR 2024 | Medium |
| `h-z-h-cell/ANN-to-SNN-DCGS` | Differential coding | ICML 2025 | Medium-High -- source available |
| `BICLab/SpikeYOLO` | SpikeYOLO detection | ECCV 2024 | Medium -- object detection focus |

---

## 4. Accuracy Loss During Conversion

### 4.1 General Principles

Accuracy loss in ANN-to-SNN conversion arises from three fundamental error sources:

1. **Quantization error**: Continuous ReLU activations are approximated by discrete spike counts
2. **Clipping error**: Activation values exceeding the firing threshold are lost
3. **Residual membrane potential error**: Information stored in membrane potential at the end of simulation is discarded

The accuracy gap follows a clear trend: **more timesteps = lower accuracy loss, but higher latency and energy**.

### 4.2 Concrete Accuracy Numbers

#### CIFAR-10 (Approximate ANN Baseline: VGG16 ~93.5%, ResNet-20 ~92%)

| Method | Architecture | Timesteps (T) | SNN Accuracy | ANN Accuracy | Loss |
|--------|-------------|---------------|-------------|-------------|------|
| QCFS (ICLR 2022) | VGG-16 | 4 | 93.05% | 93.63% | 0.58% |
| QCFS | VGG-16 | 1 | 93.05% | 93.63% | 0.58% |
| SNN Calibration (ICML 2021) | VGG-16 | 16 | 93.63% | 93.71% | 0.08% |
| SEENN (NeurIPS 2023) | VGG-16 | ~1.4 | 93.63% | -- | near-lossless |
| SlipReLU (2023) | ResNet-20 | 2 | 82.25% | ~92% | ~10% |
| One-Timestep (2025) | ResNet-18 | 1 | 93.11% | ~93.5% | ~0.4% |
| PMSM (2024) | ViT-S | 1 | 98.5% | -- | near-lossless |

#### CIFAR-100 (Approximate ANN Baseline: VGG16 ~73%, ResNet-34 ~77%)

| Method | Architecture | Timesteps (T) | SNN Accuracy | Loss |
|--------|-------------|---------------|-------------|------|
| QCFS | VGG-16 | 4 | 70.15% | ~3% |
| SNN Calibration | VGG-16 | 16 | ~71% | ~2% |
| Various | ResNet-34 | 16 | ~70.97% | ~6% |
| Knowledge Distillation (2024) | ResNet-34 | 4 | 70.04% | ~7% |

#### ImageNet (Approximate ANN Baseline: VGG16 ~71.6%, ResNet-34 ~73.3%)

| Method | Architecture | Timesteps (T) | SNN Accuracy | Loss |
|--------|-------------|---------------|-------------|------|
| Group Neurons (ICASSP 2024) | ResNet-34 | 2 | 73.61% | ~0% |
| Threshold Balancing | VGG-16 | 250 | 73.87% | ~0% |
| QCFS + TPP | VGG-16 | 16 | 73.98% | ~0% |
| QCFS + TPP | ResNet-34 | 16 | 72.03% | ~1% |
| Spiking Transformer (2025) | ViT/DeiT | 4 | 88.60% | ~1% |

### 4.3 Rules of Thumb for Accuracy Loss

| Timesteps | Typical Accuracy Loss (CIFAR-10) | Typical Loss (ImageNet) |
|-----------|--------------------------------|------------------------|
| T = 1 | 0-1% (with modern methods) | 2-5% |
| T = 2-4 | 0-0.5% | 0-2% |
| T = 8-16 | Near-lossless | 0-1% |
| T = 32-64 | Lossless | Near-lossless |
| T = 128-256 | Lossless | Lossless |
| T >= 500 | Lossless (classical methods) | Lossless (classical methods) |

**Bottom line:** With modern methods (QCFS, calibration, group neurons), you can achieve less than 1% accuracy loss at T=4-16 on CIFAR-10/100. ImageNet requires slightly more timesteps but is feasible at T=4-16 with the latest methods.

---

## 5. Which Architectures Convert Best

### 5.1 Architecture Conversion Difficulty Ranking

| Architecture | Conversion Difficulty | Key Challenges | Status (2025) |
|-------------|---------------------|----------------|---------------|
| **VGG-16** | EASY | Pure Conv+ReLU+Pool, no skip connections | Fully solved, many papers |
| **ResNet-18/20/34** | MODERATE | Skip connections cause "deviation error" at residual merges | Well-studied, good results with calibration |
| **PreActResNet** | MODERATE | Pre-activation variant, slightly different conversion dynamics | Supported in several papers |
| **MobileNet v1/v2** | MODERATE-HARD | Depthwise separable convolutions, squeeze-excite blocks | Limited published work, gap opportunity |
| **EfficientNet** | HARD | Compound scaling, Swish/SiLU activation (not ReLU), SE blocks | Very few published results, significant gap |
| **DenseNet** | MODERATE-HARD | Dense connections, high memory, feature concatenation | Minimal published results |
| **ConvNeXt** | HARD | GELU activation, LayerNorm, not standard ReLU | First converted in ICML 2024 (SignGD) |
| **MLP-Mixer** | HARD | Token mixing, GELU, non-convolutional | First converted in ICML 2024 (SignGD) |
| **Vision Transformer (ViT/DeiT)** | VERY HARD | Softmax attention, LayerNorm, GELU, multi-head attention | First successful conversion in 2025 |
| **YOLO (detection)** | HARD | Multi-scale features, non-max suppression, detection heads | SpikeYOLO (ECCV 2024), Spiking-YOLO |
| **U-Net (segmentation)** | MODERATE-HARD | Encoder-decoder, skip connections, upsampling | Spiking-UNet exists, limited conversion studies |

### 5.2 Why VGG Converts Best

VGG architectures convert most easily because:
1. They use only standard Conv2D, ReLU, MaxPool, and Dense layers
2. No skip connections means no "deviation error" from residual additions
3. BatchNorm can be cleanly folded into preceding Conv layers
4. The deep, sequential structure maps naturally to layer-by-layer spiking dynamics

### 5.3 Why ResNets are Harder

ResNets suffer from a specific problem: the last ReLU in each residual block is followed by a skip connection addition. This creates a "deviation error" because the spike-based approximation of the addition of two activations introduces systematic bias. Modern methods (calibration, QCFS) largely solve this, but ResNets still require more careful threshold tuning than VGGs.

### 5.4 The Non-ReLU Frontier

The biggest recent breakthrough is converting architectures that do NOT use ReLU:
- **ConvNeXt** uses GELU activation
- **Transformers** use GELU + Softmax + LayerNorm
- The SignGD paper (ICML 2024) was the **first** to convert these successfully
- This remains a frontier area with opportunities for contribution

---

## 6. Timestep Requirements

### 6.1 The Timestep-Accuracy-Energy Tradeoff

The fundamental tradeoff in ANN-to-SNN conversion:

```
More timesteps -> Higher accuracy -> Higher latency -> More energy (on conventional hardware)
                                                    -> BUT potentially less energy on neuromorphic hardware
                                                       (if spike sparsity is high enough)
```

### 6.2 Timestep Requirements by Method Era

| Method Generation | Typical Timesteps | Year Range |
|------------------|-------------------|------------|
| Classical (weight norm, threshold balance) | 500-2500 | 2015-2019 |
| Improved (calibration, trainable clipping) | 32-256 | 2020-2022 |
| Modern (QCFS, quantization-aware) | 4-32 | 2022-2024 |
| Cutting-edge (one-timestep, scale-and-fire) | 1-4 | 2024-2026 |

### 6.3 Practical Recommendations

For an undergraduate thesis:
- **Start with T=32-64** using a standard method (SNN Calibration or snn_toolbox) -- this will work reliably
- **Then reduce to T=8-16** using QCFS or similar modern methods
- **Then optionally try T=1-4** if time permits, to demonstrate cutting-edge results
- This progression itself can be a thesis contribution: "How does accuracy degrade as we reduce timesteps?"

### 6.4 The One-Timestep Frontier

A major 2025 paper titled "One-Timestep is Enough" proposes Scale-and-Fire neurons that achieve high-performance ANN-to-SNN conversion at T=1. At T=1, the SNN essentially operates like a quantized ANN, blurring the line between the two paradigms. This is conceptually interesting but somewhat undermines the energy efficiency argument for SNNs, since the sparsity advantage only kicks in when individual spikes are sparse across time.

### 6.5 Key Insight on Diminishing Returns

On CIFAR-10, increasing timesteps from T=2 to T=6 only improves accuracy by 0.34% while tripling latency. This diminishing-returns curve is a useful result to reproduce and demonstrate in a thesis.

---

## 7. Undergraduate Contribution Opportunities

### 7.1 Tier 1: High Feasibility, Clear Contribution (Recommended)

#### Option A: Head-to-Head Tool Comparison Study
**Thesis title:** "A Comparative Evaluation of ANN-to-SNN Conversion Tools: snn_toolbox, SpikingJelly, and QCFS on Standard Benchmarks"

- Convert the SAME pretrained models (VGG-16, ResNet-18/20) using 3+ different tools/methods
- Compare: accuracy, timesteps needed, conversion time, ease of use, code quality, documentation
- Datasets: CIFAR-10, CIFAR-100 (maybe ImageNet if GPU time permits)
- Include energy estimation using syops or spike counting
- **Why this contributes:** No published paper systematically compares these tools head-to-head. Tool comparison papers are highly cited.
- **Time estimate:** 6-8 weeks for core experiments

#### Option B: Converting a Domain-Specific Model Nobody Has Converted
**Thesis title:** "Evaluating ANN-to-SNN Conversion for [Medical Image Classification / Audio Keyword Spotting / Satellite Imagery]"

Potential under-explored domains:
1. **Medical image classification**: Convert a pretrained model (e.g., ResNet-18 trained on skin lesion classification, diabetic retinopathy grading, or chest X-ray) to SNN. Medical imaging is extremely well-studied for ANNs but almost entirely unexplored for ANN-to-SNN conversion.
2. **Audio keyword spotting**: Convert a small ANN keyword spotter to SNN. Some work exists but this is still underexplored, especially with modern conversion methods.
3. **Satellite/remote sensing imagery**: Virtually no published ANN-to-SNN conversion work for satellite image classification.
4. **DVS128 gesture recognition + conversion comparison**: Convert an ANN trained on frame-binned DVS data to SNN, then compare with a directly-trained SNN (you already have the DVS128 pipeline from your other research direction).

- **Why this contributes:** Demonstrates practical applicability to a real domain beyond CIFAR/ImageNet benchmarks.
- **Time estimate:** 4-6 weeks for core experiments (pretrained ANN likely available on HuggingFace or torchvision)

#### Option C: Architecture Comparison -- Which Architectures Survive Conversion?
**Thesis title:** "Beyond VGG: Evaluating ANN-to-SNN Conversion Across Modern Architectures"

- Convert VGG-16, ResNet-18, MobileNetV2, EfficientNet-B0, DenseNet-121 using the same conversion method
- Systematically document: which layers cause problems, what accuracy loss occurs, how many timesteps needed
- **MobileNet and EfficientNet conversion is severely underexplored** -- depthwise separable convolutions and Swish activation create known difficulties
- **Why this contributes:** Most conversion papers test only VGG + ResNet. A systematic evaluation of modern efficient architectures is genuinely missing from the literature.
- **Time estimate:** 6-10 weeks (some architectures may require debugging)

### 7.2 Tier 2: Moderate Feasibility, Stronger Contribution

#### Option D: Conversion + Neuromorphic Energy Analysis
**Thesis title:** "Energy-Accuracy Tradeoffs in ANN-to-SNN Conversion: From Theory to Practice"

- Convert models, measure spike counts and sparsity at different timesteps
- Use syops library to estimate energy on neuromorphic hardware
- Compare theoretical energy savings vs. actual GPU energy measurements
- Reference the critical paper "Are SNNs Really More Energy-Efficient than ANNs?" which shows energy savings depend critically on spike sparsity and hardware
- **Why this contributes:** Energy claims in SNN papers are often theoretical. A rigorous undergraduate analysis with actual measurements would be valuable.

#### Option E: Combining Conversion with Direct Training (Hybrid)
**Thesis title:** "Hybrid ANN-SNN Training: Does Fine-tuning a Converted SNN with Surrogate Gradients Help?"

- Convert an ANN to SNN using standard methods
- Then fine-tune the converted SNN using surrogate gradient training for a few epochs
- Measure whether this "hybrid" approach recovers accuracy lost during conversion
- Compare with pure conversion and pure direct training
- **Why this contributes:** Hybrid approaches are a growing research direction (CVPR 2024 workshop paper exists on this)

### 7.3 Tier 3: Ambitious but Publishable

#### Option F: First Systematic MobileNet/EfficientNet Conversion Study
- Would require solving depthwise separable convolution handling
- Would need to address Swish/SiLU activation for EfficientNet
- Could use SignGD (ICML 2024) approach as starting point for non-ReLU activations
- Genuinely novel -- potential workshop paper

---

## 8. Recent Papers with Reproducible Code

### 8.1 Verified Reproducible Repositories

| Repository | Paper | Venue | Language | Ease | Dataset Support |
|-----------|-------|-------|----------|------|-----------------|
| [putshua/ANN_SNN_QCFS](https://github.com/putshua/ANN_SNN_QCFS) | QCFS Optimal Conversion | ICLR 2022 | PyTorch | HIGH | CIFAR-10/100, ImageNet |
| [yhhhli/SNN_Calibration](https://github.com/yhhhli/SNN_Calibration) | SNN Calibration | ICML 2021 | PyTorch | HIGH | CIFAR-10/100, ImageNet |
| [snuhcs/snn_signgd](https://github.com/snuhcs/snn_signgd) | SignGD Beyond ReLU | ICML 2024 | PyTorch | MEDIUM | CIFAR-10/100, ImageNet |
| [Lyu6PosHao/ANN2SNN_GN](https://github.com/Lyu6PosHao/ANN2SNN_GN) | Group Neurons | ICASSP 2024 | PyTorch | MEDIUM | CIFAR-10/100, ImageNet |
| [IGITUGraz/RobustSNNConversion](https://github.com/IGITUGraz/RobustSNNConversion) | Adversarial Robustness | TMLR 2024 | PyTorch | MEDIUM | CIFAR-10/100 |
| [h-z-h-cell/ANN-to-SNN-DCGS](https://github.com/h-z-h-cell/ANN-to-SNN-DCGS) | Differential Coding | ICML 2025 | PyTorch | MEDIUM | CNNs + Transformers |
| [BICLab/SpikeYOLO](https://github.com/BICLab/SpikeYOLO) | SpikeYOLO | ECCV 2024 | PyTorch | MEDIUM | COCO, Gen1 |
| [putshua/SNN_conversion_QCFS](https://github.com/putshua/SNN_conversion_QCFS) | QCFS (older version) | -- | PyTorch | HIGH | CIFAR-10/100 |
| [nitin-rathi/hybrid-snn-conversion](https://github.com/nitin-rathi/hybrid-snn-conversion) | Hybrid SNN Conversion | -- | PyTorch | MEDIUM | CIFAR-10/100, ImageNet |
| [NeuroCompLab-psu/SNN-Conversion](https://github.com/NeuroCompLab-psu/SNN-Conversion) | General Conversion | -- | PyTorch | MEDIUM | CIFAR |
| [Saad-data/ANN-to-SNN-Conversion-with-snnTorch](https://github.com/Saad-data/ANN-to-SNN-Conversion-with-snnTorch) | snnTorch Demo | -- | PyTorch | VERY HIGH | MNIST |

### 8.2 Recommended Starting Points

**For absolute beginners:** Start with `Saad-data/ANN-to-SNN-Conversion-with-snnTorch` (MNIST, simple FC network, very short code)

**For the actual thesis work:** Start with `putshua/ANN_SNN_QCFS` or `yhhhli/SNN_Calibration` -- both have:
- Clear README with exact commands
- Fixed random seeds for reproducibility
- Support for VGG-16 and ResNet on CIFAR-10/100
- Reasonable GPU requirements

**For cutting-edge results:** Use `snuhcs/snn_signgd` which converts ConvNeXt and MLP-Mixer (ICML 2024)

### 8.3 Curated Paper Lists

For finding more papers and code:
- [Awesome-Spiking-Neural-Networks](https://github.com/zhouchenlin2096/Awesome-Spiking-Neural-Networks) -- comprehensive paper list organized by year and method
- [awesome-snn-conference-paper](https://github.com/AXYZdong/awesome-snn-conference-paper) -- top-conference papers with code
- [SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv) -- daily arxiv updates on SNN papers
- [ann-to-snn-conversion topic on GitHub](https://github.com/topics/ann-to-snn-conversion) -- all tagged repositories

---

## 9. Time to Get a Working Pipeline

### 9.1 Estimated Timeline

| Phase | Using snn_toolbox | Using SpikingJelly | Using QCFS Code | Using Custom |
|-------|------------------|-------------------|-----------------|-------------|
| Environment setup | 1-2 days | 0.5-1 day | 0.5-1 day | 1-3 days |
| Understanding the tool | 2-3 days | 1-2 days | 1-2 days | 3-7 days (read paper) |
| First successful MNIST conversion | 1 day | 1 day | N/A (starts at CIFAR) | 2-3 days |
| First CIFAR-10 conversion | 2-3 days | 1-2 days | 0.5 day (has scripts) | 3-5 days |
| Tuning for good accuracy | 3-5 days | 2-3 days | 1-2 days (pre-tuned) | 5-10 days |
| CIFAR-100 experiments | 1-2 days | 1-2 days | 0.5 day | 2-3 days |
| ImageNet experiments | 3-7 days (GPU time) | 3-7 days | 2-5 days | 5-10 days |
| **Total to working pipeline** | **~2 weeks** | **~1 week** | **~3-5 days** | **~3-4 weeks** |

### 9.2 Hardware Requirements

- **Minimum:** GPU with 6GB VRAM (RTX 2060 / Apple M1 with MPS) for CIFAR experiments
- **Recommended:** GPU with 8-12GB VRAM (RTX 3070/3080) for CIFAR + small ImageNet experiments
- **For full ImageNet:** 16GB+ VRAM or access to university compute cluster
- **CPU-only:** Possible for MNIST and small CIFAR models but painfully slow (10-100x slower)

### 9.3 Practical Tips for Fast Setup

1. **Use QCFS code as your baseline.** The commands are documented:
   - Train ANN: `python main_train.py --epochs=300 -dev=0 -L=4 -data=cifar10`
   - Test as SNN: `python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=8 -dev=0`
2. **Use pretrained ANN weights** when available (most repos provide them)
3. **Start with VGG-16 on CIFAR-10** -- this is the easiest conversion target and will verify your pipeline works
4. **Then move to ResNet-18/20 on CIFAR-10** to see how skip connections affect conversion
5. **Use CIFAR-100 to stress-test** -- accuracy gaps become more visible on harder datasets

---

## 10. Thesis Framing Recommendations

### 10.1 Recommended Thesis Structure

A strong thesis framed around ANN-to-SNN conversion should follow this structure:

**Chapter 1: Introduction**
- Motivation: energy efficiency of neuromorphic computing, bio-inspiration
- Problem: how to leverage existing trained ANNs for SNN deployment
- Thesis contribution: systematic evaluation of conversion in [specific context]

**Chapter 2: Background and Literature Review**
- SNN fundamentals (IF/LIF neurons, spike coding, membrane potential)
- ANN-to-SNN conversion theory (rate coding, threshold normalization, error sources)
- Survey of conversion methods (organized chronologically or by approach)
- Survey of tools available

**Chapter 3: Methodology**
- Selected architectures and their properties
- Selected conversion tools/methods and justification
- Experimental setup (datasets, metrics, hardware, hyperparameters)
- Energy estimation methodology (syops, spike counting, theoretical vs. measured)

**Chapter 4: Experiments and Results**
- Baseline ANN accuracy
- Converted SNN accuracy at various timesteps
- Accuracy-timestep curves
- Architecture comparison (if applicable)
- Tool comparison (if applicable)
- Energy analysis
- Failure cases and debugging observations

**Chapter 5: Discussion**
- Analysis of when conversion works well vs. poorly
- Practical recommendations for practitioners
- Comparison with direct training approaches
- Limitations of the study

**Chapter 6: Conclusion and Future Work**
- Summary of findings
- Contribution to the field
- Recommended future directions

### 10.2 Strongest Thesis Titles (Pick One)

1. "Evaluating the Practicality of ANN-to-SNN Conversion for Medical Image Classification"
2. "A Systematic Comparison of ANN-to-SNN Conversion Tools for Deep Neural Networks"
3. "Beyond VGG and ResNet: ANN-to-SNN Conversion of Modern Efficient Architectures"
4. "Energy-Accuracy Tradeoffs in ANN-to-SNN Conversion: A Practical Assessment"
5. "From Training to Spiking: How Well Do Standard Vision Models Survive ANN-to-SNN Conversion?"
6. "ANN-to-SNN Conversion for Edge Deployment: A Feasibility Study on Lightweight Architectures"

---

## 11. Consolidated Accuracy Tables

### 11.1 CIFAR-10 State of the Art (ANN-to-SNN Conversion Only)

| Method | Year | Venue | Arch | T | SNN Acc (%) | ANN Acc (%) | Gap |
|--------|------|-------|------|---|-------------|-------------|-----|
| Sengupta et al. | 2019 | Front. Neuro. | VGG-16 | 2500 | 91.55 | 91.70 | 0.15 |
| Hybrid (Rathi) | 2020 | -- | VGG-16 | 200 | 92.65 | -- | -- |
| SNN Calibration | 2021 | ICML | VGG-16 | 16 | 93.63 | 93.71 | 0.08 |
| QCFS | 2022 | ICLR | VGG-16 | 4 | 93.05 | 93.63 | 0.58 |
| SEENN | 2023 | NeurIPS | VGG-16 | ~1.4 | 93.63 | -- | ~0 |
| One-Timestep | 2025 | -- | ResNet-18 | 1 | 93.11 | ~93.5 | ~0.4 |
| PMSM | 2024 | -- | ViT-S | 1 | 98.50 | -- | ~0 |

### 11.2 ImageNet State of the Art (ANN-to-SNN Conversion Only)

| Method | Year | Venue | Arch | T | Top-1 Acc (%) |
|--------|------|-------|------|---|--------------|
| Sengupta et al. | 2019 | Front. Neuro. | VGG-16 | 2500 | 69.96 |
| SNN Calibration | 2021 | ICML | ResNet-34 | 32 | 71.78 |
| QCFS | 2022 | ICLR | ResNet-34 | 32 | ~72.5 |
| QCFS + TPP | 2024 | -- | VGG-16 | 16 | 73.98 |
| Group Neurons | 2024 | ICASSP | ResNet-34 | 2 | 73.61 |
| Spiking Transformer | 2025 | ICLR | ViT/DeiT | 4 | 88.60 |

---

## 12. Research Gaps and Open Problems

### 12.1 Gaps Suitable for Undergraduate Contribution

| Gap | Why It Matters | Difficulty | Publication Potential |
|-----|---------------|------------|---------------------|
| **No systematic tool comparison** (snn_toolbox vs SpikingJelly vs QCFS) | Practitioners do not know which tool to use | LOW | Workshop paper |
| **MobileNet/EfficientNet conversion** barely studied | Most relevant for edge deployment | MODERATE | Workshop/conference paper |
| **Medical imaging domain** almost unstudied for conversion | Huge practical impact | LOW-MODERATE | Domain-specific venue |
| **Audio/keyword spotting** with modern conversion methods | Relevant for IoT/always-on devices | MODERATE | Workshop paper |
| **Accuracy-vs-energy** with actual measurements (not just theory) | Energy claims are often unverified | MODERATE | Good thesis chapter |
| **Converting models with Swish/GELU/SiLU** activations systematically | Modern architectures all use these | MODERATE-HARD | Conference paper |
| **DenseNet conversion** not published | Dense connections create unique challenges | MODERATE | Short paper |
| **Conversion for video classification** at scale | Temporal data is natural for SNNs | HARD | Conference paper |

### 12.2 Gaps Requiring More Expertise (PhD-Level)

- Novel neuron models for better single-timestep conversion
- Theoretical analysis of conversion error bounds for new architectures
- Hardware co-design of converted SNNs for specific neuromorphic chips
- Scaling conversion to billion-parameter models

---

## 13. Risk Assessment

### 13.1 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Conversion tool has dependency/compatibility issues | HIGH | MEDIUM | Use QCFS standalone code (minimal deps); have SpikingJelly as backup |
| Specific architecture fails to convert | MEDIUM | LOW | Always have VGG-16/ResNet-18 as fallback; failure itself is a result |
| GPU access insufficient for ImageNet | MEDIUM | LOW | Focus on CIFAR-10/100; ImageNet is optional |
| Cannot reproduce paper results | MEDIUM | MEDIUM | Start with repos that claim reproducibility (QCFS has fixed seeds) |
| Accuracy loss is too high for practical use | LOW | LOW | This is itself a finding worth reporting |
| Not enough novelty for thesis | LOW | LOW | A systematic comparison IS the contribution; no need to invent new methods |

### 13.2 Confidence Assessment

| Finding | Confidence Level |
|---------|-----------------|
| ANN-to-SNN conversion is well-supported by tools | HIGH -- multiple mature frameworks exist |
| CIFAR-10 VGG-16 conversion achieves >93% at T<=16 | HIGH -- reproduced in many papers |
| ImageNet conversion is feasible at T<=32 | HIGH -- multiple papers confirm |
| MobileNet/EfficientNet conversion is underexplored | HIGH -- searched extensively, very few results |
| Medical imaging conversion is underexplored | HIGH -- no dedicated conversion studies found |
| A working pipeline can be set up in 1-2 weeks | MEDIUM-HIGH -- depends on environment and GPU |
| An undergrad can produce a meaningful thesis on this | HIGH -- clear scope, reproducible tools, identifiable contributions |

---

## 14. Sources

### Key Papers

1. [Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks (QCFS)](https://arxiv.org/abs/2303.04347) - ICLR 2022
2. [SNN Calibration (ICML 2021)](https://proceedings.mlr.press/v202/jiang23a/jiang23a.pdf)
3. [Sign Gradient Descent-based Neuronal Dynamics: ANN-to-SNN Conversion Beyond ReLU Network (ICML 2024)](https://arxiv.org/abs/2407.01645)
4. [Differential Coding for Training-Free ANN-to-SNN Conversion (ICML 2025)](https://openreview.net/forum?id=OxBWTFSGcv)
5. [Inference-Scale Complexity in ANN-SNN Conversion (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.html)
6. [Towards High-performance Spiking Transformers from ANN to SNN Conversion (ICLR 2025)](https://arxiv.org/abs/2502.21193)
7. [SpikeYOLO: Integer-Valued Training for Object Detection (ECCV 2024)](https://github.com/BICLab/SpikeYOLO)
8. [Optimal ANN-SNN Conversion with Group Neurons (ICASSP 2024)](https://github.com/Lyu6PosHao/ANN2SNN_GN)
9. [One-Timestep is Enough: Scale-and-Fire Neurons (2025)](https://arxiv.org/pdf/2510.23383)
10. [A Unified Optimization Framework of ANN-SNN Conversion (ICML 2023)](https://proceedings.mlr.press/v202/jiang23a/jiang23a.pdf)
11. [Going Deeper in Spiking Neural Networks: VGG and Residual Architectures (2019)](https://arxiv.org/pdf/1802.02627)
12. [Are SNNs Really More Energy-Efficient than ANNs? (Hardware-Aware Study)](https://cea.hal.science/cea-03852141/file/Are_SNNs_Really_More_Energy_Efficient_Than_ANNs__An_In_Depth_Hardware_Aware_Study_versionacceptee.pdf)
13. [SEENN: Towards Temporal Spiking Early-Exit Neural Networks (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/c801e68207da477bbc44182b9fac1129-Paper-Conference.pdf)
14. [A universal ANN-to-SNN framework (Neural Networks 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024001680)
15. [Adversarially Robust SNN Through Conversion (TMLR 2024)](https://github.com/IGITUGraz/RobustSNNConversion)
16. [SNN and Sound: Comprehensive Review (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/)
17. [Analog Spiking U-Net for Medical Image Segmentation (Neural Networks 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024006890)
18. [CS-QCFS: Bridging Performance Gap in Ultra-Low Latency SNNs (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024010050)

### Tool and Framework Sources

19. [snn_toolbox GitHub Repository](https://github.com/NeuromorphicProcessorProject/snn_toolbox)
20. [snn_toolbox Documentation](https://snntoolbox.readthedocs.io/en/latest/guide/intro.html)
21. [SpikingJelly GitHub Repository](https://github.com/fangwei123456/spikingjelly)
22. [SpikingJelly ann2snn Documentation](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.12/clock_driven_en/5_ann2snn.html)
23. [SpikingJelly ResNet18 CIFAR-10 Example](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/ann2snn/examples/resnet18_cifar10.py)
24. [snnTorch Documentation](https://snntorch.readthedocs.io/)
25. [snnTorch GitHub Repository](https://github.com/jeshraghian/snntorch)
26. [ANN-to-SNN-Conversion-with-snnTorch Demo](https://github.com/Saad-data/ANN-to-SNN-Conversion-with-snnTorch)
27. [QCFS Code Repository](https://github.com/putshua/ANN_SNN_QCFS)
28. [SNN_Calibration Code Repository](https://github.com/yhhhli/SNN_Calibration)
29. [SignGD Code Repository](https://github.com/snuhcs/snn_signgd)
30. [Differential Coding Code Repository](https://github.com/h-z-h-cell/ANN-to-SNN-DCGS)
31. [Open Neuromorphic - SpikingJelly Overview](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/spikingjelly/)
32. [SNN Library Benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)

### Curated Lists

33. [Awesome-Spiking-Neural-Networks (TheBrainLab)](https://github.com/zhouchenlin2096/Awesome-Spiking-Neural-Networks)
34. [awesome-snn-conference-paper](https://github.com/AXYZdong/awesome-snn-conference-paper)
35. [SNN-Daily-Arxiv](https://github.com/SpikingChen/SNN-Daily-Arxiv)
36. [ann-to-snn-conversion GitHub Topic](https://github.com/topics/ann-to-snn-conversion)
37. [awesome-snn Collection](https://github.com/coderonion/awesome-snn)

### Survey Papers

38. [Toward Large-scale Spiking Neural Networks: A Comprehensive Survey (2024)](https://arxiv.org/pdf/2409.02111)
39. [SpikingJelly: An open-source machine learning infrastructure platform (Science Advances 2023)](https://www.science.org/doi/10.1126/sciadv.adi1480)
40. [Exploring the potential of SNNs in biomedical applications (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362408/)
41. [A Practical Tutorial on Spiking Neural Networks (MDPI 2024)](https://www.mdpi.com/2673-4117/6/11/304)
