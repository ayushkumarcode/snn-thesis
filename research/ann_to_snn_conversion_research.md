# ANN-to-SNN Conversion: Could This Work as a Thesis?

looked into ANN-to-SNN conversion as a thesis direction. the basic idea: take a pre-trained ANN, replace ReLU activations with integrate-and-fire neurons, normalize thresholds, and run inference where spike rates encode activation values. it's the **cheapest way** to get high-accuracy SNNs because you leverage the mature ANN training ecosystem.

this is a solid undergrad thesis direction. conversion pipeline is well-supported by existing tools (SpikingJelly, snn_toolbox, snnTorch, standalone paper implementations), core experiments are reproducible within weeks, and there are clear contribution opportunities. actively producing top-venue papers (ICML 2024/2025, CVPR 2025, NeurIPS 2023, ECCV 2024) with open-source code.

strongest thesis framing: "Evaluating the Practicality of ANN-to-SNN Conversion for [Specific Domain/Architecture]" -- where the domain is something not yet well-studied (medical imaging, audio, lightweight architectures like MobileNet/EfficientNet, or a head-to-head tool comparison).

---

## State of the Art (2024-2026)

### Evolution of Conversion

**Phase 1 (2015-2019): Basic Rate Coding**
- replace ReLU with IF neurons, normalize weights/thresholds
- needed 500-2500+ timesteps
- limited to VGG-like stuff on CIFAR-10/MNIST

**Phase 2 (2020-2023): Optimized, Reduced Latency**
- threshold balancing, weight normalization, calibration
- down to 32-256 timesteps with good accuracy
- extended to ResNets, deeper architectures, ImageNet

**Phase 3 (2024-2026): Ultra-Low Latency and Beyond-CNN**
- 1-8 timesteps achieving near-ANN accuracy
- first Transformer-to-SNN conversions
- first non-ReLU architectures converted (ConvNeXt, MLP-Mixer, ResMLP)
- training-free conversion methods
- object detection, segmentation, video classification

### Landmark Papers

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| Sign Gradient Descent Neuronal Dynamics | ICML 2024 | First to convert ConvNeXt, MLP-Mixer, ResMLP (beyond ReLU) |
| Optimal ANN-SNN with Group Neurons | ICASSP 2024 | ResNet-34 on ImageNet: 73.61% at T=2 |
| SpikeYOLO | ECCV 2024 (Best Paper Candidate) | Integer-valued training + spike-driven detection |
| Inference-Scale Complexity | CVPR 2025 | Training-free conversion; classification + segmentation + detection + video |
| Differential Coding | ICML 2025 | Novel differential coding reduces spike counts and energy |
| Spiking Transformers | ICLR 2025 | 88.60% top-1, only 1% loss at T=4 |
| One-Timestep is Enough | 2025 | Scale-and-Fire neurons at T=1 |

### Current Frontiers

1. ultra-low latency (T=1-4) -- the holy grail
2. transformer conversion -- beyond CNNs
3. beyond-ReLU (GELU, SiLU, Swish)
4. training-free conversion -- just convert and run
5. domain expansion (detection, segmentation, video, NLP)
6. energy-accuracy co-optimization
7. adaptive inference -- dynamic timesteps per input

---

## Tools and Frameworks

### Comparison

| Feature | snn_toolbox | SpikingJelly (ann2snn) | snnTorch | Custom Paper Code |
|---------|------------|----------------------|---------|-------------------|
| **Input framework** | Keras, PyTorch, Caffe, Lasagne | PyTorch | PyTorch | Usually PyTorch |
| **Conversion method** | Weight norm + threshold balancing | MaxNorm / RobustNorm / Scaling | Basic IF replacement | Method-specific |
| **Hardware deployment** | SpiNNaker, Loihi | Limited | No | Varies |
| **Docs** | Good (ReadTheDocs) | Good (English + Chinese) | Excellent (tutorials, Colab) | Varies (often minimal) |
| **Learning curve** | Moderate | Low | Low | High (read the paper) |
| **Maintenance** | Low activity (~387 stars) | Active (Science Advances pub) | Active | Depends |
| **Best for** | Multi-backend, SpiNNaker/Loihi | Fast prototyping, research | Learning, education | SOTA results |

### snn_toolbox

- repo: https://github.com/NeuromorphicProcessorProject/snn_toolbox
- pipeline: Load ANN -> Parse -> Normalize -> Convert -> Simulate
- known issues: Conv1D normalization problems, Keras compatibility, lower maintenance in 2024-25
- best if you need SpiNNaker or Loihi deployment

### SpikingJelly ann2snn

- repo: https://github.com/fangwei123456/spikingjelly
- published in Science Advances (2023)
- modes: MaxNorm, RobustNorm, Scaling
- pre-built examples: resnet18_cifar10.py, cnn_mnist.py
- up to 11x speedup over other frameworks at T=32
- best for research prototyping

### snnTorch

- repo: https://github.com/jeshraghian/snntorch
- great tutorials, Colab notebooks
- simpler conversion approach -- more educational
- best for learning fundamentals

### Standalone Paper Code (often the best option)

for actually getting SOTA results:

| Repository | Paper | Venue | Ease |
|-----------|-------|-------|------|
| `putshua/ANN_SNN_QCFS` | QCFS | ICLR 2022 | High |
| `yhhhli/SNN_Calibration` | Calibration | ICML 2021 | High |
| `snuhcs/snn_signgd` | SignGD beyond ReLU | ICML 2024 | Medium |
| `Lyu6PosHao/ANN2SNN_GN` | Group Neurons | ICASSP 2024 | Medium |
| `IGITUGraz/RobustSNNConversion` | Adversarial robustness | TMLR 2024 | Medium |
| `h-z-h-cell/ANN-to-SNN-DCGS` | Differential coding | ICML 2025 | Medium-High |
| `BICLab/SpikeYOLO` | SpikeYOLO | ECCV 2024 | Medium |

---

## Accuracy Loss During Conversion

### Where the Loss Comes From

three fundamental error sources:
1. **Quantization error**: continuous ReLU activations approximated by discrete spike counts
2. **Clipping error**: values exceeding firing threshold are lost
3. **Residual membrane potential**: info in membrane potential at end of simulation is discarded

more timesteps = lower accuracy loss, but higher latency and energy. classic trade-off.

### Concrete Numbers

#### CIFAR-10

| Method | Architecture | T | SNN Acc | ANN Acc | Loss |
|--------|-------------|---|---------|---------|------|
| QCFS (ICLR 2022) | VGG-16 | 4 | 93.05% | 93.63% | 0.58% |
| SNN Calibration (ICML 2021) | VGG-16 | 16 | 93.63% | 93.71% | 0.08% |
| SEENN (NeurIPS 2023) | VGG-16 | ~1.4 | 93.63% | -- | ~0 |
| One-Timestep (2025) | ResNet-18 | 1 | 93.11% | ~93.5% | ~0.4% |

#### ImageNet

| Method | Architecture | T | SNN Acc | Loss |
|--------|-------------|---|---------|------|
| Group Neurons (ICASSP 2024) | ResNet-34 | 2 | 73.61% | ~0% |
| QCFS + TPP | VGG-16 | 16 | 73.98% | ~0% |
| Spiking Transformer (2025) | ViT/DeiT | 4 | 88.60% | ~1% |

### Rules of Thumb

| Timesteps | Typical Loss (CIFAR-10) | Typical Loss (ImageNet) |
|-----------|------------------------|------------------------|
| T = 1 | 0-1% (modern methods) | 2-5% |
| T = 2-4 | 0-0.5% | 0-2% |
| T = 8-16 | Near-lossless | 0-1% |
| T = 32-64 | Lossless | Near-lossless |
| T >= 500 | Lossless (classical methods) | Lossless |

with modern methods (QCFS, calibration, group neurons), less than 1% loss at T=4-16 on CIFAR-10/100. ImageNet needs slightly more steps but is feasible at T=4-16 with latest methods.

---

## Which Architectures Convert Best

| Architecture | Difficulty | Key Challenges | Status (2025) |
|-------------|-----------|----------------|---------------|
| **VGG-16** | EASY | Pure Conv+ReLU+Pool, no skip connections | Fully solved |
| **ResNet-18/20/34** | MODERATE | Skip connections cause "deviation error" | Well-studied, good results with calibration |
| **MobileNet v1/v2** | MODERATE-HARD | Depthwise separable convs, squeeze-excite | Limited work, **gap opportunity** |
| **EfficientNet** | HARD | Compound scaling, Swish/SiLU (not ReLU), SE blocks | Very few results, **significant gap** |
| **ConvNeXt** | HARD | GELU, LayerNorm | First converted in ICML 2024 |
| **Vision Transformer** | VERY HARD | Softmax, LayerNorm, GELU, multi-head attention | First successful in 2025 |
| **YOLO** | HARD | Multi-scale, NMS, detection heads | SpikeYOLO (ECCV 2024) |

VGG converts easiest because: standard Conv2D+ReLU+MaxPool only, no skip connections, BatchNorm folds cleanly, sequential structure maps naturally to spiking dynamics.

ResNets are harder because the skip connection addition after the last ReLU creates "deviation error" -- spike-based approximation of adding two activations introduces systematic bias.

the big recent breakthrough is converting non-ReLU architectures (ConvNeXt with GELU, Transformers with GELU+Softmax+LayerNorm). the SignGD paper (ICML 2024) was first to do this. still a frontier area.

---

## Timestep Requirements

### Progression by Era

| Method Generation | Typical Timesteps | Years |
|------------------|-------------------|-------|
| Classical (weight norm) | 500-2500 | 2015-2019 |
| Improved (calibration) | 32-256 | 2020-2022 |
| Modern (QCFS) | 4-32 | 2022-2024 |
| Cutting-edge (one-timestep) | 1-4 | 2024-2026 |

### Practical Advice

for a thesis:
- **start with T=32-64** using standard method (SNN Calibration or snn_toolbox) -- this will reliably work
- **then reduce to T=8-16** with QCFS or similar
- **optionally try T=1-4** if time permits

this progression itself can be a thesis contribution: "how does accuracy degrade as we reduce timesteps?"

interesting note: on CIFAR-10, going from T=2 to T=6 only improves accuracy by 0.34% while tripling latency. diminishing returns curve is a useful result to reproduce.

---

## Contribution Opportunities

### Tier 1: High Feasibility (Recommended)

**A: Head-to-Head Tool Comparison**
- convert same pretrained models using 3+ different tools
- compare accuracy, timesteps needed, conversion time, ease of use, docs
- datasets: CIFAR-10, CIFAR-100 (maybe ImageNet)
- include energy estimation
- **no published paper does this.** tool comparison papers get cited a lot.
- ~6-8 weeks

**B: Convert a Domain Nobody Has Converted**
- medical image classification (ResNet-18 on skin lesions, chest X-rays -- huge practical impact, almost unstudied for conversion)
- audio keyword spotting
- satellite/remote sensing imagery
- DVS128 gesture recognition (convert ANN on frame-binned DVS, compare with direct-trained SNN)
- ~4-6 weeks (pretrained ANNs likely on HuggingFace)

**C: Architecture Comparison**
- convert VGG-16, ResNet-18, MobileNetV2, EfficientNet-B0, DenseNet-121 using same method
- document which layers cause problems, accuracy loss, timesteps needed
- **MobileNet and EfficientNet conversion is severely underexplored** -- depthwise separable convolutions and Swish create known difficulties
- most conversion papers only test VGG + ResNet
- ~6-10 weeks

### Tier 2: Stronger Contribution

**D: Energy Analysis**
- convert models, measure spike counts/sparsity at different timesteps
- use syops library for energy estimation
- compare theoretical vs actual GPU energy measurements
- energy claims in SNN papers are often unverified

**E: Hybrid (Conversion + Direct Training)**
- convert ANN to SNN, then fine-tune with surrogate gradients for a few epochs
- measure whether this recovers accuracy lost during conversion
- growing research direction

### Tier 3: Ambitious

**F: First Systematic MobileNet/EfficientNet Conversion**
- need to handle depthwise separable convolutions
- need to address Swish/SiLU for EfficientNet
- could use SignGD (ICML 2024) as starting point
- genuinely novel, potential workshop paper

---

## Getting a Working Pipeline

### Timeline Estimates

| Phase | snn_toolbox | SpikingJelly | QCFS Code |
|-------|-----------|-------------|-----------|
| Setup | 1-2 days | 0.5-1 day | 0.5-1 day |
| Understanding | 2-3 days | 1-2 days | 1-2 days |
| First MNIST | 1 day | 1 day | N/A |
| First CIFAR-10 | 2-3 days | 1-2 days | 0.5 day |
| Tuning | 3-5 days | 2-3 days | 1-2 days |
| **Total** | **~2 weeks** | **~1 week** | **~3-5 days** |

### Hardware

- minimum: 6GB VRAM GPU for CIFAR
- recommended: 8-12GB for CIFAR + small ImageNet
- full ImageNet: 16GB+ or university cluster
- CPU-only: possible for MNIST but painful (10-100x slower)

### Tips

1. use QCFS code as baseline: `python main_train.py --epochs=300 -dev=0 -L=4 -data=cifar10`, then test: `python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=8 -dev=0`
2. use pretrained weights when available (most repos provide them)
3. start with VGG-16 on CIFAR-10 -- easiest conversion target, verifies your pipeline
4. then ResNet-18/20 to see how skip connections affect things
5. use CIFAR-100 to stress-test -- accuracy gaps show up more on harder datasets

---

## Research Gaps

### Suitable for Undergrad

| Gap | Difficulty | Publication Potential |
|-----|-----------|---------------------|
| No systematic tool comparison | LOW | Workshop paper |
| MobileNet/EfficientNet barely studied | MODERATE | Workshop/conference |
| Medical imaging almost unstudied for conversion | LOW-MODERATE | Domain-specific venue |
| Audio/keyword spotting with modern methods | MODERATE | Workshop paper |
| Accuracy-vs-energy with actual measurements | MODERATE | Good thesis chapter |
| Converting Swish/GELU/SiLU architectures | MODERATE-HARD | Conference paper |
| DenseNet conversion not published | MODERATE | Short paper |

### PhD-Level

- novel neuron models for single-timestep conversion
- theoretical conversion error bounds for new architectures
- hardware co-design for specific neuromorphic chips
- scaling to billion-parameter models

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Tool dependency/compatibility issues | HIGH | Use QCFS standalone; SpikingJelly as backup |
| Architecture fails to convert | MEDIUM | VGG-16/ResNet-18 as fallback; failure is a result |
| Not enough GPU for ImageNet | MEDIUM | Focus on CIFAR; ImageNet is optional |
| Can't reproduce paper results | MEDIUM | Start with repos that claim reproducibility (QCFS has fixed seeds) |
| Accuracy loss too high | LOW | This is itself a finding worth reporting |
| Not enough novelty | LOW | Systematic comparison IS the contribution |

---

## Confidence

| Finding | Confidence |
|---------|-----------|
| Conversion well-supported by tools | HIGH |
| CIFAR-10 VGG-16 gets >93% at T<=16 | HIGH -- reproduced in many papers |
| ImageNet feasible at T<=32 | HIGH |
| MobileNet/EfficientNet conversion underexplored | HIGH |
| Medical imaging conversion underexplored | HIGH |
| Working pipeline in 1-2 weeks | MEDIUM-HIGH |
| Undergrad can produce meaningful thesis | HIGH |

---

## Sources

### Key Papers

1. [QCFS (ICLR 2022)](https://arxiv.org/abs/2303.04347)
2. [SNN Calibration (ICML 2021)](https://proceedings.mlr.press/v202/jiang23a/jiang23a.pdf)
3. [SignGD Beyond ReLU (ICML 2024)](https://arxiv.org/abs/2407.01645)
4. [Differential Coding (ICML 2025)](https://openreview.net/forum?id=OxBWTFSGcv)
5. [Inference-Scale Complexity (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.html)
6. [Spiking Transformers (ICLR 2025)](https://arxiv.org/abs/2502.21193)
7. [SpikeYOLO (ECCV 2024)](https://github.com/BICLab/SpikeYOLO)
