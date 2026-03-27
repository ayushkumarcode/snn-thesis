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

