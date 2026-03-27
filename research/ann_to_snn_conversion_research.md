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
