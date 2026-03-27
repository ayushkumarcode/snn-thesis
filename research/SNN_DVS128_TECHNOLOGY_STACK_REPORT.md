# Technology Stack for SNN DVS128 Gesture Recognition

Researching what i'd actually need to build an SNN-based DVS128 gesture recognition system from scratch. Going through the frameworks, data pipeline, training setup, etc.

The two main SNN frameworks are **SpikingJelly** (Peking University group, published in Science Advances) and **snnTorch** (UC Santa Cruz, Jason Eshraghian). Both sit on top of PyTorch and support surrogate gradient backprop for training deep SNNs.

SpikingJelly has a more complete built-in pipeline for DVS128 Gesture -- its own dataset loader, a pre-built DVSGestureNet model, full training scripts. snnTorch is more modular and tutorial-driven, and integrates with Tonic for neuromorphic data loading. For a thesis, SpikingJelly gets you to a working baseline faster, but snnTorch has better educational resources and more flexible visualization.

The DVS128 Gesture dataset has 1176 training and 288 test samples across 11 gesture classes, recorded with a Dynamic Vision Sensor at 128x128 resolution. Raw data is AEDAT 3.1 format -- polarity events (x, y, timestamp, polarity). Events need to be converted to frame tensors for batch training, and both frameworks handle this.

Training on a single GPU (RTX 2080 Ti, 12GB VRAM) takes about 18-28 seconds per epoch, with 64-256 epochs to reach ~96-97% accuracy. Apple M-series Macs work via PyTorch MPS backend, but you lose the CuPy/Triton acceleration. Energy can be estimated without neuromorphic hardware using the **syops** library (computes SynOps, estimates energy at 45nm technology costs).

---

## SpikingJelly Framework

### Version Requirements

| Dependency | Version |
|---|---|
| Python | >= 3.6 (recommended: 3.9 - 3.11) |
| PyTorch | >= 2.2.0 (tested on 2.7.1) |
| torchvision | Required (version matching PyTorch) |
| torchaudio | Required (version matching PyTorch) |
| numpy | Required |
| scipy | Required |
| tensorboard | Required |
| einops | Required |
