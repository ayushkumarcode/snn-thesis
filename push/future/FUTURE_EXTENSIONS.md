# Future Extensions — Ideas for Later
*16 March 2026*

These are NOT urgent. The core project is complete. Come back to these
if there's time before thesis submission or for a follow-up session.

## Analysis Extensions (no training needed)

### 1. Per-Class SNN vs ANN Analysis
Which sound classes does the SNN get RIGHT that the ANN gets WRONG?
One script, zero training. Could reveal that spiking neurons handle
transient sounds (clicks, knocks) better than sustained (rain, wind).
Good thesis figure.

### 2. Per-Class SpiNNaker vs snnTorch Analysis
Same idea but for hardware: which classes transfer well to SpiNNaker
vs which degrade? Already have 5-fold data, just need to aggregate
per-class accuracy across SpiNNaker runs.

### 3. Inference Speed Comparison (wall-clock)
Time 400 forward passes for SNN vs ANN on MPS.
Produces: "SNN takes X ms/sample, ANN takes Y ms/sample."
Trivial to implement, concrete number for the thesis.

### 4. Model Size / Memory Footprint
Both are ~622K params but SNN has binary activations (1-bit).
Calculate deployment memory footprint under different quantization
scenarios. Good for the hardware deployment narrative.

### 5. ESC-50 SOTA Context Table
Where our results sit in the broader landscape:
- Human: 81.3% (Piczak 2015)
- Our SNN: 47.15%
- Our ANN: 63.85%
- Our PANNs+SNN: 92.5%
- SOTA: 99.1% (OmniVec2, CVPR 2024)
No new experiments, just a formatted table.

## Training Extensions (need GPU)

### 6. UrbanSound8K Cross-Dataset
Script exists (`experiments/urbansound8k_1fold.py`).
Needs the dataset downloaded. Would validate encoding hierarchy
generalizes beyond ESC-50. Listed as future work in ICONS paper.

### 7. Surrogate Ablation Multi-Seed
3-seed run was submitted to CSF3 but never retrieved.
Would strengthen the bimodal finding (learning vs failure surrogates).

### 8. SNN with Different Architecture
Wider (128 channels), deeper (3 conv layers), or residual connections.
Would answer: "is the SNN-ANN gap architecture-dependent?"

### 9. Spike Rate Regularization 5-Fold
Pareto frontier is fold-1 only. 5-fold would validate the
0.62% spike rate finding (below energy break-even threshold).

## SpiNNaker Extensions

### 10. Full FC1+FC2 with Retrained FC2
Train FC2 on the pruned FC1 hidden spike patterns (in software),
then deploy the retrained FC2 on SpiNNaker. Addresses the core
issue: FC2 was trained for full FC1 output, not pruned FC1.

### 11. SpiNNaker Latency/Energy Measurement
Script exists (`experiments/spinnaker_latency_energy.py`).
Would give real hardware energy numbers instead of theoretical.

### 12. PANNs+SNN Head on SpiNNaker
Deploy the 3-layer SNN head (from PANNs transfer learning) on
SpiNNaker. Small network (2048→512→256→50), should fit easily.
Would complete the "hybrid edge deployment" narrative.

## Paper Extensions

### 13. DCASE 2026 Submission
Environmental sound is exactly DCASE's focus. Deadline ~July 2026.
Could be a second paper with different angle (environmental sound
benchmark focus vs ICONS neuromorphic systems focus).

### 14. ICASSP 2027 Submission
Signal processing angle. Deadline ~September 2026.
Reframe around audio processing methodology.
