# Spike Encoding Methods: Could a Comparison Study Work as a Thesis?

i went deep into spike encoding methods for SNNs -- how you convert real-valued data into spike trains -- and whether doing a comparison study would be a viable thesis topic.

the key finding: several comparison studies exist, but none of them are truly complete. each one compares a subset of encodings on a narrow set of tasks (usually just MNIST/Fashion-MNIST, or one sensor modality). nobody has done a proper comparison of all major encoding methods across multiple data types (images, audio, time-series, event-driven) using a unified framework with consistent evaluation metrics. that gap is genuine and achievable for an undergrad thesis.

encoding choice demonstrably matters -- accuracy differences of 3-5% between methods on the same task are common, while latency and energy can differ by 4-7.5x. this is not a trivial question with a known answer.

---

## Complete Taxonomy of Spike Encoding Methods

based on Auge, Hille, Mueller, and Knoll (2021) in Neural Processing Letters plus other sources.

### Rate-Based Encoding

information is in firing frequency. robust against noise, simple to implement, but needs many timesteps and lots of spikes (energy-expensive).

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Poisson Rate Coding** | Each input value = probability of spike at each timestep (Bernoulli process). Higher value = more spikes on average. | Most common baseline; stochastic; high spike count |
| **Regular Rate Coding** | Deterministic variant, spikes evenly spaced with frequency proportional to input. | Lower variance; easier to analyze |
| **Population Rate Coding** | Group of neurons collectively encodes a value through combined firing rate. | Higher info capacity; uses more neurons |

### Temporal/Latency-Based Encoding

information is in precise timing of spikes. a single spike carries way more meaning than in rate codes. much fewer spikes needed, but more susceptible to noise.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Time-to-First-Spike (TTFS)** | Each neuron fires exactly once. Stronger inputs fire earlier. Based on LIF RC model. | Very low spike count; fast inference; ~4x lower latency than rate |
| **Rank-Order Coding** | Only relative ordering of spike times matters, not absolute times. | Robust to time distortions; loses amplitude info |
| **Inter-Spike Interval (ISI)** | Information in time gap between consecutive spikes from same neuron. | Compact; good for periodic signals |

### Delta Modulation / Temporal Contrast

event-driven encoding that generates spikes only when input changes by more than a threshold. inspired by how biological retinas and DVS cameras work.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Simple Delta** | Spike when diff between consecutive timesteps exceeds threshold. Optional "off-spikes" for decreases. | Natural for time-series; very sparse; event-driven |
| **Multi-Threshold Delta** | Multiple threshold levels for finer-grained encoding. | Better reconstruction; more spikes |
| **Sigma-Delta Modulation** | Accumulates error (sigma), spikes when accumulated error exceeds threshold (delta). | Lower quantization error; hardware-efficient |

### Phase Coding

input encoded in spike patterns whose phases correlate with internal oscillations (inspired by hippocampal theta oscillations).

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Phase Coding** | Input features determine phase offset relative to global oscillator. Higher values = earlier phase spikes. | Best noise resilience; periodic; highest SOP cost |

### Burst Coding

information through rapid successive bursts within short time windows.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Burst Coding** | Number of spikes in burst proportional to input strength. More reliable synaptic communication than single spikes. | Best fault tolerance; best compression; higher spike count than TTFS |

### Population Coding with Gaussian Receptive Fields (GRF)

each scalar input projected onto a population of neurons with different Gaussian receptive field centres. neuron closest to input fires earliest/most.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **GRF Population Coding** | N neurons cover input range with overlapping Gaussians. Activation determines spike timing. | High info capacity; requires multiple neurons per input feature |

### Direct / Learned Encoding

trainable layer converts raw input into spike trains. encoding learned jointly with network during training.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Direct Coding** | Trainable linear layer converts pixels to float values at each timestep; thresholding produces spikes. | Best accuracy with few timesteps; requires multi-bit first layer; less robust to adversarial attacks |
| **H-Direct (Homeostatic)** | Improved direct coding with homeostasis to prevent encoding collapse. | Addresses training efficiency limitations |

### Signal-Reconstruction-Oriented (for FPGA/hardware)

focused on accurate reconstruction from spike train, important for signal processing and hardware.

| Method | Description | Key Property |
|--------|-------------|--------------|
| **Step Forward (SF)** | Adjusts baseline threshold when signal crosses it. | Fastest encoding speed; lowest energy; unstable with abrupt transitions |
| **Ben's Spiker Algorithm (BSA)** | FIR filter deconvolution approach. | Good for square waves; very slow |
| **PWM** | Compare signal against sawtooth carrier wave. | Poor reconstruction accuracy |
| **Binary Encoding** | Multi-bit binary representation. | Best SNR (139dB with 10 bits); balanced noise resistance |

---

## Impact of Encoding Choice on Performance

### the impact is real and well-documented

the choice of encoding has a meaningful, measurable impact across every metric. this is not marginal.

### Accuracy

from Guo et al. (2021), on a 2-layer STDP-trained SNN:

| Encoding | MNIST Accuracy | Fashion-MNIST Accuracy |
|----------|---------------|----------------------|
| Rate | 87.46% | 68.29% |
| TTFS | 88.57% | 71.31% |
| Phase | 88.18% | 71.36% |
| Burst | 88.39% | 71.27% |

1-3% accuracy difference on MNIST, ~3% on Fashion-MNIST between rate and temporal methods. on deeper networks, Kim et al. (2022) found direct coding beats rate coding, especially with fewer timesteps (T=5-10).

from Bian et al. (2024), on IMU activity recognition:

| Encoding | Accuracy |
|----------|---------|
| Rate (Beta mapping) | 91.7% |
| TTFS (Log) | 89.2% |
| Binary (10-bit) | 89.6% |
| Multi-threshold Delta | 89.8% |

### Latency

from Guo et al. (2021):

| Encoding | Training Latency (ms) | Inference Latency (ms) |
|----------|----------------------|----------------------|
| Rate | 320 | 150 |
| TTFS | 80 | 20 |
| Phase | 90 | 30 |
| Burst | 60 | 30 |

TTFS needs **4x lower training latency and 7.5x lower inference latency** vs rate coding. that's a big deal.

### Synaptic Operations (Energy Proxy)

from Guo et al. (2021), SOPs x 10^8:

| Encoding | Training SOPs | Inference SOPs |
|----------|-------------|---------------|
| Rate | 130.785 | 9.932 |
| TTFS | 37.300 | 1.506 |
| Phase | 690.072 | 57.798 |
| Burst | 104.947 | 5.679 |

TTFS gets **3.5x fewer SOPs in training and 6.5x fewer in inference** vs rate. phase coding is the worst at ~5x MORE SOPs than rate. interesting trade-off.

### Noise Resilience

| Encoding | Input Noise Resilience | Synaptic Noise Tolerance |
|----------|----------------------|------------------------|
| Rate | Moderate | Poor (worst at training) |
| TTFS | Poor (worst) | Moderate |
| Phase | **Best** | Good |
| Burst | Poor | **Best** (at 20% fault rate) |

### Hardware Cost (NAND gates per module)

| Encoding | Hardware Cost |
|----------|-------------|
| Rate | 316N |
| TTFS | 340N + 1,703 (shared overhead) |
| Phase | 76N (simplest -- just muxes and 8-bit registers) |
| Burst | 544N (most expensive) |

### Bottom Line: No Single Best Encoding

each encoding creates different trade-offs:
- **TTFS**: best efficiency (latency + SOPs), worst noise resilience
- **Phase**: best noise resilience, simplest hardware, worst SOPs
- **Burst**: best fault tolerance and compression, most expensive hardware
- **Rate**: robust baseline, best adversarial robustness, highest latency/SOPs

this multi-dimensional trade-off space is exactly what makes a comparison thesis worth doing.

---

## Existing Comparison Studies

i went through every significant comparison study i could find and noted what each covers and what it leaves out.

**Study 1: Guo, Fouda, Eltawil, Salama (2021)** -- "Neural Coding in SNNs: A Comparative Study for Robust Neuromorphic Systems" (Frontiers in Neuroscience)
- Compared: Rate, TTFS, Phase, Burst
- Network: 2-layer SNN with STDP
- Datasets: MNIST, Fashion-MNIST
- Strengths: most complete multi-metric comparison i found, includes hardware analysis
- Gaps: only MNIST/Fashion-MNIST (image only), only STDP, no delta/direct/population, shallow network only
- Source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full

**Study 2: Kim, Park, Moitra et al. (2022)** -- "Rate Coding or Direct Coding: Which One is Better?" (ICASSP 2022)
- Compared: Rate (Poisson) vs Direct (trainable layer)
- Networks: MLP, VGG5, VGG9
- Datasets: MNIST, CIFAR-10, CIFAR-100
- Strengths: larger datasets, deeper architectures, adversarial robustness, code on GitHub
- Gaps: only 2 encoding methods, no temporal/phase/burst/delta/population
- Code: https://github.com/Intelligent-Computing-Lab-Panda/Rate-vs-Direct

**Study 3: Forno, Fra, Pignari, Urgese (2022)** -- "Spike encoding techniques for IoT time-varying signals" (Frontiers in Neuroscience)
- Compared: rate-based variants, temporal coding variants
- Datasets: Free Spoken Digit Dataset (audio), WISDM (IMU sensors)
- Strengths: multi-modal (audio + sensor), IoT-focused
- Gaps: limited encoding method coverage

**Study 4: Bian, Donati, Magno (2024)** -- "Evaluation of Encoding Schemes on Ubiquitous Sensor Signal" (arXiv)
- Compared: Rate variants, TTFS variants, Binary, Multi-threshold Delta
- Dataset: RecGym (IMU activity recognition)
- Strengths: most diverse metric set, includes deployment metrics, includes binary
- Gaps: single dataset (IMU only), no phase/burst/population

**Study 5: Plank et al. (2022)** -- "Evaluating Encoding and Decoding Approaches for Spiking Neuromorphic Systems" (ICONS 2022, ACM)
- Compared: Rate, Temporal, Population, Spike encoding
- Also compared decoding approaches (voting, first-to-spike, etc.)
- Tasks: classification, regression, control
- Hardware: Caspian neuromorphic processor
- Strengths: includes decoding (not just encoding), multi-task, actual neuromorphic hardware
- Gaps: specific to Caspian architecture

**Study 6: Vasilache et al. (2025)** -- "A PyTorch-Compatible Spike Encoding Framework" (arXiv)
- Compared: LIF, Step Forward, PWM, BSA
- Test signals: synthetic (vibration, trended, rectangular, sinusoidal)
- Strengths: open-source PyTorch framework, hardware-oriented
- Gaps: signal reconstruction only (no classification), synthetic signals only

**Study 7: IEEE Sensors Journal (2023)** -- "Comparison and Selection of Spike Encoding Algorithms for SNN on FPGA"
- Compared: sliding window, PWM-based, step-forward, BSA
- Focus: FPGA implementation
- Strengths: practical FPGA selection criteria
- Gaps: hardware-focused, no classification evaluation

### What's Missing (the gap)

| Gap | Description |
|-----|-------------|
| **No study compares ALL major methods** | Each covers 2-4; none includes rate + TTFS + delta + phase + burst + population + direct |
| **No cross-modality study** | Nobody tests encodings across images AND audio AND time-series AND events |
| **No modern deep SNN architectures** | Guo used 2-layer STDP; Kim used VGG. Nobody compares on ResNet-based or transformer-based SNNs |
| **No unified framework** | Different frameworks, neuron models, and hyperparams everywhere, so you can't compare across studies |
| **No snnTorch-based comparison** | snnTorch is the most popular educational framework but nobody benchmarks all its encoding options |
| **No encoding-decoding interaction analysis** | Only Plank et al. touched this, on limited hardware |
| **No GRF vs other methods** | GRF is well-documented but rarely compared head-to-head |

---

## Which Encoding for Which Data Type

| Data Type | Best Encoding(s) | Why | Evidence |
|-----------|-----------------|-----|----------|
| **Static images (MNIST, CIFAR)** | Rate (baseline), Direct (best accuracy) | Pixel intensities map naturally to firing rates; direct learns optimal conversion | Kim et al. 2022; Guo et al. 2021 |
| **Time-series sensor (IMU, IoT)** | Delta modulation, Rate with beta mapping | Delta captures changes; rate captures magnitude | Bian et al. 2024; Forno et al. 2022 |
| **Audio / speech** | Temporal contrast, cochlea-inspired | Audio is inherently temporal; cochlea models produce sparse spike trains | Forno et al. 2022; SHD dataset papers |
| **Event-driven (DVS cameras)** | Already spikes (no encoding needed), Delta for frame-based conversion | DVS data is natively event-driven | CIFAR10-DVS, DVS128 literature |
| **Noisy environments** | Phase coding | Highest resilience to input noise | Guo et al. 2021 |
| **Low-power / edge** | TTFS, Delta modulation | Fewest spikes = lowest energy | Guo et al. 2021; Bian et al. 2024 |
| **Hardware with faults** | Burst coding | Best fault tolerance at 20% fault rate | Guo et al. 2021 |
| **Real-time / low-latency** | TTFS, Direct coding (few timesteps) | TTFS fires once; direct works with T=5-10 | Guo et al. 2021; Kim et al. 2022 |

the "no free lunch" principle applies hard here. no single encoding wins on all dimensions. choice has to be guided by application priorities:
- accuracy? **Direct** or **TTFS**
- energy? **TTFS** or **Delta**
- noise robustness? **Phase**
- hardware reliability? **Burst**
- simplicity? **Rate**

this trade-off space is exactly what makes the comparison thesis valuable -- practitioners need actual guidance.

---

## Implementation in snnTorch

### Built-in Encodings (snntorch.spikegen)

snnTorch has three natively. phase, burst, GRF, and direct need custom implementation.

#### Rate Coding: `spikegen.rate()`

```python
import snntorch as snn
from snntorch import spikegen

spike_data = spikegen.rate(data_it, num_steps=100, gain=1.0)
# Output: [num_steps x batch x input_size]
# Each element is 0 or 1 (Bernoulli trial per timestep)
```

key params: `num_steps`, `gain` (scale factor), `offset`, `first_spike_time`, `time_var_input`

#### Latency/TTFS: `spikegen.latency()`

```python
spike_data = spikegen.latency(
    data_it, num_steps=100, tau=5,
    threshold=0.01, normalize=True, linear=True, clip=True
)
# Each neuron fires AT MOST once
```

key params: `tau` (RC time constant), `threshold` (min input), `normalize`, `linear` (vs log), `clip`

#### Delta Modulation: `spikegen.delta()`

```python
# data: [num_steps x batch x input_size] (time-series)
spike_data = spikegen.delta(data, threshold=0.1, padding=False, off_spike=True)
# Values are +1 (increase), -1 (decrease), or 0 (below threshold)
```

#### Target Encoding: `spikegen.targets_convert()`

```python
spike_targets = spikegen.targets_convert(
    targets, num_classes=10, code='rate',
    num_steps=100, correct_rate=0.8, incorrect_rate=0.2
)
```

### Custom Implementations Needed

these would need to be written as custom PyTorch functions. here are sketches:

#### Phase Coding

```python
def phase_encode(data, num_steps, num_phases=8):
    batch_size, input_size = data.shape
    spike_train = torch.zeros(num_steps, batch_size, input_size)
    period = num_steps // num_phases
    phase_offsets = ((1 - data) * (period - 1)).long()
    for t in range(num_steps):
        current_phase = t % period
        spike_train[t] = (current_phase == phase_offsets).float()
    return spike_train
```

#### Burst Coding

```python
def burst_encode(data, num_steps, max_burst_length=5, burst_gap=10):
    batch_size, input_size = data.shape
    spike_train = torch.zeros(num_steps, batch_size, input_size)
    burst_lengths = (data * max_burst_length).long().clamp(0, max_burst_length)
    num_windows = num_steps // burst_gap
    for w in range(num_windows):
        start = w * burst_gap
        for b in range(max_burst_length):
            t = start + b
            if t < num_steps:
                spike_train[t] = (b < burst_lengths).float()
    return spike_train
```

#### GRF Population Coding

```python
def grf_population_encode(data, num_steps, num_neurons_per_feature=10, tau=5, threshold=0.01):
    batch_size, input_size = data.shape
    n = num_neurons_per_feature
    centres = torch.linspace(0, 1, n)
    sigma = 1.0 / (2 * (n - 1))
    data_exp = data.unsqueeze(-1)
    centres_exp = centres.unsqueeze(0).unsqueeze(0)
    activations = torch.exp(-0.5 * ((data_exp - centres_exp) / sigma) ** 2)
    activations = activations.reshape(batch_size, input_size * n)
    spike_train = spikegen.latency(
        activations, num_steps=num_steps, tau=tau,
        threshold=threshold, normalize=True, linear=True
    )
    return spike_train
```

#### Direct Coding (Trainable)

```python
class DirectEncoder(nn.Module):
    def __init__(self, input_size, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.encoder = nn.Linear(input_size, input_size * num_steps, bias=True)

    def forward(self, x):
        batch_size, input_size = x.shape
        encoded = self.encoder(x)
        encoded = encoded.reshape(batch_size, self.num_steps, input_size)
        encoded = encoded.permute(1, 0, 2)
        spikes = (encoded > 0.5).float()
        spikes = spikes + encoded - encoded.detach()  # STE for gradient
        return spikes
```

### Other Frameworks with More Built-in Encodings

| Framework | Built-in Encodings | Notes |
|-----------|-------------------|-------|
| **snnTorch** | Rate, Latency, Delta | Best tutorials; PyTorch-based |
| **BindsNET** | Rate, Poisson, Rank-order, GRF/Binning | More bio-oriented; STDP focus |
| **SpikingJelly** | Rate, Latency, Direct, Poisson | More complete; better for deep SNNs; some Chinese docs |
| **Norse** | Current-based, LIF-based, custom | Lower-level; max flexibility |
| **Lava (Intel)** | Custom (hardware-oriented) | Loihi; production focus |

i'd use snnTorch as the primary framework since implementing the missing encodings is part of the contribution.

---

## Thesis Viability

### Is This Valid?

yes, pretty clearly. the evidence:

1. **Active research**: papers still being published in 2024-2025 (Bian et al. 2024, Vasilache et al. 2025), so the question isn't settled
2. **Acknowledged gap**: Plank et al. (2022): "it is not clear which is the most appropriate approach or whether the choice has a significant impact on performance." Bian et al. (2024): "a systematic approach to quantitatively evaluate spike encoding performance is currently lacking."
3. **Practical relevance**: every SNN practitioner must choose an encoding and there's no definitive guide
4. **Clear methodology**: systematic evaluation / benchmarking is well-understood in CS
5. **Publishable potential**: if broader than existing studies, could go to ICONS, NICE, or SNN workshops

### What Would Make It Novel vs What Already Exists

| Existing Work | Your Thesis Could Add |
|--------------|----------------------|
| 2-4 methods per study | 6-8 in one unified study |
| Single data modality | Multiple (image + audio + time-series) |
| MNIST/Fashion-MNIST only | MNIST + CIFAR-10 + SHD + sensor data |
| 2-layer STDP networks | Modern deep SNNs with surrogate gradients |
| Accuracy only or accuracy + 1-2 metrics | Accuracy, latency, spike count, energy proxy, noise robustness |
| Different frameworks per study | Unified snnTorch for everything |
| No practical guidelines | Decision framework / recommendation guide |

### Feasibility

| Aspect | Assessment |
|--------|-----------|
| Technical difficulty | Moderate. rate/latency/delta are trivial (built-in). phase/burst/GRF need custom code but are straightforward. direct needs basic NN knowledge. |
| Compute | Low-Moderate. MNIST on laptop in minutes. CIFAR-10 needs GPU but runs in hours. SHD manageable. No HPC needed. |
| Background knowledge | LIF model, surrogate gradients, basic signal processing. learnable in a few weeks from snnTorch tutorials. |
| Risk | Low. experiments are well-defined, reproducible, not dependent on external resources. worst case = confirming existing results (still valid as replication + extension). |
| Timeline | 8-12 weeks for core experiments (6 encodings x 3 datasets x 3 metrics). |

### Potential Concerns

| Concern | Response |
|---------|---------|
| "It's just a comparison" | "systematic evaluation" is a recognized contribution type in CS. you're creating knowledge about trade-offs that doesn't exist in one place. adding a decision framework adds originality. |
| "Guo et al. already did this" | they used STDP on MNIST with 4 methods. you'd use modern training (surrogate gradients), more datasets, more methods. overlap is partial. |
| "Results might be obvious" | they're not -- phase coding has best noise resilience but worst SOPs, TTFS has best efficiency but worst noise tolerance. these trade-offs are complex and data-dependent. |

---

## Research Gaps and Contribution Opportunities

### Primary Gaps (highest value)

1. **Unified cross-modality comparison**: test same 6+ encodings on image + audio + time-series using same architecture and training. nobody's done this.
2. **Encoding-architecture interaction**: does the best encoding change with network architecture (feedforward vs recurrent vs conv)? no systematic study on this.
3. **Practitioner decision framework**: a flowchart or decision matrix for choosing encoding based on data type, hardware constraints, performance priorities. doesn't exist.

### Secondary Gaps (nice extensions)

4. **Encoding + decoding interaction**: which combo works best? only Plank et al. partially addressed this.
5. **Impact on learning dynamics**: how does encoding affect convergence speed, loss curves, gradient flow?
6. **Sensitivity analysis**: how sensitive is each encoding to its hyperparameters (tau, threshold, etc.)?
7. **Information-theoretic analysis**: mutual information between input signal and spike train for each encoding.

### Stretch Goals

8. Deploy on Loihi or SpiNNaker and compare on real hardware
9. Propose a hybrid that combines strengths of multiple methods
10. Adaptive encoding that switches based on input characteristics

---

## Thesis Structure

### Title Ideas

- "A Systematic Evaluation of Spike Encoding Methods for Spiking Neural Networks Across Data Modalities"
- "Comparing Spike Encoding Strategies: Performance Trade-offs in Modern Spiking Neural Networks"
- "Which Spike Encoding? A Benchmark Study Across Tasks, Architectures, and Metrics"

### Experimental Matrix

| Encoding | MNIST | CIFAR-10 | SHD (Audio) | Sensor (TBD) |
|----------|-------|----------|-------------|---------------|
| Rate (Poisson) | X | X | X | X |
| Latency (TTFS) | X | X | X | X |
| Delta | X | X | X | X |
| Phase | X | X | X | X |
| Burst | X | X | X | X |
| GRF Population | X | X | X | X |
| Direct (Learned) | X | X | X | X |

= 7 encodings x 4 datasets x 5+ metrics x 3 repetitions = 420+ experiment runs

each MNIST/Fashion-MNIST run takes ~5-15 min. CIFAR-10 ~30-60 min. SHD ~15-30 min. total: roughly 50-100 GPU-hours, doable on a personal GPU or free Colab over several weeks.

---

## Key Papers

| # | Authors | Year | Title | Venue | Encodings Covered |
|---|---------|------|-------|-------|-------------------|
| 1 | Guo et al. | 2021 | Neural Coding in SNNs: A Comparative Study | Frontiers in Neuroscience | Rate, TTFS, Phase, Burst |
| 2 | Kim et al. | 2022 | Rate Coding or Direct Coding: Which One is Better? | ICASSP 2022 | Rate, Direct |
| 3 | Forno et al. | 2022 | Spike encoding for IoT time-varying signals | Frontiers in Neuroscience | Rate variants, Temporal variants |
| 4 | Bian et al. | 2024 | Evaluation of Encoding Schemes on Sensor Signal | arXiv | Rate, TTFS, Binary, Delta |
| 5 | Plank et al. | 2022 | Evaluating Encoding and Decoding for Neuromorphic Systems | ICONS (ACM) | Rate, Temporal, Population |
| 6 | Vasilache et al. | 2025 | PyTorch-Compatible Spike Encoding Framework | arXiv | LIF, SF, PWM, BSA |
| 7 | Auge et al. | 2021 | Survey of Encoding Techniques for SNNs | Neural Processing Letters | All major categories |
| 8 | Petro et al. | 2019 | Selection and Optimization of Temporal Spike Encoding | IEEE TNNLS | BSA, SF, SW, other temporal |

---

## Sources

### Survey Papers
- [Auge et al. 2021 - Encoding techniques survey](https://link.springer.com/article/10.1007/s11063-021-10562-2)
- [Guo et al. 2021 - Neural coding comparison](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.638474/full)
- [SNN in Imaging review (2025)](https://www.mdpi.com/1424-8220/25/21/6747)

### Encoding Comparison Studies
- [Kim et al. 2022 - Rate vs Direct](https://arxiv.org/abs/2202.03133) + [code](https://github.com/Intelligent-Computing-Lab-Panda/Rate-vs-Direct)
- [Forno et al. 2022 - IoT signals](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.999029/full)
- [Bian et al. 2024 - Sensor signals](https://arxiv.org/html/2407.09260v1)
- [Plank et al. 2022 - ICONS](https://dl.acm.org/doi/fullHtml/10.1145/3546790.3546792)
- [IEEE 2023 - FPGA encoding comparison](https://ieeexplore.ieee.org/document/10021878/)
- [Vasilache et al. 2025 - PyTorch framework](https://ar5iv.labs.arxiv.org/html/2504.11026)
- [Petro et al. 2019 - Temporal encoding](https://ieeexplore.ieee.org/document/8689349/)

### Rate vs Temporal / Direct
- [First-spike coding for SNNs (2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1266003/full)
- [Stochastic first-to-spike (2024)](https://arxiv.org/html/2404.17719v2)
- [H-Direct homeostatic encoding (2024)](https://openreview.net/forum?id=QkDUdPRcma)

### snnTorch
- [Tutorial 1: Spike Encoding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
- [spikegen API docs](https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html)
- [Population Coding Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html)
- [snnTorch GitHub](https://github.com/jeshraghian/snntorch)

### Datasets
- [SHD](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
- [CIFAR10-DVS](https://www.semanticscholar.org/paper/CIFAR10-DVS:-An-Event-Stream-Dataset-for-Object-Li-Liu/e72b7962133921fa3e84299cd6a4a2aeb60bab19)

### Frameworks
- [Open Neuromorphic Software Guide](https://open-neuromorphic.org/neuromorphic-computing/software/)
- [SNN framework benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)
- [BindsNET](https://github.com/BindsNET/bindsnet)

### General SNN Reviews
- [Comprehensive SNN review (2025)](https://link.springer.com/article/10.1007/s12530-025-09755-0)
- [SNNs for ubiquitous computing (2025)](https://arxiv.org/html/2506.01737v1)
- [Non-traditional encoding (Oak Ridge)](https://www.osti.gov/servlets/purl/1607189)

---

## Quick-Start Code Skeleton

```python
"""
Thesis Experiment Runner: Spike Encoding Comparison
"""

import torch
import snntorch as snn
from snntorch import spikegen, surrogate, functional
import time

# ---- Encoding Functions ----

def encode_rate(data, num_steps):
    return spikegen.rate(data, num_steps=num_steps)

def encode_latency(data, num_steps):
    return spikegen.latency(data, num_steps=num_steps, tau=5,
                            threshold=0.01, normalize=True, linear=True)

def encode_delta(data, num_steps):
    repeated = data.unsqueeze(0).repeat(num_steps, 1, 1)
    return spikegen.delta(repeated, threshold=0.1, off_spike=True)

# Phase, Burst, GRF, Direct: use custom implementations above

# ---- Metrics ----

def evaluate_encoding(encode_fn, data_loader, model, num_steps, device):
    correct = 0
    total = 0
    total_spikes = 0
    total_time = 0.0
    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            t_start = time.time()
            spike_data = encode_fn(data, num_steps)
            t_encode = time.time() - t_start
            t_start = time.time()
            spk_rec, mem_rec = model(spike_data)
            t_infer = time.time() - t_start
            _, predicted = spk_rec.sum(0).max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            total_spikes += spike_data.sum().item()
            total_time += t_encode + t_infer
    return {
        'accuracy': correct / total,
        'avg_spikes_per_sample': total_spikes / total,
        'total_time': total_time,
    }

# ---- Main ----

ENCODINGS = {
    'rate': encode_rate,
    'latency': encode_latency,
    'delta': encode_delta,
    # 'phase': encode_phase,
    # 'burst': encode_burst,
    # 'grf': encode_grf,
}

