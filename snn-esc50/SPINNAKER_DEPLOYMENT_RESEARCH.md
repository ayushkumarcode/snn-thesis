# SpiNNaker 1 deployment research notes

date: 27 march 2026
context: ESC-50 conv SNN (622K params), partial SpiNNaker deployment already working

---

went through 8 research areas for deploying energy-optimized SNN models on SpiNNaker 1. here's what i found:

1. rhythm/oscillatory neurons CAN run on SpiNNaker 1 -- either custom C neuron model (moderate difficulty), segmented sim.run() with i_offset updates (easy but coarse), or built-in Izhikevich resonator mode (easiest but different dynamics).

2. dendritic/multi-compartment neurons have been demonstrated on SpiNNaker 1 (Ward et al. 2022, open-source). 11x slower than LIF and memory-heavy though. our simpler multi-branch LIF (no HH channels) would be much lighter.

3. SpiNNaker 1 energy measurement is inherently estimation-based -- no on-chip power monitors. accepted method uses Stromatias et al. (2013) power model: ~1W/chip, 43-110 nJ/synaptic event.

4. conv layers on SpiNNaker 1 are supported via KernelConnector. our hybrid approach (conv on CPU, FC on SpiNNaker) remains most practical.

5. energy-optimized models (pruned, quantized, early-exit) map well to SpiNNaker.

6. competing hardware dramatically outperforms SpiNNaker 1: Xylo Audio 2 gets 6.6 uJ/inference vs our ~6.4 mJ. SpiNNaker 2 at ~7.1 uJ is competitive with Xylo. for our paper, SpiNNaker deployment shows feasibility and accuracy, not energy superiority.

---

## 1. rhythm/oscillatory neurons on SpiNNaker 1

### the problem

our RhythmLIF adds a learnable sinusoidal term:
```
v[t] = beta * v[t-1] * (1-spk[t-1]) + I[t] + A * sin(2*pi*f*t/T + phi)
```
where A, f, phi are per-neuron learnable parameters. gets 61.10% +/- 1.99% on ESC-50. can it run on SpiNNaker 1?

### option A: custom neuron model in C (recommended)

yes -- demonstrated framework exists. sPyNNaker has a modular neuron model architecture (input type, neuron model, synapse type, threshold type). official template at https://github.com/SpiNNakerManchester/sPyNNaker8NewModelTemplate

implementation approach:
1. fork the template
2. create new neuron model C file extending IF_curr_exp
3. add three state variables: amplitude (A), frequency (f), phase (phi)
4. compute sin using lookup table (ARM968 lacks hardware sin/cos)
5. add oscillatory current to standard LIF update
6. Python bindings inheriting from AbstractPyNNNeuronModelStandard
7. compile with GCC ARM toolchain

technical challenges:
- ARM968 cores lack hardware floating-point; all math is s16.15 fixed-point
- sin() needs a LUT -- 256-entry takes ~512 bytes per core
- per-neuron params (A, f, phi) consume 3 words (12 bytes) per neuron in DTCM
- with 255 neurons max: 255 * 12 = 3060 bytes extra, well within 32KB DTCM

difficulty: MEDIUM (2-3 days for someone familiar with C on SpiNNaker). should be faithful to snnTorch model. minimal energy impact -- sin LUT is one memory read per neuron per timestep.

source: sPyNNaker New Model Template, Creating New Neuron Models Lab Manual (http://spinnakermanchester.github.io/spynnaker/5.0.0/NewNeuronModels-LabManual.pdf)

### option B: segmented sim.run() with i_offset (easiest)

sPyNNaker lets you update `i_offset` between sim.run() calls:
```python
for t in range(NUM_STEPS):
    for neuron_id in range(pop_size):
        osc = A[neuron_id] * math.sin(2*pi*f[neuron_id]*t/T + phi[neuron_id])
        pop.set(i_offset=osc)
    sim.run(1.0)
```

problems: sim.run(1) called 25 times instead of sim.run(25) once. each call has ~50-200ms Python overhead. total ~1.25-5 seconds per sample vs ~25ms for single run. also i_offset is a population-level param -- can't easily set per-neuron (would need one population per neuron which is absurd).

verdict: only viable as proof-of-concept, not practical for inference.

### option C: Izhikevich resonator mode (built-in)

sPyNNaker supports Izhikevich natively. with resonator params (a=0.1, b=0.26) it produces subthreshold oscillations conceptually similar to RhythmLIF. but the oscillations are emergent from 2D dynamics, not explicit sinusoidal injection. frequency isn't directly controllable per-neuron. would need complete retraining.

easy to implement (built-in) but unknown accuracy impact and ~2x more computation than IF_curr_exp due to quadratic terms.

### option D: SpikeSourceArray oscillatory input (creative workaround)

inject oscillatory component as a synthetic spike train from separate SpikeSourceArray. pre-compute spike times that approximate sinusoidal current. problems: coarse approximation, uses extra neurons/synapses, need separate oscillator per target neuron (256 extra for 256 hidden). counterproductive for energy.

### recommendation

for the thesis: deploy standard LIF on SpiNNaker (already working at 33.1%). rhythm model is the SOFTWARE result (61.1%). acknowledge that custom C model deployment (option A) is feasible and cite the sPyNNaker framework. Ward et al. (2022) used sin LUTs consuming only 1.6KB for dendritic calcium -- same approach would work for us.

---

## 2. dendritic/multi-compartment neurons on SpiNNaker 1

### the problem

our DendriticLIF has K=3 branches with learnable decay and gating:
```
mem_branch_k[t] = beta_k * mem_branch_k[t-1] + w_k * input[t]
mem_soma[t] = sum_k(w_k * mem_branch_k[t])
spike if mem_soma > threshold
```
gets 61.65% +/- 3.69% (best overall).

### Ward et al. (2022) -- the key paper

"Beyond LIF Neurons on Neuromorphic Hardware," Frontiers in Neuroscience, DOI: 10.3389/fnins.2022.881598. open source: https://github.com/mollie-ward/beyondLIFNeurons

they implemented: single-compartment Hodgkin-Huxley neuron on SpiNNaker 1, and two-compartment neuron (soma + dendrite with calcium channels) on SpiNNaker 1 and SpiNNaker 2 prototype.

technical details:
- fixed-point s16.15 accum
- LUTs for exponentials: 12 KB for HH, 1.6 KB for dendritic calcium
- HH model: 8.34 us per neuron per timestep (vs 0.32 us for LIF = 26x slower)
- two-compartment: 12.91 us per neuron per timestep (40x slower than LIF)
- max error: 0.00314 mV for dendritic compartment

our DendriticLIF is MUCH simpler:

| feature | Ward et al. | our DendriticLIF |
|---------|-------------|------------------|
| compartments | 2 (soma + dendrite) | K branches (3) + soma |
| ion channels | Na+, K+, Ca2+ (HH) | NONE -- pure LIF |
| computation/neuron | 12.91 us | ~1-2 us (estimated) |
| exponentials needed | yes (gating variables) | no |
| LUT needed | yes (12KB + 1.6KB) | just sigmoid (~512 bytes) |
| params per neuron | ~20+ | 7 (3 betas, 3 gates, 1 thresh) |

per-neuron computation:
```c
// for each branch k (K=3):
beta_k = sigmoid(beta_logit_k);  // one LUT lookup
mem_k = beta_k * old_mem_k + gate_k * input;  // 2 mul + 1 add
soma += gate_k * mem_k;  // 1 mul + 1 add
// total: K * (1 LUT + 3 mul + 2 add) = 3 LUT + 9 mul + 6 add
// vs LIF: 1 mul + 1 add
// estimated ~3-5x LIF computation, easily within 1ms timestep
```

memory: K*2 params + K state = 9 extra words = 36 bytes per neuron. for 255 neurons: 9180 bytes, fits in 32KB DTCM.

implementation path: start from Ward's spinnaker_twoComp.c, strip out ALL HH equations, replace with simple branch LIF dynamics, add branch state variables, sigmoid via 256-entry LUT (512 bytes), precompute gate softmax on host.

difficulty: MEDIUM-HIGH (3-5 days). Ward code provides the infrastructure but needs significant modification.

### multi-population approximation (alternative)

instead of custom model, use separate IF_curr_exp populations for each branch with different tau_m:
```python
branch1 = sim.Population(256, sim.IF_curr_exp(tau_m=2.80))   # beta=0.7
branch2 = sim.Population(256, sim.IF_curr_exp(tau_m=9.49))   # beta=0.9
branch3 = sim.Population(256, sim.IF_curr_exp(tau_m=99.5))   # beta=0.99
soma = sim.Population(256, sim.IF_curr_exp(tau_m=20.0, v_thresh=1.0))
```

beta to tau_m: tau_m = -dt / ln(beta). problems: branch populations fire spikes to soma but our model uses continuous integration, branches leak independently rather than with coupled dynamics. 3x more neurons/cores.

easy (1-2 days) but inaccurate.

### recommendation

same as oscillatory: deploy standard LIF on SpiNNaker, report dendritic results from snnTorch. cite Ward et al. for feasibility.

citation for paper: "Ward et al. demonstrated multi-compartment neurons with HH dynamics on SpiNNaker 1, requiring 12.91 us per neuron per timestep. Our DendriticLIF uses purely linear branch dynamics without ion channels, requiring only ~3-5x the LIF computation."

---

## 3. SpiNNaker energy measurement

### the fundamental challenge

SpiNNaker 1 has NO on-chip power monitors. all energy numbers are either: (1) board-level measurements with external multimeter, (2) model-based estimates using Stromatias (2013), or (3) synaptic event counting + published per-event figures.

### published energy figures

| source | metric | value | method |
|--------|--------|-------|--------|
| Stromatias 2013 (IEEE IJCNN) | idle power/chip | ~1W at 1.2V | board measurement |
| Stromatias 2013 | per neuron | 100 nJ/neuron/ms | board measurement |
| Stromatias 2013 | per synaptic event | 43 nJ | board measurement |
| van Albada 2018 (Frontiers) | total energy/synop | 110 nJ | board measurement |
| van Albada 2018 | incremental energy/synop | 8 nJ | board measurement |
| Stromatias 2015 | DBN classification power | 0.3W | board measurement |

note on the 5.9 uJ/event in our code: the value in `spinnaker_latency_energy.py` comes from van Albada et al. 2018 who measured during a full cortical microcircuit on 6 boards at 0.1ms timesteps. it's inflated because: (1) 0.1ms timesteps = 10x overhead vs our 1ms, (2) model sparsely distributed across 6 boards (massive idle overhead), (3) measured entire propagation phase.

for our use case (small model, 1ms timestep, single chip), the appropriate figures are:
- Stromatias 2013: 20 nJ total / 8 nJ incremental per synop
- Sharp et al. 2012: 110 nJ per synop
- our code should use 20-110 nJ range, NOT 5.9 uJ

### per-inference energy for our model

method 1 (idle + incremental events):
- FC2-only: ~55.6 spikes/timestep * 25 = ~1,390 total
- synaptic events: 1,390 * 50 = ~69,500
- synaptic energy: 69,500 * 110 nJ = 7.65 mJ (total) or * 8 nJ = 0.56 mJ (incremental)
- idle: 1W * 25ms = 25 mJ
- total: ~25.6-32.6 mJ (dominated by idle power!)

method 2 (Stromatias model):
- neuron cost: 50 * 25 * 100 nJ = 0.125 mJ
- synaptic: 69,500 * 43 nJ = 2.99 mJ
- idle: 25 mJ
- total: ~28.1 mJ

the key insight: SpiNNaker 1 per-inference energy is completely dominated by 1W idle chip power. even with zero spikes, running 25ms costs 25 mJ. thats 55,000x more than NeuroBench's 454 nJ for the ANN. SpiNNaker 1 was designed for brain-scale simulation (millions of neurons), not single-sample inference.

### how to report energy in the paper

1. NeuroBench software energy as primary metric (968 nJ SNN, 454 nJ ANN)
2. SpiNNaker synaptic event counts from actual hardware
3. SpiNNaker energy estimate using Stromatias model, clearly noting idle-power dominance
4. compare with SpiNNaker 2 (~10 pJ/synop) and Xylo Audio 2 (6.6 uJ)
5. frame as: "on dedicated inference hardware (Xylo Audio 2), our model would consume approximately X uJ"

### SpiNNaker 2 comparison

| metric | SpiNNaker 1 | SpiNNaker 2 |
|--------|-------------|-------------|
| energy/synop | 8-110 nJ | ~10 pJ |
| power/chip | ~1W | ~0.1W (with DVFS) |
| MNIST energy | ~28 mJ (est.) | ~23 uJ (measured) |
| improvement | -- | ~1000x |

---

## 4. SpiNNaker full conv model deployment

### conv support in sPyNNaker

sPyNNaker supports convolutional connectivity through KernelConnector:
```python
kernel = np.array(trained_conv_weights)  # (out_ch, in_ch, kH, kW)
sim.Projection(pre_pop, post_pop, sim.KernelConnector(kernel, ...))
```

demonstrated in: Serrano-Gotarredona & Linares-Barranco (2015), Perez-Carrasco et al. (2019, Neural Networks -- used 103-chip system), Gronauer's SpikingConvNet (STDP training), Blouw et al. (2019).

### mapping our conv architecture

population sizes:
- input: 64*216 = 13,824 (flattened spectrogram)
- conv1 output (before pool): 32*64*216 = 442,368
- after MaxPool: 32*32*108 = 110,592
- conv2 output: 64*32*108 = 221,184
- after MaxPool: 64*16*54 = 55,296
- after AvgPool: 64*4*9 = 2,304
- FC1: 256
- FC2: 50

total neurons: ~843,000. core budget at 255/core: conv1 alone needs 434 cores = 24 chips. this is way beyond a single chip.

### why hybrid is the right approach

our hybrid (conv on CPU, FC on SpiNNaker) is standard and correct:
1. conv layers are compute-intensive but not naturally spiking (weight sharing, pooling, BN)
2. FC layers are naturally event-driven
3. conv unrolled = 442K+ neurons, impractical on SpiNNaker 1
4. FC1+FC2 = only 306 neurons, fits on a single core

literature consensus: most SpiNNaker CNN deployments use hybrid or very small conv layers. Perez-Carrasco needed 103 chips for full conv.

could technically run conv on ARM cores but ARM968 at 200 MHz with 32KB DTCM is hopelessly slow for conv operations. no SIMD, no vector instructions. not practical.

SpiNNaker 2 has hardware MAC accelerators and 2D convolution support. NIR-SpiNNaker2 (arXiv:2504.06748) demonstrated DVS gesture with conv layers on-chip.

---

## 5. deploying energy-optimized models

### pruned models

pruning maps directly to fewer connections in FromListConnector. 90% pruning of FC2 (256->50) goes from 12,800 to 1,280 connections. fewer connections = less DTCM per core = more neurons per core = less router traffic.

```python
pruned_conns = [(pre, post, w, d) for (pre, post, w, d) in conns if abs(w) > threshold]
sim.Projection(input_pop, output_pop, sim.FromListConnector(pruned_conns))
```

note from docs: "FromListConnector will result in slower operation of the tools" vs AllToAllConnector. but overhead is in setup, not runtime.

### quantized/binary/ternary weights

SpiNNaker 1 uses s16.15 fixed-point natively. further quantization options:
- INT8: possible via custom synapse models, limited benefit since already fixed-point
- binary {-1, +1}: add/subtract instead of multiply. on ARM968 thats ~3-5x faster per synapse
- ternary {-1, 0, +1}: same as binary but zero weights pruned

### early exit on SpiNNaker

achievable via segmented sim.run():
```python
for t in range(MAX_TIMESTEPS):
    sim.run(1.0)
    v_data = output_pop.get_data("v")
    confidence = compute_confidence(v_data)
    if confidence > THRESHOLD and t >= MIN_TIMESTEPS:
        break
```

challenge: each sim.run(1) + get_data() has ~50-200ms Python overhead. for 25 timesteps thats 1.25-5 seconds, dwarfing the 25ms actual sim time.

better approach: run sim.run(T_min) for minimum timesteps (eg T=7), read data, check confidence, if not confident run sim.run(T_max - T_min). only 2 calls instead of T_max. moderate implementation effort, 60-72% energy saving.

### reduced timesteps

simplest optimization. just change sim.run(25) to sim.run(7). everything else stays the same. trivial implementation, linear energy reduction (3.6x).

---

## 6. SpiNNaker vs other neuromorphic hardware for audio

| platform | type | energy/inference (audio KWS) | notes |
|----------|------|------------------------------|-------|
| Xylo Audio 2 | ASIC, 28nm | 6.6 uJ | best for audio. 461 neurons, 64K synapses |
| SpiNNaker 2 | digital, 22nm | ~7.1 uJ | 153 ARM M4F, MAC accelerators |
| Loihi 1 | digital, 14nm | 37 uJ | 128K neurons |
| Loihi 2 | digital, 7nm | ~3-10 uJ (est.) | 200x less than Jetson Orin |
| SpiNNaker 1 | digital, 130nm | ~25-30 mJ | our measurement, idle-dominated |
| TrueNorth | digital, 28nm | ~0.27-108 uJ | 1M neurons |
| MOVIDIUS NCS | accelerator | 1,500 uJ | neural compute stick |
| Jetson Nano | GPU, 20nm | 5,580 uJ | embedded GPU |
| GPU (A100) | GPU, 7nm | 29,670 uJ | datacenter |

SpiNNaker 1 is NOT competitive on per-inference energy. it was designed for brain-scale simulation (millions of neurons in parallel), not single-sample classification. ~1W idle per chip completely dominates.

but SpiNNaker 1 deployment is still valuable because it shows: (1) feasibility of running trained SNN on real neuromorphic hardware, (2) accuracy gap between software and hardware (quantified), (3) hardware-in-the-loop metrics, (4) our FC2-only approach (33.1%) is a real contribution.

for energy claims, use: NeuroBench for software comparison, Xylo/SpiNNaker 2 as projected targets, SpiNNaker 1 for hardware validation, always-on monitoring argument for real-world advantage.

could our model fit on Xylo Audio 2? no -- only 8 output channels (we need 50) and max 63 fan-in (we need 256). Xylo Audio 3 (TSMC 40nm, 2024 tapeout) might have expanded capacity.

---

## 7. previous SpiNNaker audio work

### Dominguez-Morales et al. (2016) -- the only direct precedent

"Multilayer SNN for Audio Samples Classification Using SpiNNaker," ICANN 2016, LNCS 9886, pp 45-53.

they classified pure tones (130.813 Hz to 1396.91 Hz) on a 4-chip SpiNNaker board. LIF neurons with rate-based training. >85% hit rate at SNR > 3 dB. open source: https://github.com/jpdominguez/Multilayer-SNN-for-audio-samples-classification-using-SpiNNaker

limitations vs us: pure tones only (single frequency sinusoids), no mel spectrogram/MFCC, simple rate-based training, no conv layers, no ANN comparison. our novelty claim remains strong.

### other SpiNNaker audio work

- sound source localization (Gutierrez-Galan et al. 2023, Sensors) -- ITD-based, not classification
- SCNN for speech (Dominguez-Morales 2018) -- 48-chip board, speech not environmental sounds
- heart murmur classification -- medical audio, not environmental

### best practices from previous deployments

1. spike rate reduction is critical (Perez-Carrasco achieved 34x reduction)
2. layer-by-layer training works (Gronauer's SpikingConvNet)
3. weight scaling matters -- all successful deployments calibrate this
4. 4-chip board is sufficient for small networks
5. event-based input encoding works best with SpiNNaker's architecture

---

## 8. SpiNNaker parameter calibration

### our current calibrated params
```python
LIF_PARAMS = {
    "cm": 1.0,        # nF
    "tau_m": 20.0,     # ms
    "tau_refrac": 0.1, # ms
    "v_reset": 0.0,    # mV
    "v_rest": 0.0,     # mV
    "v_thresh": 1.0,   # mV
    "tau_syn_E": 5.0,  # ms
    "tau_syn_I": 5.0,  # ms
}
```

### critical bug we fixed

population.initialize(v=0.0) -- without this, sPyNNaker defaults initial membrane potential to the NEURON TYPE default of -65.0 mV, NOT our v_rest parameter. with v_thresh=1.0 that creates a 66mV gap thats impossible to cross. this was the root cause of our early deployment failures.

sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 32) -- for populations >32 neurons, improves load balancing and prevents timer tick overruns.

### snnTorch to sPyNNaker mapping

snnTorch Leaky: `mem[t] = beta * mem[t-1] + I[t]`, spike if mem >= threshold
sPyNNaker IF_curr_exp: `cm * dV/dt = -(V - V_rest)/R_m + I_syn`, R_m = tau_m/cm

the relationship: snnTorch beta = exp(-dt/tau_m). for beta=0.95, dt=1ms: tau_m = -1/ln(0.95) = 19.5 ms (we use 20ms, close match). threshold = 1.0 maps directly. v_rest = 0.0 (snnTorch reset is 0).

weight scaling: weight_scale = 1.0 (no scaling needed) because v_rest=0.0, v_thresh=1.0 (same as snnTorch), cm=1.0 nF, tau_syn=5.0ms (fast decay). the learned weights already produce currents in the right range.

### handling negative weights

current-based models (IF_curr_exp) MUST split weights by sign:
- positive -> excitatory projection (receptor_type="excitatory")
- negative -> inhibitory projection, weight stored as absolute value

sPyNNaker IGNORES weight sign and relies solely on receptor_type. our code already handles this:
```python
fc2_exc = [c for c in fc2_conns if c[2] > 0]
fc2_inh = [(c[0], c[1], abs(c[2]), c[3]) for c in fc2_conns if c[2] < 0]
```

### router congestion

with many simultaneous spikes (FC1 has up to 2304 active inputs), router can congest and drop packets. mitigations:
1. set_number_of_neurons_per_core() distributes across more cores
2. temporal spreading: add 0.1ms jitter to stagger bursts
3. pruning reduces multicast destinations
4. check provenance for "router_provenance_for_dropped_mc_packets"

for FC2-only (256 in, 50 out): 12,800 connections, well within capacity. for FC1 (2304 in, 256 out): 589,824 connections, caused issues. our exc-only Step 3a approach worked (231/256 neurons firing).

---

## 9. recommendations

### for the paper (by April 1 ICONS deadline)

1. keep current SpiNNaker results: FC2-only, 33.1% +/- 6.9%, hardware gap 12.8pp
2. report energy using NeuroBench: SNN 968 nJ, ANN 454 nJ
3. report SpiNNaker energy as estimated ~25-30 mJ, note idle-power dominance
4. compare with Xylo Audio 2 and SpiNNaker 2
5. cite Ward et al. (2022) for multi-compartment feasibility
6. cite Dominguez-Morales et al. (2016) as only prior SpiNNaker audio work
7. frame always-on monitoring argument

### for the thesis (if time)

1. fix energy estimation bug in spinnaker_latency_energy.py (5.9 uJ should be 43-110 nJ)
2. implement early exit with 2-segment sim.run()
3. deploy pruned model (90% pruning = only 1,280 FC2 connections)
4. deploy with T=7 instead of T=25 (one-line change, 72% saving)
5. attempt FC1+FC2 with the population.initialize(v=0.0) fix

### future work statement

"future work includes: (1) deploying RhythmLIF and DendriticLIF using sPyNNaker's custom neuron model framework, building on Ward et al. (2022); (2) targeting SpiNNaker 2 with hardware MAC accelerators and DVFS for ~1000x energy improvement; (3) investigating Xylo Audio 2 for ultra-low-power (6.6 uJ) environmental sound monitoring."

---

## corrected energy estimation

the 5.9 uJ/event in our code is wrong for our setup. appropriate values for single-chip, small-model, 1ms timestep:
- 20 nJ (Stromatias 2013, total per event) -- recommended default
- 8 nJ (incremental) to 110 nJ (upper bound) for uncertainty range

corrected FC2-only (256->50, T=25):
- total input spikes: 55.6 * 25 = 1,390
- synaptic events: 1,390 * 50 = 69,500
- synaptic energy at 20 nJ: 1.39 mJ
- idle (25ms at 1W): 25 mJ
- total: ~25.6 - 32.6 mJ (idle-dominated regardless)

for FC1+FC2 (full model, T=25):
- FC1 events: ~3.89M (at 26.4% input activity)
- FC2 events: ~69.4K
- synaptic at 20 nJ: 79.2 mJ
- idle: 25 mJ
- total: ~104 mJ (20 nJ) to ~460 mJ (110 nJ)

SpiNNaker 1 energy is millijoules, not nanojoules. expected for brain-simulation platform, not inference accelerator.

---

## confidence

| finding | confidence | basis |
|---------|------------|-------|
| custom neuron models possible | HIGH | official docs, template, Ward et al. |
| multi-compartment demonstrated | HIGH | published + open source |
| RhythmLIF deployable via custom C | HIGH | simpler than demonstrated HH |
| SpiNNaker 1 energy ~25-30 mJ/inference | MEDIUM | Stromatias model, not direct measurement |
| Xylo Audio 2 energy 6.6 uJ | HIGH | on-hardware current monitors |
| SpiNNaker 2 ~10 pJ/synop | MEDIUM-HIGH | published benchmark |
| conv layers via KernelConnector | HIGH | multiple papers |
| full conv impractical on single chip | HIGH | neuron count analysis |

---

## references

SpiNNaker architecture and sPyNNaker:
1. Furber et al. "SpiNNaker: A 1-W 18-Core SoC." IEEE JSSC 2014.
2. Rhodes et al. "sPyNNaker." Frontiers in Neuroscience 2018.
3. sPyNNaker models: http://spinnakermanchester.github.io/spynnaker/5.0.0/SPyNNakerModelsAndLimitations.html
4. new model template: https://github.com/SpiNNakerManchester/sPyNNaker8NewModelTemplate
5. lab manual: http://spinnakermanchester.github.io/spynnaker/5.0.0/NewNeuronModels-LabManual.pdf

multi-compartment:
6. Ward et al. "Beyond LIF Neurons on Neuromorphic Hardware." Frontiers in Neuroscience 2022.
7. https://github.com/mollie-ward/beyondLIFNeurons

energy:
8. Stromatias et al. "Power Analysis of Large-Scale Real-Time Neural Networks on SpiNNaker." IEEE IJCNN 2013.
9. van Albada et al. Frontiers in Neuroscience 2018.
10. Stromatias et al. IEEE IJCNN 2015.

conv on SpiNNaker:
11. Serrano-Gotarredona & Linares-Barranco. IEEE ISCAS 2015.
12. Perez-Carrasco et al. Neural Networks 2019.
13. SpikingConvNet: https://github.com/SvenGronauer/SpikingConvNet
14. Blouw et al. NICE 2019.

SpiNNaker 2:
15. Hoeppner et al. arXiv:2401.04491, 2024.
16. NIR-SpiNNaker2: arXiv:2504.06748, 2025.

other hardware:
17. Xylo Audio 2: arXiv:2406.15112, 2024.
18. Blouw et al. arXiv:1812.01739, 2019.

audio on SpiNNaker:
20. Dominguez-Morales et al. ICANN 2016, LNCS 9886.
21. https://github.com/jpdominguez/Multilayer-SNN-for-audio-samples-classification-using-SpiNNaker
22. Gutierrez-Galan et al. Sensors 2023.

rhythm-SNN:
23. Zhao et al. Nature Communications 2025.

router:
25. Navaridas et al. Parallel Computing 2015.
