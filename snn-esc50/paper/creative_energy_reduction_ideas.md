# creative energy reduction ideas -- brainstorming notes

date: 26 march 2026
context: ~622K param SNN for ESC-50, deployed on SpiNNaker. current energy: ~968 nJ/sample (SNN), ~454 nJ/sample (ANN). sparsity: 73.6%. T=25 timesteps. direct encoding.

these are 25 unconventional ideas for dramatically reducing inference energy. some are practical, some are pretty out there. wrote most of these down after reading a ton of papers late at night so some might not make total sense in the morning.

the ones i'm most excited about:
1. weightless neural networks (DWNs) -- replace the SNN with lookup tables. 56 nJ per sample demonstrated. 5 ns inference. up to 926,000x savings over deep learning.
2. logic gate networks -- learned NAND/OR/XOR gates. 24 ns inference on FPGA. NeurIPS 2024 oral.
3. single-timestep multi-level SNN (T=1) -- collapse 25 timesteps to 1 using multi-level neurons. 66% energy reduction demonstrated.
4. fly olfactory circuit -- random projection + winner-take-all. biologically proven for classification.
5. hierarchical coarse-to-fine cascade -- classify super-category first, then fine class. most inputs exit at cheap first stage.

ordered roughly from most applicable to most speculative.

---

## idea 1: single-timestep multi-level SNN (T=1)

source: "All in One Timestep" (arXiv:2510.24637, 2024); "Scale-and-Fire Neurons" (arXiv:2510.23383, 2024)

so the core idea is: standard LIF neurons output binary {0,1} and need T=25 timesteps to accumulate enough info. multi-level neurons output integers {0,1,...,N} in a single timestep, encoding N bits where you previously needed N binary timesteps. a 4-level neuron at T=1 is mathematically equivalent to a binary neuron at T=4.

for our system:
- our temporal ablation shows T=7 reaches 90% of full accuracy
- with 4-level neurons at T=1, we could match T=4 performance
- with 8-level neurons at T=1, could approach T=8
- Scale-and-Fire variant: learn adaptive thresholds via Bayesian optimization

could probably get 15-25x out of this (from T=25 to T=1, multi-level ops cost ~2-3x per timestep but only 1 timestep total).

accuracy: T=1 binary gives 7.25% on our model. with 4-level neurons expect ~25-30%. with 8-level ~35%. still a gap vs T=25 (47.15%) but huge energy win.

SpiNNaker: partially. SpiNNaker 1 only supports binary spikes natively. would need to encode multi-level spikes as multiple rapid binary spikes, or use SpiNNaker 2, or implement as quantized ANN on the ARM cores.

probably a couple days to implement. snnTorch would need custom neuron model. the ANN-to-SNN conversion path is well-documented though.

"we compress 25 timesteps into 1 with only 15% accuracy loss" -- applied to audio for the first time -- thats a decent claim.

---

## idea 2: temporal early exit (input-adaptive timesteps)

source: "SEENN" (arXiv:2304.01230, 2023); "SPARQ" (arXiv:2603.14380, march 2026)

not all sounds need T=25 to classify. a dog bark might be identifiable at T=3 while distinguishing engine types might need T=20. monitor confidence at each timestep, exit when confident enough.

for our system:
- temporal ablation: T=5 gets 33.5%, T=10 gets 38.25%
- many ESC-50 classes have distinctive onsets (dog bark, gunshot, clock alarm)
- train a lightweight confidence estimator on membrane potential / output spike counts
- RL-guided: SPARQ uses Q-learning to learn per-class exit policies

could get 3-5x on average (most samples exit at T=5-10, hard samples go full T=25). SPARQ reports 5.15% accuracy improvement while reducing ops by 96%.

runs on SpiNNaker -- it already processes timestep-by-timestep. just add a confidence check after each timestep. when confidence exceeds threshold, stop sending input spikes.

not too hard to implement -- could do it as a post-hoc wrapper around existing model.

well-studied for vision but novel for audio SNN classification.

---

## idea 3: fly olfactory circuit -- random projection + winner-take-all

source: Dasgupta et al. "A neural algorithm for a fundamental computing problem" (Science 2017); Zheng et al. "Fly-CL" (arXiv:2510.16877, 2024)

ok so this one is my favourite. the fruit fly classifies ~50 odors using a 3-stage circuit:
1. 50 projection neurons (input)
2. random sparse projection to 2000 Kenyon cells (40x expansion, only 5% active via WTA)
3. 34 output neurons with simple learned weights

the random projection is FIXED -- no learning. only the output weights are learned. biologically proven to work as well as or better than locality-sensitive hashing.

for our system:
- mel spectrogram has 64x216 = 13,824 features per frame
- random sparse projection to ~50,000-dim space (4x expansion)
- winner-take-all: keep only top 5% active (like fly's APL inhibition)
- train only a 50,000 -> 50 linear readout
- the random projection matrix is FIXED and can be hardcoded

could probably get 10-50x. the random projection needs no multiply-accumulate (just sparse binary connections). WTA is a single comparison pass. only the tiny readout needs real computation.

accuracy: unknown for audio. for odor classification (analogous problem), fly algorithm matches or beats LSH. for ESC-50 with 50 classes (same as fly's ~50 odor types!!!) this is eerily well-matched. expect 20-35% on raw features, potentially higher with PANNs embeddings.

runs perfectly on SpiNNaker -- multicast routing IS the random projection. WTA implementable via lateral inhibition. this is literally what SpiNNaker was designed for.

could prototype in an afternoon. its just a random sparse binary matrix + linear readout.

"we replaced a 622K-parameter deep SNN with a fly brain circuit and achieved X% accuracy on ESC-50" would be a memerable paper line.

---

## idea 4: weightless neural networks (lookup table classification)

source: "Differentiable Weightless Neural Networks" (ICML 2024, Bacellar et al.); "nanoML for HAR" (arXiv:2502.12173, 2025); "LogicWiSARD" (ASAP 2022)

replace ALL neural computation with cascaded lookup tables. input is binarized into n-bit tuples. each tuple indexes into a small RAM that outputs a binary value. layers of LUTs are chained. no multiplication. no addition. just memory reads.

for our system:
- binarize mel features (or use spike patterns directly -- already binary!)
- form n-tuples from input bits
- cascade 2-3 layers of LUT lookups
- final discriminator per class counts matching patterns

the energy numbers are insane -- DWNs demonstrated 56 nJ per inference for HAR. 5 ns inference. up to 926,000x energy savings over deep learning. our SNN uses 968 nJ, so a DWN could potentially classify audio at 50-100 nJ which would be 10-20x less.

accuracy: DWNs get 96.3% on HAR (competitive with deep learning). for ESC-50 with 50 classes, expect lower (maybe 25-40%) but the energy savings are extraordinary.

doesn't run natively on SpiNNaker (optimized for spikes not LUT cascades). but SpiNNaker's ARM cores could implement LUT lookups and the entire model would fit in local SRAM. alternatively ideal for FPGA. 5 ns inference demonstrated.

DWN code available (PyTorch-compatible). would need custom encoding of audio features into binary tuples. probably a couple days to get working.

"we replaced a spiking neural network with cascaded lookup tables and achieved competitive classification at 56 nJ" -- if this works it'd be pretty jaw-dropping.

---

## idea 5: logic gate networks

source: "Convolutional Differentiable Logic Gate Networks" (NeurIPS 2024 Oral, Petersen et al.)

instead of learning weights for multiply-accumulate ops, learn which logic gates (NAND, OR, XOR, AND, etc.) to wire between binary signals. the network IS the hardware -- no abstraction layer between model and silicon. inference = executing a circuit of logic gates.

for our system:
- binarize spike patterns from the SNN
- learn a logic gate network mapping binary spike patterns -> 50 class labels
- inference on FPGA: 24 nanoseconds, 41.6 million samples/second
- model size: 29-61x smaller than SOTA

could potentially get 100-1000x reduction. logic gates consume femtojoules. 24 ns inference = negligible dynamic power.

on CIFAR-10 (comparable complexity to ESC-50): 86.29% with 61M logic gates. for our 50-class audio: expect 30-50%.

doesn't run on SpiNNaker (lacks programmable logic fabric). ideal for FPGA or custom ASIC. but the concept of "distill SNN to logic gates" is hardware-agnostic.

would take a bit more work -- need to train the logic gate network to mimic SNN behavior. the difflogic library exists though.

"neural network inference in pure combinational logic" applied to audio for the first time -- thats pretty paradigm-shifting.

---

## idea 6: learned audio fingerprint / hash-based classification

source: PeakNetFP (arXiv:2506.21086); SAMAF (ACM TOMM 2020); "Can LSH be replaced by Neural Network?" (Soft Computing 2024)

instead of running a full forward pass, compute a compact binary hash (128-256 bits) of the input spectrogram. compare via Hamming distance to 50 pre-computed class prototypes. classification = argmin of 50 Hamming distances. Hamming distance is just XOR + popcount.

for our system:
- train small encoder to produce 256-bit hash from mel spectrogram
- store 50 prototype hashes
- at inference: encode input -> 50 XOR+popcount ops -> argmin
- total: one small encoding + 50 integer comparisons

could get 50-200x. hash encoding can be tiny (even a single layer). the 50 comparisons are trivially cheap.

accuracy depends on hash quality. with well-trained encoder, expect 30-45%. audio fingerprinting achieves near-perfect matching with 128-bit hashes.

partially runs on SpiNNaker (hash computation could run there, Hamming distance on ARM cores). but better suited for general-purpose hardware.

would need contrastive learning to train the hash encoder. probably a few days.

"replacing neural network inference with 50 Hamming distance computations" for audio classification -- that'd be pretty novel.

---

## idea 7: hierarchical coarse-to-fine cascade

source: "ECHO: ESC with Hierarchical Ontology" (arXiv:2409.14043, 2024); "Coarse-to-Fine DNN inference" (2024)

ESC-50 has 5 super-categories (Animals, Nature, Human, Domestic, Urban) with 10 classes each. use a tiny first-stage classifier (5-class) followed by category-specific second-stage classifiers (10-class each). saves energy because stage 1 is much cheaper, and stage 2 is only 10-class not 50-class.

for our system:
- stage 1: tiny SNN (~10K params) classifies into 5 super-categories
- stage 2: 5 specialized tiny SNNs (~30K params each) for fine classification
- easy samples may not even need stage 2
- our per-category analysis shows some categories are much easier

could get 3-10x. stage 1 is ~5-10x cheaper than full model. many samples exit early.

accuracy could actually IMPROVE since each specialized sub-model focuses on distinguishing 10 similar sounds rather than 50 diverse ones. our confusion matrix shows most errors are within categories.

runs on SpiNNaker. two-stage pipeline with conditional routing.

not hard -- train 6 models instead of 1.

not groundbreaking in terms of novelty but systematic ESC-50 cascade with SNN + SpiNNaker hasn't been done.

---

## idea 8: reservoir computing / liquid state machine

source: "EARL: Energy-Aware Optimization of LSMs" (arXiv:2601.05205, 2026); "Deep Echo State Network for ESC" (Springer 2024); LSM on SpiNNaker (validated)

replace the trained convolutional layers with a RANDOM, FIXED recurrent spiking network (the "reservoir"). only train a linear readout from reservoir state to 50 classes. the reservoir separates inputs through complex dynamics -- no backprop needed.

for our system:
- random recurrent SNN of ~500-2000 neurons (fixed weights)
- feed mel spectrogram spikes into reservoir
- read out reservoir state at the end
- train only a 2000->50 linear layer (100K params, no backprop through reservoir)
- random reservoir can be HARDCODED in SpiNNaker routing tables

could get 5-20x. no gradient computation. random weights = no weight memory updates.

accuracy: deep ESN on ESC-10 got competitive results. for ESC-50 (harder), expect 15-30%. with optimized reservoir (EARL framework) potentially higher.

runs perfectly on SpiNNaker. LSMs are one of SpiNNaker's native use cases. several published implementations exist.

not hard to implement -- sPyNNaker has built-in support for random connectivity.

"we achieved X% accuracy on ESC-50 with NO TRAINING of the spiking layers" -- thats a good narrative.

---

## idea 9: extreme pruning to 95-99% sparsity (lottery tickets for SNN)

source: "Exploring Lottery Ticket Hypothesis in SNNs" (ECCV 2022, arXiv:2207.01382); "QP-SNN" (ICLR 2025)

our pruning experiment shows SNN retains 93.2% accuracy at 90% pruning. the lottery ticket hypothesis says there's a tiny subnetwork (~1-5% of original) that performs as well as the full network.

for our system:
- current: 622K params, 90% pruning retains ~38% accuracy
- push to 95%: expect ~35%
- push to 99%: ~6.2K params remaining. accuracy may drop to 25-30% but energy savings are massive
- combined with quantization (QP-SNN approach): 2-4 bit weights + 99% pruning

could get 10-100x at 99% sparsity. only 6.2K of 622K weights non-zero. SpiNNaker can skip all zero-weight synaptic ops.

runs on SpiNNaker. sparse connectivity is native. fewer synapses = fewer routing entries = less packet traffic = less energy.

iterative magnitude pruning + fine-tuning. not trivial but doable.

LTH for SNN exists but ESC-50 + SpiNNaker deployment would be new.

---

## idea 10: predictive coding -- only process the unexpected

source: "Predictive Coding Light" (Nature Communications 2025); "Predictive coding is a consequence of energy efficiency" (Cell Patterns 2023); survey arXiv:2409.05386

the brain doesnt process every incoming signal. it maintains a prediction of what it expects and ONLY processes the prediction error (the "surprise"). for audio: if spectrogram at time t is similar to t-1, dont process it. only transmit the difference.

for our system:
- environmental sounds have high temporal redundancy (dog barking has repeating patterns)
- build a simple prediction model of spectrogram's next frame
- only encode and process frames that significantly differ from prediction
- naturally produces SPARSE spike trains (only "surprise" frames generate spikes)

could get 5-20x depending on sound type. continuous sounds (engine, rain) are highly predictable = very sparse. transient sounds (gunshot) produce brief bursts.

accuracy: minimal loss if prediction model is good. the "surprise" signal contains all classification-relevant info by definition.

runs on SpiNNaker. prediction can be implemented as simple recurrent connection. error computation is subtraction (excitatory vs inhibitory spikes).

need to design prediction mechanism for spectrograms. delta encoding (which we already have!) is a simple form of this. probably a few days work.

predictive coding SNN for audio on neuromorphic hardware would be genuinely novel.

---

## idea 11: spike latency coding with delay learning

source: "Delay learning based on temporal coding in SNNs" (Neural Networks 2024); "Learnable axonal delay" (Frontiers 2023); "Latency Coding Framework" (arXiv:2603.23206, 2026)

instead of rate coding (many spikes = large value), use time-to-first-spike coding (earlier spike = larger value). each neuron fires at most ONE spike. info is in the timing, not the count. maximally sparse: exactly N spikes for N neurons.

combined with delay learning: instead of learning weights, learn DELAYS on synaptic connections. a spike arriving with the right delay at the right time is equivalent to a weight but uses zero multiply-accumulate ops.

for our system:
- convert to TTFS coding: each input neuron fires exactly once
- learn optimal delays on connections
- total spikes per inference: exactly N_input + N_hidden + N_output = ~2600 + 256 + 50 = ~2906
- deterministic, minimal

could get 5-15x. exact spike count known and minimal. delays can be hardcoded.

TTFS-based SNNs achieve comparable accuracy to rate-coded on vision benchmarks. for audio, recent work shows improved spoken word recognition with learnable delays.

partially runs on SpiNNaker. it supports synaptic delays natively. but TTFS decoding requires precise timing which SpiNNaker's 1ms timestep may limit.

requires retraining with TTFS encoding and delay learning. not trivial but the literature is there.

delay-learning SNN for audio on neuromorphic hardware is unexplored. pretty novel.

---

## idea 12: binary / ternary weight SNN

source: "Ternary Weight Networks" (arXiv:1605.04711); "xTern" (2024); "1 Bit is All We Need" (arXiv:2509.07025, 2025)

constrain all weights to {-1, 0, +1}. multiply-accumulate becomes add/subtract/skip. with binary weights {0,1}, MAC becomes a single AND gate. with ternary {-1,0,+1}, MAC becomes conditional add/subtract.

for our system:
- our SNN already has binary spikes (0/1). if weights are also ternary then:
  - synaptic operation = add, subtract, or skip (no multiplication at all)
  - 16x memory reduction (ternary weights need 2 bits vs 32-bit float)
- train with ternary quantization-aware training
- SpiNNaker already uses fixed-point weights; ternary is a natural extension

could get 5-16x. elimination of all multiplications. 16x less weight memory.

ternary typically loses 1-5% vs full precision. our 47.15% might drop to 42-45%.

runs on SpiNNaker. synaptic weights stored as 2-bit values. computation simplifies to conditional add/subtract.

not too hard -- ternary QAT is well-established. need to verify snnTorch compatibility.

ternary SNNs exist for vision. first application to audio on SpiNNaker is new.

---

## idea 13: spectrogram subsampling -- classify from less

source: our own temporal ablation data; "How Low Can You Go?" (arXiv:1911.04824)

we use 64 mel bins x 216 time frames = 13,824 features. but do we need all of them? our saliency maps show the SNN attends to specific regions. what if we:
- use 16 mel bins instead of 64 (4x fewer)
- use first 1 second instead of 5 seconds (5x fewer, ~43 frames)
- total: 16 x 43 = 688 features (20x reduction)

for our system:
- retrain with 16 mel bins x 43 time frames
- much smaller input = much smaller model = much less computation
- feature selection: use saliency maps to identify which bins matter
- per-class importance: maybe only 8 bins matter for "dog_bark" but 32 for "engine"

could get 5-20x (proportional to input size reduction).

significant accuracy impact but quantifiable. 48 mel bins performs nearly as well as 128 in published work. first 1-2 seconds likely sufficient for many ESC-50 sounds. expect 30-40% with aggressive subsampling.

runs on SpiNNaker. smaller input = fewer neurons = simpler deployment.

easy -- just change mel spectrogram params and retrain.

not super novel but systematic ESC-50 ablation with SNN energy analysis would be useful data.

---

## idea 14: knowledge distillation -- large teacher, tiny student

source: "Dynamic Activation with KD for Energy-Efficient SNN Ensembles" (arXiv:2502.14023, 2025); "Efficient Speech Command Recognition via SNN and KD" (arXiv:2412.12858, 2024)

train a large accurate ANN teacher, distill into tiny SNN student (10-50x smaller). student learns to mimic teacher's soft output distribution.

for our system:
- teacher: PANNs CNN14 (81M params, 92.5% on ESC-50)
- student: tiny SNN (10-50K params)
- student learns "dark knowledge" -- which classes are similar, which features matter
- deploy only tiny student on SpiNNaker

could get 10-60x from model size reduction. KD can reduce timesteps by 60% and energy by 54.8%.

BUT... our own results showed KD hurts everything. every combo with KD performed worse. so im skeptical about this one for us specifically. the literature says specialized SNN-aware KD methods (SAMD, NLD, SAKD) can help but theyre more complex.

probably not worth pursuing unless we implement specialized SNN KD. filing this under "maybe later."

---

## idea 15: hyperdimensional computing (HDC)

source: "A Robust and Energy-Efficient Classifier Using Brain-Inspired HDC" (ISLPED 2016); "Hierarchical HDC for Energy Efficient Classification" (DAC 2018)

represent each feature and class as a very high-dimensional binary vector (e.g., 10,000 bits). encode inputs by binding and bundling hypervectors. classify by cosine similarity to class prototypes. all operations are bitwise (XOR, majority vote).

for our system:
- encode each mel bin as a 10,000-bit hypervector
- bind time and frequency positions using XOR
- bundle across time using majority vote
- compare to 50 class prototypes via Hamming distance
- total: bitwise operations only, no floating point

could get 20-100x. 39.4 nJ/prediction demonstrated for multi-channel physiological data.

HDC gets 93-96% on simple tasks. for ESC-50, expect 25-40% with raw features, potentially higher with pre-extracted features.

partially runs on SpiNNaker (ARM cores can do bitwise ops). but not really optimized for this paradigm. better on FPGA.

torchhd library available. need to design encoding for spectrograms. not too hard.

HDC for environmental sound classification is completely unexplored territory. reviewers might find this interesting.

---

## idea 16: event-driven processing -- skip silence and redundancy

source: "Exploiting neuro-inspired dynamic sparsity" (Nature Communications 2025); Speech2Spikes (NICE 2023)

environmental sounds are not continuous. a 5-second "dog bark" clip might have 0.5s of barking and 4.5s of silence. process ONLY the event-rich portions.

for our system:
- compute energy envelope of audio
- only generate spikes for frames where energy exceeds threshold
- skip all silence/low-energy frames (zero computation)
- many ESC-50 sounds are sparse events (clock tick, door knock, coughing)
- 80-95% of frames can be skipped for many classes

could get 2-10x depending on class. sparse sounds (clock tick): 10x. continuous (rain): 1.5x. average across ESC-50: ~4x.

minimal accuracy impact -- classification info is in the events, not the silence.

runs perfectly on SpiNNaker -- its event-driven by design. no input spikes = no computation.

very easy to implement. simple energy-based gating on input.

conceptually simple but systematic ESC-50 analysis would be useful data.

---

## idea 17: stochastic computing -- multiply with AND gates

source: "Stochastic Computing CNN Architecture Reinvented for FPGA" (MDPI 2024); "Hardware-Efficient Stochastic Computing-Based NNs with SNN-Isomorphic LIF" (2025)

represent numbers as random bitstreams where the probability of a 1 encodes the value (e.g., 0.7 = bitstream with 70% ones). multiplication becomes a single AND gate. addition becomes a multiplexer. entire NN layers become simple combinational logic.

for our system:
- already have binary spikes -- natural fit for stochastic representation
- convert SNN weights to stochastic bitstreams
- synaptic op: AND(spike, weight_bit) instead of multiply
- 4.5x more energy efficient and 4x more area efficient than conventional FPGA

could get 3-5x per operation. combined with spike sparsity: 5-15x overall.

stochastic noise adds ~1-3% accuracy loss.

doesnt run on SpiNNaker (deterministic computation). better for FPGA or custom hardware.

requires custom hardware design. this is probably too hard for us right now.

SC for SNN has been proposed but not widely implemented for audio.

---

## idea 18: L-Mul -- replace multiplications with integer additions

source: "Addition is All You Need for Energy-efficient Language Models" (arXiv:2410.00907, 2024, ICLR 2025)

the L-Mul algorithm approximates floating-point multiplication using only integer addition by cleverly manipulating the exponent and mantissa. one integer addition costs ~37x less energy than one float32 multiply.

for our system:
- replace all MAC ops with L-Mul
- 95% energy reduction in element-wise ops
- 80% in dot products
- applicable to our ANN baseline too

could get 3-10x for the compute portion. but our SNN energy is dominated by synaptic ops (accumulate-and-fire) which are already additions. bigger impact on ANN comparison. L-Mul with 4-bit mantissa matches float8 precision.

partially applicable to SpiNNaker (already integer-based). but reinforces our existing efficiency story.

low effort for software -- drop-in replacement. recent paper, hot topic.

---

## idea 19: neuromorphic cochlea frontend -- bypass mel spectrograms

source: "Neuromorphic acoustic sensing using adaptive MEMS cochlea" (Nature Electronics 2023); "FPGA-based Neuromorphic Cochlea" (arXiv:2405.15923, 2024); Speech2Spikes (NICE 2023)

replace the entire mel spectrogram pipeline (FFT, mel filterbank, log compression) with a neuromorphic cochlea that directly produces spike trains from raw audio. operates in analog domain, consuming microwatts.

for our system:
- current: audio -> FFT -> mel -> encode -> SNN (all digital)
- proposed: audio -> silicon cochlea (analog) -> spikes -> SNN
- ~120 frequency channels, event-driven, microwatt power
- eliminates ALL digital preprocessing energy

preprocessing savings: 100-1000x for the frontend. total system: 2-5x (preprocessing isnt the dominant cost).

bio-inspired preprocessing often IMPROVES noise robustness. Speech2Spikes showed competitive results with event-driven encoding.

runs on SpiNNaker. spike-based cochlea output feeds directly in. this is the intended neuromorphic pipeline.

requires hardware cochlea or software simulation. probably too complex for thesis but cool concept.

full neuromorphic audio pipeline (cochlea + SpiNNaker SNN) for ESC-50 would be a first.

---

## idea 20: forward-forward algorithm -- no backpropagation

source: "Backpropagation-free SNNs with the Forward-Forward Algorithm" (arXiv:2502.20411, 2025; Scientific Reports 2026)

replace backprop with two forward passes (positive with correct label, negative with wrong label). each layer learns locally. no backward pass = no storing intermediate activations = 60%+ memory reduction.

for our system:
- train SNN using FF instead of backprop
- each layer learns independently
- on-device learning becomes possible (no backward pass = neuromorphic-compatible training)
- memory: <40% of BP training cost

training: 3-10x energy reduction. inference: similar to BP-trained, but enables on-chip learning on SpiNNaker for adaptation.

FF-SNNs get ~92% on MNIST (vs 99% BP). for ESC-50, expect 5-15% accuracy drop.

runs perfectly on SpiNNaker for inference. also enables ON-CHIP training since no backward pass needed. thats the key advantage.

snnTorch has a FF tutorial. would need adaptation for audio. probably a few days work.

FF-SNN for audio on SpiNNaker with on-chip learning -- genuinely novel.

---

## idea 21: dendritic computation -- fewer but smarter neurons

source: "Dendritic Integration Inspired ANNs" (NeurIPS 2024); "Introducing the Dendrify framework" (Nature Communications 2022)

standard LIF neurons are "point neurons" -- they sum all inputs linearly. real neurons have dendrites that perform nonlinear computation WITHIN the neuron. one dendritic neuron can compute what takes 10-100 point neurons.

for our system:
- replace 256 hidden LIF neurons with 25-50 dendritic neurons
- each has 5-10 branches processing different feature subsets
- same computational power, 5-10x fewer neurons
- fewer neurons = fewer spikes = less routing on SpiNNaker

could get 3-10x (proportional to neuron reduction).

dendritic models get comparable or better accuracy (91% vs 88% in digit classification). expect maintained or slightly improved accuracy.

partially runs on SpiNNaker -- it supports multi-compartment neuron models. dendritic branches modeled as separate compartments. more computation per neuron but far fewer neurons.

implementation is harder though. need custom neuron model for SpiNNaker. limited sPyNNaker support for multi-compartment.

dendritic SNN for audio on neuromorphic hardware is unexplored. reviewers would probably like this.

---

## idea 22: inference caching / memoization

source: CacheNet (arXiv:2007.01793); "Energy-efficient acceleration of CNNs using computation reuse" (2022)

many environmental sounds are repetitive (factory has same 5-10 sounds recurring). cache recent classification results. if new input is "similar enough" to cached input, reuse result. zero inference energy for cache hits.

for our system:
- maintain small cache of K most recent (feature_hash, class_label) pairs
- new input: compute simple hash, check cache
- cache hit: return cached label (near-zero energy)
- cache miss: run full SNN, update cache
- for streaming audio: high cache hit rate due to temporal continuity

could get 5-50x in real environments (80-98% cache hit rate for stationary sources). for one-shot ESC-50 evaluation: 1x (no benefit).

zero accuracy impact if similarity threshold set correctly.

runs on SpiNNaker. cache in SRAM. simple hash comparison before triggering SNN.

straightforward engineering. hash function + small lookup table. easy.

novel in neuromorphic context. practical contribution.

---

## idea 23: membrane potential quantization (ultra-low bit)

source: "SpQuant-SNN" (Frontiers 2024); "QUEST" (arXiv:2504.00679, 2025)

our SNN uses 32-bit floating-point membrane potentials. quantize to 2-4 bits. reduces memory by 7-13x and eliminates floating-point arithmetic entirely.

for our system:
- weights: ternary {-1,0,+1} = 2 bits
- membrane potentials: 4-bit integer
- spikes: 1-bit binary (already)
- total model memory: ~10 KB (down from ~2.5 MB)
- all operations: integer add/subtract with 4-bit accumulator

could get 7-13x from memory reduction alone plus elimination of float ops.

4-bit membrane: ~2-5% accuracy drop. 2-bit: ~10-15% drop. quantization-aware training mitigates losses.

runs on SpiNNaker -- already uses fixed-point internally. lower precision = less computation per neuron update.

quantization-aware training with snnTorch. need to verify SpiNNaker compatibility. not too hard.

---

## idea 24: self-powered neuromorphic audio sensor

source: "Self-aware artificial auditory neuron with triboelectric sensor" (Nano Energy 2023)

use a triboelectric nanogenerator (TENG) as both the audio sensor AND power source. sound waves hitting the TENG generate both the sensing signal and electricity to power the classifier. net external energy: ZERO.

for our system:
- TENG converts sound waves to electrical spikes (natural spiking interface)
- harvested energy powers small SNN classifier
- completely self-powered, maintenance-free
- total external energy: 0 J per inference

this is kind of insane -- literally infinite energy reduction (self-powered).

TENG bandwidth is limited though. may not capture full spectral detail. expect 15-25% for simplified classification.

doesnt run on SpiNNaker (draws too much power for harvesting). needs tiny custom neuromorphic chip.

extremely high implementation difficulty. requires hardware TENG + custom circuit.

but "self-powered audio classification with zero external energy" is a pretty wild claim to make, even theoretically.

---

## idea 25: Phi framework -- pattern-based hierarchical sparsity

source: "Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency SNNs" (ISCA 2025, Duke University)

binary spike activations contain repeating patterns across neurons and timesteps. Phi identifies these via k-means clustering, pre-computes results for each pattern, skips redundant computation. two-level hierarchy: vector-wise pattern matching + intra-pattern zero skipping.

for our system:
- analyze spike activation patterns in trained SNN
- cluster into representative patterns
- pre-compute partial results per pattern
- at inference: match input pattern -> lookup pre-computed result
- 3.45x speedup and 4.93x energy improvement demonstrated

could get 3-5x with Phi alone. combined with existing sparsity (73.6%): 5-8x.

pattern-aware fine-tuning recovers accuracy. <1% degradation reported.

partially runs on SpiNNaker -- pattern matching on ARM cores. full benefit needs dedicated hardware.

would need to extract patterns from our model. not trivial but doable.

very recent (june 2025). first application to audio would be novel.

---

## synthesis: most promising combinations

### "the fly-brain audio classifier" (ideas 3 + 8 + 16)
- silicon cochlea or event-driven encoding (skip silence)
- random sparse projection (fixed, no training)
- winner-take-all
- tiny trained readout
- total energy: ~50-200 nJ. accuracy: 20-35%. novelty: extreme.

### "the one-shot SNN" (ideas 1 + 2 + 12)
- multi-level neurons at T=1-3
- temporal early exit for easy samples
- ternary weights
- total energy: ~50-100 nJ. accuracy: 30-40%. novelty: high.

### "the cascaded lookup classifier" (ideas 4 + 6 + 7)
- coarse super-category classification via simple hash
- fine classification via class-specific DWN lookup tables
- total energy: ~30-80 nJ. accuracy: 35-45%. novelty: extreme.

### "the pruned predictive SNN" (ideas 9 + 10 + 23)
- 95-99% pruning (lottery ticket)
- predictive coding (only process surprises)
- membrane quantization to 4 bits
- total energy: ~50-150 nJ. accuracy: 35-42%. novelty: high.

### "the practical deployable improvement" (ideas 2 + 7 + 13 + 16)
- temporal early exit (easy sounds exit at T=5)
- hierarchical coarse-to-fine cascade
- spectrogram subsampling (first 2 seconds, 32 mel bins)
- event-driven silence skipping
- total energy: ~100-300 nJ. accuracy: 40-47%. most practical option.

---

## energy comparison table (estimated)

| approach | energy (nJ) | accuracy (%) | hardware | how novel |
|----------|-------------|-------------|----------|-----------|
| current SNN (T=25) | 968 | 47.15 | SpiNNaker | baseline |
| current ANN | 454 | 63.85 | GPU | baseline |
| multi-level T=1 SNN | 50-100 | 30-40 | SpiNNaker* | pretty novel |
| temporal early exit | 200-400 | 45-47 | SpiNNaker | established for vision |
| fly olfactory circuit | 50-200 | 20-35 | SpiNNaker | very novel |
| weightless NN (DWN) | 30-100 | 25-40 | FPGA | extremely novel |
| logic gate network | 1-10 | 30-50 | FPGA | extremely novel |
| hash-based | 20-50 | 30-45 | any | pretty novel |
| hierarchical cascade | 100-300 | 42-48 | SpiNNaker | modest |
| reservoir/LSM | 200-500 | 15-30 | SpiNNaker | novel for audio |
| 99% pruned SNN | 50-150 | 25-30 | SpiNNaker | modest |
| predictive coding SNN | 100-300 | 40-45 | SpiNNaker | novel |
| TTFS + delay learning | 100-200 | 35-45 | SpiNNaker* | novel |
| ternary weight SNN | 200-400 | 42-45 | SpiNNaker | modest |
| HDC (10,000-bit) | 50-100 | 25-40 | FPGA | novel |
| stochastic computing | 100-300 | 40-44 | FPGA | modest |
| KD tiny student | 100-300 | 40-44 | SpiNNaker | modest |
| SpQuant (4-bit membrane) | 200-500 | 42-45 | SpiNNaker | modest |
| Phi pattern sparsity | 200-400 | 45-47 | custom | novel |
| combined best | 30-100 | 35-45 | various | extreme |

*with modifications

---

## difficulty vs impact

| | low impact (1-3x) | medium (3-10x) | high (10-100x) | extreme (100x+) |
|---|---|---|---|---|
| easy | silence skip, subsampling | early exit, ternary | 99% pruning | -- |
| medium | predictive coding | KD student, cascade, LSM | multi-level T=1, HDC | hash-based, DWN |
| hard | -- | delay learning, Phi | logic gates | neuromorphic cochlea |
| very hard | -- | stochastic computing | self-powered TENG | -- |

---

## research gaps

1. nobody has applied DWNs (weightless NNs) to audio classification. completely open.
2. nobody has tried the fly olfactory circuit for environmental sound. the 50-class / 50-odor parallel is just too perfect.
3. nobody has combined temporal early exit with SpiNNaker for audio. SPARQ (march 2026) is brand new.
4. nobody has distilled an audio SNN into logic gate networks.
5. the rate-distortion theoretical minimum for classifying 50 environmental sounds has never been computed.

---

## confidence

- high: temporal early exit, hierarchical cascade, pruning, ternary weights -- well-established with predictable outcomes.
- medium: multi-level T=1, HDC, reservoir, hash-based -- demonstrated in other domains, transferability plausible but unverified.
- low (but high novelty): DWN for audio, fly circuit, logic gates, self-powered -- speculative but could be breakthroughs. these are the ones that could lead to a really memorable paper.

---

## references

### single-timestep and multi-level SNNs
- [All in One Timestep](https://arxiv.org/html/2510.24637)
- [Scale-and-Fire Neurons](https://arxiv.org/html/2510.23383)
- [One Timestep Is All You Need](https://openreview.net/forum?id=swRxhFpK5ds)

### temporal early exit
- [SEENN](https://ar5iv.labs.arxiv.org/html/2304.01230)
- [SPARQ](https://arxiv.org/html/2603.14380)
- [Early-Exit Survey](https://dl.acm.org/doi/full/10.1145/3698767)

### biological circuits
- [Fly Olfactory Algorithm](https://www.quantamagazine.org/new-ai-strategy-mimics-how-brains-learn-to-smell-20180918/)
- [Fly-CL](https://arxiv.org/html/2510.16877v1)
- [Sparse Code for Natural Sound](https://pmc.ncbi.nlm.nih.gov/articles/PMC10749876/)

### weightless and logic gate networks
- [DWN at ICML 2024](https://proceedings.mlr.press/v235/bacellar24a.html)
- [nanoML for HAR](https://arxiv.org/html/2502.12173v1)
- [Logic Gate Networks NeurIPS 2024](https://arxiv.org/html/2411.04732v1)
- [LogicWiSARD](https://labs.engineering.asu.edu/advent/wp-content/uploads/sites/123/2023/09/LogicWisard_ASAP22_May20.pdf)

### pruning and sparsity
- [LTH for SNN](https://arxiv.org/abs/2207.01382)
- [QP-SNN](https://openreview.net/forum?id=MiPyle6Jef)
- [Phi Framework](https://arxiv.org/html/2505.10909)
- [Dynamic Spatio-Temporal Pruning](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1545583/full)

### energy-efficient computation
- [Addition is All You Need](https://arxiv.org/abs/2410.00907)
- [Neuro-Inspired Dynamic Sparsity](https://www.nature.com/articles/s41467-025-65387-7)
- [Horowitz 2014](https://ieeexplore.ieee.org/document/6757323)

### reservoir computing
- [EARL Framework](https://arxiv.org/pdf/2601.05205)
- [Deep ESN for ESC](https://link.springer.com/chapter/10.1007/978-981-96-0994-9_7)

### knowledge distillation
- [SNN Ensemble KD](https://arxiv.org/html/2502.14023v1)
- [Speech SNN KD](https://arxiv.org/html/2412.12858v1)

### predictive coding
- [Predictive Coding Light](https://www.nature.com/articles/s41467-025-64234-z)
- [Predictive Coding Survey for SNN](https://arxiv.org/html/2409.05386v1)

### hyperdimensional computing
- [HDC Processor](https://ieeexplore.ieee.org/abstract/document/9645008)
- [Efficient HDC](https://openreview.net/forum?id=9RQh6MOOaD)

### neuromorphic audio
- [Neuromorphic Cochlea](https://www.nature.com/articles/s41928-023-00957-5)
- [Speech2Spikes](https://dl.acm.org/doi/fullHtml/10.1145/3584954.3584995)
- [Neuromorphic KWS with PDM](https://arxiv.org/html/2408.05156v1)

### quantization and binary networks
- [SpQuant-SNN](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full)
- [1 Bit Is All We Need](https://arxiv.org/html/2509.07025v1)
- [xTern](https://www.aimodels.fyi/papers/arxiv/xtern-energy-efficient-ternary-neural-network-inference)

### delay learning
- [Delay Learning in SNN](https://www.sciencedirect.com/science/article/abs/pii/S0893608024006026)
- [Learnable Axonal Delay](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1275944/full)

### forward-forward algorithm
- [FF for SNN](https://arxiv.org/abs/2502.20411)
- [snnTorch FF Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_forward_forward.html)

### hierarchical classification
- [ECHO Framework](https://arxiv.org/pdf/2409.14043)
- [Coarse-to-Fine Edge](https://www.sciencedirect.com/science/article/pii/S0167739X24000736)

### thermodynamic limits
- [Thermodynamic Bounds on DNN Energy](https://arxiv.org/html/2503.09980v2)
- [Landauer Principle](https://en.wikipedia.org/wiki/Landauer%27s_principle)

### self-powered sensors
- [Triboelectric Auditory Neuron](https://www.sciencedirect.com/science/article/abs/pii/S2211285523001593)

### SpiNNaker energy
- [SpiNNaker 2 Architecture](https://arxiv.org/pdf/2103.08392)
- [SpiNNaker 2 for ML](https://arxiv.org/html/2401.04491v1)
