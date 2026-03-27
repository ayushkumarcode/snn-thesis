# cross-domain energy reduction ideas -- deep dive notes

27 march 2026

context: SNN for ESC-50 (50-class environmental sound, 622K params, 968 nJ/sample, T=25, spike rate 26.4%). trying to get 10-100x energy reduction.

did about 40+ web searches across biology, signal processing, info theory, hardware design, and unconventional computing. papers from 2017-2026. heres everything i found organized by domain.

the top 5 most reviewer-surprising ideas:

1. **moth auditory classifier** -- classify environmental sounds with a 2-neuron decision circuit. the noctuid moth classifies bat threats using literally ONE auditory neuron with two thresholds. could adapt this as a hierarchical cascade.
2. **weightless NNs / DWN** -- replace entire SNN with cascaded lookup tables. 56 nJ per inference demonstrated on FPGA at 5 ns latency. 926,000x energy savings over deep learning.
3. **fly olfactory circuit** -- 50 classes maps perfectly to the fly's ~50 odor types. random sparse binary projection (FIXED) + winner-take-all + linear readout. SpiNNaker's multicast routing IS the random projection.
4. **logic gate networks** (NeurIPS 2024 oral) -- distill SNN into pure NAND/OR/XOR gates. 4 ns inference on FPGA. 86.29% on CIFAR-10.
5. **predictive coding light** (Nature Comms 2025) -- suppress predictable spikes, transmit only "surprise." for temporally redundant audio (engine hum, rain), could eliminate 80-95% of spikes.

---

## category 1: biology-inspired computation

### idea 1.1: moth auditory classifier -- the 1-neuron sound detector

source: Ratcliffe et al. "Tiger moths and the threat of bats" (Biology Letters, 2009); Roeder "Auditory System of Noctuid Moths" (Science, 1966)

so the noctuid moth classifies bat echolocation threats using the simplest auditory system in nature: 1-2 receptor neurons per ear. the A1 cell alone provides enough info. decision-making uses two amplitude thresholds on a single neuron's firing rate:
- low amplitude (distant bat) -> turn away
- high amplitude (close bat) -> last-ditch evasive maneuver (dive/spiral)

the moth effectively does 3-class classification (no threat / distant / close) with ONE neuron and TWO thresholds. inputs from both ears are just summed.

could get 1000-10,000x reduction. a 2-threshold comparator on one neuron vs our 622K-parameter network. as a cascade (moth-like gate + specialist), the gate rejects 80%+ of "uninteresting" inputs at near-zero cost.

adapation for our system:
- stage 0 (moth gate): single LIF neuron receives energy envelope. two thresholds classify: silence (skip), simple sound (tiny model), complex sound (full model).
- 80% of ESC-50's 5-second clips have long silent/low-energy regions. skip those entirely.
- for "simple sound": many classes have distinctive energy profiles (gunshot = sharp onset, rain = continuous noise). a 2-neuron energy-profile classifier could handle 10-15 classes.

easy to implement -- the moth gate is literally a threshold comparator on the audio energy envelope.

"we reduced inference energy 100x by copying the moth's 1-neuron auditory system" -- thats an unforgettable paper line if it works.

---

### idea 1.2: fruit fly olfactory circuit -- random projection + WTA

source: Dasgupta, Stevens, Navlakha "A neural algorithm for a fundamental computing problem" (Science, 2017); Zheng et al. "Fly-CL" (arXiv:2510.16877, 2024); Evolving the Olfactory System with ML (Neuron, 2021)

the drosophila classifies ~50 odors with a 3-layer circuit:
1. 50 projection neurons (PNs) -- input
2. ~2000 Kenyon cells (KCs) -- RANDOM, SPARSE, BINARY, FIXED projection (40x expansion). each KC recieves input from ~6 random PNs.
3. APL neuron -- global inhibition, only top 5% of KCs active (WTA)
4. ~34 output neurons -- ONLY these weights are learned

the random projection matrix is NEVER updated. only the tiny readout learns. biologically proven to match or beat locality-sensitive hashing for similarity search.

the parallel to ESC-50 is uncanny:
- fly: 50 odor types, ~50 projection neurons, ~2000 Kenyon cells, ~34 outputs
- us: 50 sound classes, ~2304 input features, could use ~10,000 expansion neurons, 50 outputs
- the problem dimensions are nearly IDENTICAL

could get 10-50x:
- random binary projection: no multiply-accumulate, just sparse fanout (free on SpiNNaker via multicast)
- WTA: single comparison pass
- linear readout: 10,000 x 50 = 500K ops, but only 5% non-zero = 25K effective ops
- total: ~25K ACs vs current ~1.08M ACs = ~43x reduction

adaptation:
- AvgPool spectrogram to 2304-dim vector (we already do this)
- fixed random sparse binary projection to 10,000-dim (each expansion neuron gets ~10 random inputs)
- WTA: keep top 5% active (500 neurons)
- train only 10,000 -> 50 linear readout
- on SpiNNaker: random projection IS the multicast routing table. WTA via lateral inhibition. only readout weights stored.

easy to implement. scikit-learn has SparseRandomProjection. could prototype in an afternoon. SpiNNaker mapping is natural.

"we replaced a 4-layer deep SNN with a fruit fly brain circuit and achieved X% accuracy at 43x lower energy" -- reviewers would remember that.

code exists: https://github.com/tian-kun/Fly-LSH

---

### idea 1.3: barn owl sound localization -- delay lines + coincidence detection

source: Ashida "Barn owl and sound localization" (Acoust. Sci. & Tech., 2015); Carruthers et al. "Neuromorphic object localization using resistive memories" (Nature Comms, 2022)

the barn owl localizes sound with 2-degree precision using:
1. delay lines -- axons of different lengths for precise time delays
2. coincidence detectors -- neurons that fire only when two inputs arrive simultaneously
3. place code -- position of maximally firing detector encodes source location

the CEA-Leti group built a neuromorphic barn-owl system with 5 orders of magnitude less energy than conventional systems.

adaptation for audio classification:
- environmental sounds have characteristic temporal patterns (knocking = periodic impulses at ~5 Hz, siren = sweeping frequency)
- replace learned convolutions with delay-line + coincidence detector banks tuned to each class's temporal patterns
- "knocking detector" = coincidence detector tuned to ~200ms delay
- "siren detector" = tuned to frequency-sweep rate

could get 10-100x (delay lines are passive, coincidence detection is just a threshold on summation).

SpiNNaker natively supports configurable synaptic delays so this would be a natural fit.

"bio-inspired delay-line coincidence detectors for environmental sound classification" -- hasnt been published.

---

### idea 1.4: cricket phonotaxis -- temporal pattern recognition with 4 neurons

source: Hennig et al. "Neuroethology of acoustic communication in crickets" (Progress in Neurobiology, 2020); Schoeneich et al. neuromorphic cricket (IEEE EMBS NER, 2013)

crickets recognize species-specific calling songs using just 4 neuron types. the circuit acts as a bandpass filter for temporal patterns -- no spectral analysis needed, just temporal pattern matching. published neuromorphic implementations exist.

adaptation:
- many ESC-50 sounds have characteristic temporal rhythms (clock tick, knocking, rooster)
- bank of 50 cricket-like temporal pattern detectors, each tuned to a different inter-pulse interval
- no spectral features needed for temporally distinctive sounds
- cheap first-stage: if temporal pattern is distinctive enough, skip spectral analysis

could get 5-20x for temporally distinctive classes.

"cricket-inspired temporal pattern detectors for environmental sound classification" -- novel.

---

### idea 1.5: insect brain scale classifier -- 3,016 neurons for everything

source: Winding et al. "The connectome of an insect brain" (Science, 2023); Rapp & Nawrot "Insect bio-inspired neural network" (PLoS Comp Bio, 2017)

the complete drosophila larval brain was mapped in 2023: 3,016 neurons and 548,000 synapses. this tiny brain navigates, forages, learns, and makes decisions. our 622K-parameter network has more parameters than a fruit fly larva has synapses. thats kind of embarrassing.

adaptation: use the connectome topology (publicly available now) as architecture inspiration:
- sparse, structured connectivity
- feedforward + lateral inhibition + recurrent loops
- target: ~3,000-10,000 neuron SNN

could get 50-200x (proportional to parameter reduction). but this would be a pretty hard project -- novel architecture design.

"we designed an SNN audio classifier with fewer neurons than a fruit fly larva brain" -- extraordinary claim if it works.

---

## category 2: weightless neural networks / lookup tables

### idea 2.1: differentiable weightless neural networks (DWN)

source: Bacellar et al. "Differentiable Weightless Neural Networks" (ICML 2024); "nanoML for HAR" (arXiv:2502.12173, 2025)

replace ALL neural computation with cascaded lookup tables (LUTs). input binarized via thermometer encoding. each tuple indexes a small RAM. layers of LUTs chained. training uses Extended Finite Difference for differentiating binary values.

demonstrated:
- 56 nJ per inference on Xilinx XC7Z020CLG400 FPGA at 199 MHz
- 5 nanoseconds per sample (one clock cycle!)
- 96.34% accuracy on UCI HAR
- 926,000x energy savings vs deep learning
- model fits entirely in FPGA LUT fabric -- no external memory

adaptation:
- binarize mel features via thermometer encoding
- spike patterns from SNN hidden layers are ALREADY binary -- natural DWN input
- train DWN to classify binary spike patterns -> 50 classes
- hybrid: SNN encodes temporal dynamics -> DWN classifies the spike code

could get 17-170x:
- DWN-only on raw binarized features: ~56-100 nJ vs 968 nJ = 10-17x
- hybrid SNN(T=3)+DWN: ~116 nJ + ~56 nJ = ~172 nJ = 5.6x
- DWN on PANNs embeddings (binarized): ~100 nJ, potentially higher accuracy

DWN code available: https://github.com/alanbacellar/DWN. PyTorch-compatible.

"we achieved competitive audio classification at 56 nJ using cascaded lookup tables with zero multiply-accumulate operations" -- pretty jaw-dropping.

---

### idea 2.2: LogicWiSARD -- memoryless weightless networks

source: "LogicWiSARD" (ASAP 2022); Susskind et al. "Weightless NNs for Efficient Edge Inference" (PACT 2022)

WiSARD uses RAM-based lookup tables per class (discriminators). LogicWiSARD goes further: compiles the learned lookup tables into pure logic gates, eliminating even the RAM. the classifier becomes a fixed combinational circuit.

>80% energy reduction vs standard WiSARD. real-time music tracking demonstrated.

our SNN already produces binary spike patterns. feed directly into WiSARD discriminator bank. one per ESC-50 class. then compile to logic gates.

could get 100-1000x (pure logic, no arithmetic, no memory access).

"from spiking neural network to pure logic gates: zero-arithmetic audio classification."

---

## category 3: logic gate networks

### idea 3.1: convolutional differentiable logic gate networks

source: Petersen et al. "Convolutional Differentiable Logic Gate Networks" (NeurIPS 2024, Oral); "Deep Differentiable Logic Gate Networks" (NeurIPS 2022)

learn WHICH logic gates (from 16 possible 2-input Boolean functions) to wire between binary signals. training uses differentiable relaxation where gate probabilities learned via gradient descent. at inference, highest-probability gate selected = fully Boolean network.

results:
- CIFAR-10: 86.29% with 61M logic gates (29-61x smaller than SOTA)
- FPGA: 4 nanoseconds, 41.6 million FPS
- deep logic gate tree convolutions, logical OR pooling, residual initializations
- library: https://github.com/Felix-Petersen/difflogic

adaptation:
- option A: binarize spectrogram, train logic gate network end-to-end
- option B: train SNN normally, distill into logic gate network
- option C: SNN temporal encoding (binary spikes) -> logic gate classification

could get 100-10,000x. logic gates consume femtojoules. at 61M gates: total ~0.1-1 nJ. even conservative: 100 nJ vs 968 nJ = 10x.

recurrent logic gate networks published in 2025 (arXiv:2508.06097) so handling temporal dimension is possible.

"audio classification in pure combinational logic at 4 nanoseconds" -- first application to audio. paradigm shift territory.

---

## category 4: compressed sensing / information theory

### idea 4.1: information-theoretic minimum for 50-class classification

source: rate-distortion theory; Shannon's channel coding theorem

how much information does 50-class classification actually NEED?
- Shannon entropy of 50 equiprobable classes: log2(50) = 5.64 bits
- at 47% accuracy: mutual information I(X;Y) ~ 2.6 bits
- at 90% accuracy: I(X;Y) ~ 4.7 bits

we need to reliably transmit ~5.64 bits through the network. our SNN spends 1.08M accumulate ops to communicate ~5.64 bits. thats an information efficiency of 5.2 x 10^-6 bits per operation. astronomically wasteful.

theoretical minimum energy:
- Landauer limit: kT ln(2) = 2.85 x 10^-21 J per bit at room temperature
- for 5.64 bits: ~1.6 x 10^-20 J = 0.016 attojoules
- our SNN: 968 nJ
- gap to thermodynamic limit: ~6 x 10^13 (60 trillion times above Landauer limit)

this is primarily a framing argument for the paper. any classifier that reliably communicates >5.64 bits could theoretically classify 50 classes. the question is what's the simplest circuit that can extract those 5.64 bits from a spectrogram.

framing audio classification as an info-theoretic channel capacity problem is uncommon in neuromorphic literature. reviewers would find it interesting.

---

### idea 4.2: compressed sensing SNN -- subsample then classify

source: CompSNN (Neurocomputing, 2021); general compressed sensing theory

compressed sensing says: if a signal is K-sparse in some basis, you can reconstruct from M = O(K log(N/K)) random measurements. for classification you dont even need reconstruction -- classify directly from compressed measurements.

our spectrogram: 64 x 216 = 13,824 features. saliency maps show effective sparsity around 100-500 bins. compressed sensing theory: ~500 random measurements should suffice for 50-class classification.

adaptation:
- randomly sample ~500-1000 spectro-temporal bins (FIXED random mask)
- tiny SNN (500 -> 128 -> 50 instead of 13,824 -> 2304 -> 256 -> 50)
- total params: ~70,400 (9x reduction)

could get 5-15x. easy to implement (random subsampling is trivial).

---

## category 5: signal processing tricks for audio

### idea 5.1: psychoacoustic masking to kill spikes

source: Pan & Chua "BAE: Biologically plausible Auditory Encoding" (Frontiers in Neuroscience, 2020)

the human auditory system cant perceive all frequencies simultaneously -- loud sounds mask nearby quiet sounds (simultaneous masking) and sounds mask subsequent quiet sounds (temporal masking). the BAE scheme applies psychoacoustic masking BEFORE spike encoding.

demonstrated results:
- 50.48% of spikes eliminated on TIDIGITS (speech)
- 39.38% reduction on RWCP (environmental sounds -- similar to ESC-50!)
- 97.4% accuracy maintained on TIDIGITS

adaptation:
- apply masking to mel spectrogram BEFORE direct encoding
- expected ~40% spike reduction for environmental sounds
- combined with our 26.4% rate: new rate = 26.4% * 0.6 = 15.8%
- further combined with spike reg to 6%: final = 6% * 0.6 = 3.6% (below Dampfhoffer threshold!)

1.7x standalone, but as a critical enabler to get below 6.4% spike rate when combined with reg.

psychoacoustic masking is well-implemented in audio processing libraries. easy.

---

### idea 5.2: sigma-delta neural networks -- only transmit changes

source: "Sigma-Delta Neural Network Conversion on Loihi 2" (arXiv:2505.06417, 2025)

standard SNNs transmit absolute values via spike rates each timestep. sigma-delta networks only transmit CHANGES. neuron maintains running sum (sigma), only sends spike when accumulated change (delta) exceeds threshold. for temporally correlated signals like audio, most activations change very little between timesteps.

demonstrated:
- 17x synaptic operation sparsity vs dense ANN
- Loihi 2: 0.55 mJ dynamic energy per inference
- ANN-to-SDNN conversion in 30 seconds post-training

audio spectrograms have MASSIVE temporal redundancy (adjacent frames differ by <5% for continuous sounds). expected 5-17x (rain/engine = very redundant = 17x; gunshot = transient = 2x; average ~8x).

conversion algorithm exists. need to verify with our model. not too hard.

sigma-delta for audio SNN is novel. the conversion path is elegant.

---

### idea 5.3: audio fingerprint / hash classification

adapted from Shazam/Chromaprint; "Can LSH be replaced by Neural Network?" (Soft Computing, 2024)

compute compact binary hash of input, compare via Hamming distance to 50 class prototypes.

- hash computation: small encoder (single FC layer) -> 256-bit hash
- comparison: 50 XOR + popcount operations
- total: one matrix multiply (13,824 x 256 bits) + 50 integer comparisons

could get 20-50x.

need to train a good hash function via contrastive learning. probably a few days.

"replacing neural network inference with 50 Hamming distance computations" -- novel.

---

## category 6: hierarchical / cascade classification

### idea 6.1: coarse-to-fine cascade (ESC-50 super-categories)

source: "ECHO" (arXiv:2409.14043, 2024); our per-category SpiNNaker data

ESC-50 has 5 super-categories: Animals, Nature, Human, Domestic, Urban. tiny stage-1 (5-class) + category-specific stage-2 (10-class each).

our data supports this -- SpiNNaker Run 6 super-category accuracies: Animals 45%, Nature 61.3%, Human 46.2%, Domestic 31.2%, Urban 31.2%. most confusion is WITHIN categories, not between.

architecture:
- stage 1: tiny SNN (~10K params, T=5) -> 5 super-categories
- stage 2: 5 specialized tiny SNNs (~20K each)
- stage 1 energy: ~10K/622K * 968 * 5/25 = ~3.1 nJ
- stage 2: ~20K/622K * 968 * 10/25 = ~12.5 nJ
- total: ~15.6 nJ for cascade vs 968 nJ = 62x reduction

could get 10-60x. easy to implement (train 6 small models).

### idea 6.2: mixture of tiny SNN experts

source: conditional computation survey (Scardapane et al., 2024); Phi (arXiv:2505.10909, 2025)

instead of one 622K model, gating network selects which expert sub-network to activate. each expert specializes in a subset of classes.

- tiny gating SNN (~5K params): first 3 timesteps -> selects 2 of 10 experts
- 10 expert SNNs (~30K params each): each specializes in 5 classes
- total: 5K + 10*30K = 305K, but only 65K active per inference = 10.5% of current

could get 5-10x from activation reduction. combined with early exit: 15-30x.

"mixture of spiking experts" for audio hasnt been published. reviewers would like this.

---

## category 7: extreme compression

### idea 7.1: KD to sub-10K model

source: "Efficient Speech Command Recognition via SNN and KD" (arXiv:2412.12858, 2024); "Dynamic Activation with KD" (arXiv:2502.14023, 2025)

proposed student:
- Conv1: 1->4, 3x3 (36 params)
- MaxPool(2)
- Conv2: 4->8, 3x3 (288 params)
- MaxPool(2)
- AvgPool(4,6)
- FC1: 288->32 (9,216 params)
- FC2: 32->50 (1,600 params)
- total: ~11,140 params (56x smaller)

could get 30-56x from size reduction. with T=7: additional 3.5x.

but we know KD hurts everything for us... so im skeptical. maybe with specialized SNN KD methods. filing under "maybe."

### idea 7.2: binary/ternary weight SNN

source: "Memory Efficient Audio Classification Using BNN" (2025); "Sound Event Detection with BNNs on IoT" (arXiv:2101.04446)

constrain weights to {-1, 0, +1}. our SNN already has binary spikes. with ternary weights: MAC becomes add/subtract/skip. sound event detection with BNNs has been demonstrated on IoT devices.

could get 5-16x.

ternary QAT is well-established. "ternary-weight SNN for audio on SpiNNaker" is modestly novel.

### idea 7.3: depthwise separable spiking convolutions

source: "Spike-TCN with Depthwise-Separable Conv" (Springer, 2025); CVPR 2025 brain-inspired SNNs

replace Conv2d(32,64,3) with DepthwiseConv(32,32,3) + PointwiseConv(32,64,1). reduces Conv2 ops from ~63M to ~8M (8x in most expensive layer).

could get 3-5x. easy -- PyTorch nn.Conv2d(groups=in_channels).

well-known technique. not groundbreaking.

---

## category 8: radar/sonar signal processing

### idea 8.1: matched filter bank for sound detection

source: matched filter theory (classical signal processing); spectrogram-correlation-tutorial (GitHub)

in radar, matched filter is the OPTIMAL detector for a known signal in Gaussian noise. for audio:
1. pre-compute "template spectrogram" for each of 50 classes (average of training examples)
2. at inference: 2D cross-correlate input with all 50 templates
3. classification = highest correlation

with downsampled templates (16x54 = 864 pixels): 50 * 864 = 43,200 MACs = 199 nJ. with binary templates (threshold): XOR + popcount = ~20 nJ.

accuracy: template matching on ESC-50 probably 25-35%. random forest with MFCC gets 44.3% for comparison.

very easy to implement. average training spectrograms per class, binarize, XOR at inference.

"radar-inspired matched filter classification of environmental sounds at 20 nJ" -- novel framing.

### idea 8.2: correlation-based binary classifier bank (sonar-inspired)

instead of full-spectrogram templates, extract discriminative sub-templates:
- "dog bark": template = 0.5s x 8 mel bins covering bark's spectral signature
- "siren": template = 2s x 4 mel bins covering frequency sweep

could get 100x+ since each mini-template correlation is tiny.

need to identify discriminative sub-templates per class. more work but "sonar-inspired sub-template matching" sounds cool.

---

## category 9: video compression analogies

### idea 9.1: I-frame / P-frame spectrogram processing

in video compression: I-frames (key frames) fully encoded, P-frames only encode differences from previous. 90-95% of frames are P-frames, reducing data 50-100x.

audio spectrograms are time-frequency images. same logic:
- timestep 0 (I-frame): full spectrogram through SNN
- timesteps 1-24 (P-frames): only process delta from previous

most mel bins change <5% between adjacent timesteps for continuous sounds. related to sigma-delta encoding but framed at the input level.

could get 5-20x (sound-type dependent, average ~8x). easy to implement.

### idea 9.2: only process "interesting" temporal regions (attention gating)

our temporal ablation shows T=7 reaches 90% accuracy. but which 7 timesteps matter? the "interesting" ones (onset, transitions, offset) carry most info.

cheap energy-based attention gate: only process timesteps where energy exceeds threshold OR changes significantly. for "dog barking": actual barks occupy ~1-2 seconds, skip the rest. on SpiNNaker: event-driven = zero cost for skipped timesteps.

could get 3-10x. very easy to implement.

---

## category 10: thermodynamic / physical / unconventional computing

### idea 10.1: photonic spiking neural network

source: Optics Express (2022); MIT photonic processor (2024); Nature Comms (2025)

use photons instead of electrons. recent results:
- 0.005 attojoules per MAC (0.013 photons!) at 92% on MNIST
- femtojoule per neuron op in photonic spiking networks
- MIT: ultrafast AI with extreme energy efficiency

our SNN at 0.005 aJ/MAC with 1.08M ACs: 5.4 picojoules per inference. thats 180,000x reduction.

obviously a hardware change not algorithm change. but framing our SNN as "photonic-ready" is valuable since binary spikes map to photons, sparse computation means fewer sources, temporal coding maps to time-of-flight.

cant actually implement this. but the argument is free.

### idea 10.2: stochastic computing -- multiply with AND gates

source: Electronics 15(4):768 (2025); Research (2024)

represent numbers as random bitstreams. multiplication = AND gate. the isomorphism with SNN spike trains is EXACT:
- SC bitstream with probability p = SNN spike train with firing rate p
- SC AND gate = SNN weight multiplication
- SC MUX = SNN synaptic summation

our SNN IS already a stochastic computer! interesting theoretical framing.

could get 3-5x on FPGA. combined with sparsity: 5-15x.

requires FPGA. but the SNN-SC isomorphism is an elegant argument for ICONS.

### idea 10.3: memristor crossbar array for SNN

source: Shooshtari et al. "Review of Memristors for In-Memory Computing and SNNs" (Advanced Intelligent Systems, 2026)

memristor crossbars do matrix-vector multiply in ANALOG using Ohm's law. weight matrix stored AS resistance values. inference = apply voltages, read currents. no digital computation.

our 1.08M ACs at 4.28 aJ each: ~4.6 nJ per inference (210x reduction).

map our weight matrices to crossbar arrays. binary spikes as input voltages. conceptually clean.

requires custom hardware. theoretical/future-work framing.

### idea 10.4: physical reservoir computing -- computation in matter

source: Fernando & Sojakka "Pattern Recognition in a Bucket" (2003); Hopf reservoir (Nature Sci Rep, 2023); Skyrmion reservoir (2023)

use a physical system (water, springs, magnetic skyrmions, even LEGO) as computational reservoir. input perturbs the system, complex dynamics separate inputs, simple readout classifies.

demonstrated for audio:
- water bucket: vowel recognition from sound-induced ripple patterns
- Hopf oscillator reservoir: sound recognition without preprocessing
- skyrmion reservoir: 97.4% spoken digit classification

"we demonstrated that our SNN classifier could theoretically be replaced by a bucket of water" -- genuinely mind-bending for a discussion section.

practically cant implement this in a thesis. but the concept is powerful.

### idea 10.5: hyperdimensional computing for audio

source: DAC 2018; torchhd library

10,000-bit binary hypervectors. encode with XOR binding, bundle with majority vote. classify via Hamming distance to 50 prototypes. ALL bitwise.

39.4 nJ per prediction demonstrated.

could get 10-25x. torchhd library available. need to design spectrgram encoding.

HDC for environmental sound is completely unexplored.

### idea 10.6: self-powered neuromorphic audio sensor

source: "Self-aware artificial auditory neuron with triboelectric sensor" (Nano Energy, 2023)

TENG as both audio sensor AND power source. sound waves generate sensing signal AND electricity. external energy = zero.

"our neuromorphic audio classifier is powered entirely by the sound it is classifying." -- the ultimate ICONS argument.

requires custom hardware though. just a discussion point.

---

## bonus: hybrid and exotic approaches

### idea 11.1: predictive coding light -- only transmit surprise

source: Nature Communications (2025); survey arXiv:2409.05386

PCL suppresses predictable spikes, transmits compressed representation. each layer maintains internal prediction; only surprising deviations generate spikes.

for audio: environmental sounds are highly predictable over short windows. running engine/rain/AC has nearly constant spectral content for seconds. after 2-3 timesteps PCL builds internal model, subsequent timesteps only process departures. continuous sounds: near-zero spikes after initial transient.

could get 5-20x (continuous sounds 15-20x, transient 2-3x).

Nature Comms 2025 paper + first audio application = reviewers would like this.

### idea 11.2: bloom filter classification

source: DATE 2018; "Weightless NNs as Memory Segmented Bloom Filters" (Neurocomputing, 2020)

bloom filters for set membership. during training: build bloom filter per class from binary feature patterns. during inference: test input against all 50 filters. lookup = k hash computations + k memory reads. with k=3: ~3-5 nJ total.

BTHOWeN uses counting bloom filters with learned hash functions. 47.5% energy savings with 1% accuracy drop.

could get 50-200x. "bloom filter classification of environmental sounds" is deeply unconventional.

### idea 11.3: DNA / molecular computing

source: "Supervised learning in DNA neural networks" (Nature, 2025); Nature Comms (2025)

DNA strand displacement cascades implementing winner-take-all networks, decision trees, multi-class classification. DNA molecules autonomously performing supervised learning in vitro.

purely speculative for audio but the argument is powerful: "if DNA can classify patterns, energy requirements are far below any electronic system." framing argument for discussion.

---

## priority ranking for ICONS 2026

ranked by (energy_reduction x novelty x feasibility):

| rank | idea | energy | novelty | feasibility |
|------|------|--------|---------|-------------|
| 1 | fly olfactory circuit | 10-50x | very high | high |
| 2 | DWN lookup tables | 17-170x | extreme | medium |
| 3 | moth cascade gate | 10-1000x | extreme | high |
| 4 | logic gate networks | 100-10000x | extreme | medium |
| 5 | coarse-to-fine cascade | 10-60x | medium | high |
| 6 | psychoacoustic masking | 1.7x (critical enabler) | medium | high |
| 7 | matched filter bank | 50x | high | high |
| 8 | sigma-delta conversion | 5-17x | medium-high | medium |
| 9 | HDC for audio | 10-25x | high | medium |
| 10 | predictive coding light | 5-20x | very high | medium |

---

## recommended combinations for 10-100x

### the "practical 50x" stack (implementable before ICONS deadline):
1. spike rate reg to 6% (4.4x) -- already have code
2. adaptive early exit T=25 -> avg T=7 (3.5x) -- 1 day
3. psychoacoustic masking on input (1.7x) -- existing algorithms
4. total: 4.4 x 3.5 x 1.7 = 26x -> ~37 nJ

### the "surprising 100x" stack (for discussion section):
1. fly olfactory circuit (10x) + WTA sparsity (5x) = 50x
2. + early exit (2x) = 100x -> ~10 nJ

### the "future vision 1000x" stack (for future work):
1. DWN/logic gate hardware (100x)
2. photonic implementation (1000x)
3. self-powered sensor (infinite)
4. result: sub-picojoule classification

---

## what makes these surprising for reviewers

1. biology pulls: moth (1 neuron!), fly (random projection), barn owl (delay lines), cricket (4 neurons). these are REAL working systems.
2. zero-arithmetic approaches: DWN (lookup tables), logic gates (Boolean), HDC (bitwise), matched filters (XOR+popcount). classification without a single multiply.
3. info-theoretic framing: 50 classes needs only 5.64 bits. we spend 968 nJ to communicate 5.64 bits. 60 trillion times above Landauer limit.
4. cross-pollination: video compression for spectrograms, radar matched filters for sound, bloom filters for classification, stochastic computing for synaptic ops.
5. the fly circuit parallel: 50 odors <-> 50 sounds, random projection <-> SpiNNaker multicast, WTA <-> lateral inhibition. the mapping is too perfect to ignore.

---

## references

### biology
- Dasgupta, Stevens, Navlakha. "A neural algorithm for a fundamental computing problem." Science 358:793-796 (2017). [link](https://www.science.org/doi/full/10.1126/science.aam9868)
- Ratcliffe et al. "Tiger moths and the threat of bats." Biology Letters 5(3):368-371 (2009). [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC2679932/)
- Roeder. "Auditory System of Noctuid Moths." Science 154:1515-1521 (1966).
- Ashida. "Barn owl and sound localization." Acoust. Sci. & Tech. 36(4):275-285 (2015).
- Carruthers et al. "Neuromorphic object localization using resistive memories." Nature Comms 13:3506 (2022).
- Hennig et al. "Neuroethology of acoustic communication in crickets." Prog. Neurobiol. 195:101882 (2020).
- Winding et al. "The connectome of an insect brain." Science 379:eadd9330 (2023).
- Zheng et al. "Fly-CL." arXiv:2510.16877 (2024).

### weightless / lookup table
- Bacellar et al. "Differentiable Weightless Neural Networks." ICML 2024.
- Bacellar et al. "nanoML for HAR." arXiv:2502.12173 (2025).
- LogicWiSARD. ASAP 2022.
- Susskind et al. "Weightless NNs for Efficient Edge Inference." PACT 2022.
- DWN code: https://github.com/alanbacellar/DWN

### logic gate networks
- Petersen et al. "Convolutional Differentiable Logic Gate Networks." NeurIPS 2024 (Oral).
- Petersen et al. "Deep Differentiable Logic Gate Networks." NeurIPS 2022.
- Code: https://github.com/Felix-Petersen/difflogic
- Recurrent variant: arXiv:2508.06097 (2025).

### signal processing
- Pan & Chua. "BAE." Frontiers in Neuroscience 14:1420 (2020).
- SDNN on Loihi 2. arXiv:2505.06417 (2025).
- Spectrogram correlation: https://github.com/sebastianmenze/Spectrogram-correlation-tutorial

### unconventional computing
- "Predictive Coding Light." Nature Comms (2025).
- Stochastic computing + SNN isomorphism. Electronics 15(4):768 (2025).
- Photonic SNN. Optics Express (2022).
- MIT photonic processor (2024).
- Quantum-limited optical NN. Nature Comms (2025).
- Hopf physical reservoir. Sci. Rep. (2023).
- Fernando & Sojakka. "Pattern Recognition in a Bucket." (2003).
- DNA learning. Nature (2025).

### hyperdimensional computing
- "Hierarchical HDC for Energy Efficient Classification." DAC 2018.
- Torchhd: https://github.com/hyperdimensional-computing/torchhd

### hierarchical / cascade
- "ECHO: ESC with Hierarchical Ontology." arXiv:2409.14043 (2024).
- Conditional computation survey. Scardapane et al. (2024).

### extreme compression
- "Efficient Speech Command Recognition via SNN and KD." arXiv:2412.12858 (2024).
- "Memory Efficient Audio Classification Using BNN." (2025).
- "Sound Event Detection with BNNs on IoT." arXiv:2101.04446.
- "Spike-TCN with Depthwise-Separable Conv." Springer (2025).

### information theory
- Shannon. "A Mathematical Theory of Communication." (1948).
- Landauer. "Irreversibility and Heat Generation in the Computing Process." (1961).
- Rate-distortion theory of neural coding. PLoS Comp. Bio. (2023).
