# creative energy reduction ideas -- brainstorming notes

date: 26 march 2026
context: ~622K param SNN for ESC-50, deployed on SpiNNaker. current energy: ~968 nJ/sample (SNN), ~454 nJ/sample (ANN). sparsity: 73.6%. T=25 timesteps. direct encoding.

these are 25 unconventional ideas for dramatically reducing inference energy. some are practical, some are pretty out there. wrote most of these down after reading a ton of papers late at night so some might not make total sense in the morning.

the ones i'm most excited about:
1. weightless neural networks (DWNs) -- replace the SNN with lookup tables. 56 nJ per sample demonstrated. 5 ns inference. up to 926,000x savings over deep learning.
2. logic gate networks -- learned NAND/OR/XOR gates. 24 ns inference on FPGA. NeurIPS 2024 oral.
3. single-timestep multi-level SNN (T=1) -- collapse 25 timesteps to 1 using multi-level neurons. 66% energy reduction demonstrated.
4. fly olfactory circuit -- random projection + winner-take-all. biologically proven for classification.

2. **Logic Gate Networks** -- replace neural computation with learned combinations of NAND/OR/XOR gates. 24 ns inference on FPGA. NeurIPS 2024 oral paper.

3. **Single-Timestep Multi-Level SNN (T=1)** -- collapse 25 timesteps to 1 using multi-level neurons that encode information in spike amplitude. 66% energy reduction demonstrated. Direct applicability.

4. **Fly Olfactory Circuit Architecture** -- replace 4-layer SNN with random projection + winner-take-all in one layer. Biological proof that classification needs no learned feature extraction.

5. **Hierarchical Coarse-to-Fine Cascade** -- classify the super-category first (5 classes), then the fine class (10 within category). Most inputs exit at the cheap first stage.

The ideas below are ordered roughly from most directly applicable to most speculative.

---

## IDEA 1: Single-Timestep Multi-Level SNN (T=1)

**Source:** "All in One Timestep: Enhancing Sparsity and Energy Efficiency in Multi-level Spiking Neural Networks" (arXiv:2510.24637, 2024); "One-Timestep is Enough: Scale-and-Fire Neurons" (arXiv:2510.23383, 2024)

**Core mechanism:** Standard LIF neurons output binary {0,1} and need T=25 timesteps to accumulate enough information. Multi-level neurons output integers {0,1,...,N} in a single timestep, encoding N bits of information where previously you needed N binary timesteps. A 4-level neuron at T=1 is mathematically equivalent to a binary neuron at T=4.

**Application to our system:**
- Our temporal ablation shows T=7 reaches 90% of full accuracy (36.5% vs 40.5% at T=25)
- With 4-level neurons at T=1, we could match T=4 performance
- With 8-level neurons at T=1, we could approach T=8 performance
- Scale-and-Fire variant: learn adaptive thresholds via Bayesian optimization, convert ANN directly to T=1 SNN

**Estimated energy reduction:** 15-25x (from T=25 to T=1). Multi-level ops cost ~2-3x per timestep but only 1 timestep total.

**Accuracy impact:** T=1 binary gives 7.25% on our model. With 4-level neurons, expect ~25-30%. With 8-level, expect ~35%. Still a gap vs T=25 (47.15%) but dramatic energy win.

**Can it run on SpiNNaker?** Partially. SpiNNaker 1 only supports binary spikes natively. Would need to encode multi-level spikes as multiple rapid binary spikes within a timestep, or use SpiNNaker 2's extended features. Alternatively, implement as a quantized ANN on SpiNNaker's ARM cores.

**Implementation difficulty:** Medium. The ANN-to-SNN conversion path is well-documented. snnTorch would need custom neuron model.

**Novelty rating:** Medium-high. Applied to audio for the first time. "We compress 25 timesteps into 1 with only 15% accuracy loss" is a strong claim.

---

## IDEA 2: Temporal Early Exit (Input-Adaptive Timesteps)

**Source:** "SEENN: Towards Temporal Spiking Early-Exit Neural Networks" (arXiv:2304.01230, 2023); "SPARQ: Spiking Early-Exit Neural Networks for Energy-Efficient Edge AI" (arXiv:2603.14380, March 2026)

**Core mechanism:** Not all sounds need T=25 to classify. A dog bark might be identifiable at T=3 while distinguishing between two types of engine noise might need T=20. Monitor confidence at each timestep; exit when confident enough.

**Application to our system:**
- Our temporal ablation: T=5 already gets 33.5%, T=10 gets 38.25%
- Many ESC-50 classes have distinctive onsets (dog bark, gunshot, clock alarm)
- Train a lightweight confidence estimator on membrane potential / output spike counts
- RL-guided: SPARQ uses Q-learning to learn per-class exit policies

**Estimated energy reduction:** 3-5x on average (most samples exit at T=5-10, hard samples go to T=25)

**Accuracy impact:** <1% loss with calibrated thresholds. SPARQ reports 5.15% accuracy improvement over baselines while reducing operations by 96%.

**Can it run on SpiNNaker?** Yes. SpiNNaker already processes timestep-by-timestep. Add a simple confidence check after each timestep's output layer processing. When confidence exceeds threshold, stop sending further input spikes.

**Implementation difficulty:** Low-Medium. Could implement as a post-hoc wrapper around existing trained model.

**Novelty rating:** Medium. Well-studied for vision, novel for audio SNN classification.

---

## IDEA 3: Fly Olfactory Circuit -- Random Projection + Winner-Take-All

**Source:** Dasgupta et al. "A neural algorithm for a fundamental computing problem" (Science 2017); Zheng et al. "Fly-CL: A Fly-Inspired Framework" (arXiv:2510.16877, 2024)

**Core mechanism:** The fruit fly classifies ~50 odors using a 3-stage circuit:
1. 50 projection neurons (input) -- analogous to our mel features
2. Random sparse projection to 2000 Kenyon cells (40x expansion, only 5% active via WTA)
3. 34 output neurons with simple learned weights

The random projection is FIXED -- no learning. Only the output weights are learned. This is biologically proven to work as well as or better than locality-sensitive hashing for classification.

**Application to our system:**
- Our mel spectrogram has 64x216 = 13,824 features per frame
- Random sparse projection to ~50,000-dimensional space (4x expansion)
- Winner-take-all: keep only top 5% active (like fly's APL inhibition)
- Train only a 50,000 -> 50 linear readout
- The random projection matrix is FIXED and can be hardcoded in hardware

**Estimated energy reduction:** 10-50x. The random projection needs no multiply-accumulate (just sparse binary connections). WTA is a single comparison pass. Only the tiny readout needs real computation.

**Accuracy impact:** Unknown for audio. For odor classification (analogous problem), fly algorithm matches or beats LSH. For ESC-50 with 50 classes (same as fly's ~50 odor types!), this is eerily well-matched. Expect 20-35% on raw features, potentially higher with learned features (PANNs embeddings).

**Can it run on SpiNNaker?** Perfectly. SpiNNaker's multicast routing IS the random projection -- one input spike fans out to random subset of Kenyon cells. WTA is implementable via lateral inhibition. This is what SpiNNaker was designed for.

**Implementation difficulty:** Low. Random sparse binary matrix + linear readout. Could prototype in an afternoon.

**Novelty rating:** Very high. "We replaced a 622K-parameter deep SNN with a fly brain circuit and achieved X% accuracy on ESC-50" would be a memorable paper line.

---

## IDEA 4: Weightless Neural Networks (Lookup Table Classification)

**Source:** "Differentiable Weightless Neural Networks" (ICML 2024, Bacellar et al.); "nanoML for Human Activity Recognition" (arXiv:2502.12173, 2025); "LogicWiSARD" (ASAP 2022)

**Core mechanism:** Replace all neural computation with cascaded lookup tables (LUTs). Input is binarized into n-bit tuples. Each tuple indexes into a small RAM that outputs a binary value. Layers of LUTs are chained. No multiplication. No addition. Just memory reads.

**Application to our system:**
- Binarize mel spectrogram features (or use spike patterns directly -- already binary!)
- Form n-tuples from input bits
- Cascade 2-3 layers of LUT lookups
- Final discriminator per class counts matching patterns

**Estimated energy reduction:** 10,000-926,000x. DWNs demonstrated 56 nJ per inference for HAR. Our SNN uses 968 nJ. A DWN could potentially classify audio at 50-100 nJ, roughly 10-20x less than our SNN.

**Accuracy impact:** DWNs achieve 96.3% on HAR (competitive with deep learning). For ESC-50 with 50 classes, expect lower accuracy (maybe 25-40%) but the energy savings are extraordinary.

**Can it run on SpiNNaker?** Not natively (SpiNNaker is optimized for spike computation, not LUT cascades). But SpiNNaker's ARM cores could implement LUT lookups, and the entire model would fit in local SRAM. Alternatively, this is ideal for FPGA deployment. 5 ns inference demonstrated.

**Implementation difficulty:** Medium. DWN code available (PyTorch-compatible). Would need custom encoding of audio features into binary tuples.

**Novelty rating:** Extreme. "We replaced a spiking neural network with cascaded lookup tables and achieved competitive classification at 56 nJ" is jaw-dropping if it works.

---

## IDEA 5: Logic Gate Networks

**Source:** "Convolutional Differentiable Logic Gate Networks" (NeurIPS 2024 Oral, Petersen et al.)

**Core mechanism:** Instead of learning weights for multiply-accumulate operations, learn which logic gates (NAND, OR, XOR, AND, etc.) to wire between binary signals. The network IS the hardware -- no abstraction layer between model and silicon. Inference = executing a circuit of logic gates.

**Application to our system:**
- Binarize spike patterns from the SNN
- Learn a logic gate network that maps binary spike patterns -> 50 class labels
- Inference on FPGA: 24 nanoseconds, 41.6 million samples/second
- Model size: 29-61x smaller than SOTA

**Estimated energy reduction:** 100-1000x. Logic gates consume femtojoules. 24 ns inference time means negligible dynamic power.

**Accuracy impact:** On CIFAR-10 (comparable complexity to ESC-50): 86.29% with 61M logic gates. For our 50-class audio: expect 30-50%.

**Can it run on SpiNNaker?** No (SpiNNaker lacks programmable logic fabric). Ideal for FPGA or custom ASIC. But the concept of "distill SNN to logic gates" is hardware-agnostic.

**Implementation difficulty:** Medium-High. Would need to train the logic gate network to mimic SNN behavior.

**Novelty rating:** Extreme. "Neural network inference in pure combinational logic" is a paradigm shift. First application to audio SNN would be novel.

---

## IDEA 6: Learned Audio Fingerprint / Hash-Based Classification

**Source:** PeakNetFP (arXiv:2506.21086); SAMAF (ACM TOMM 2020); "Can LSH be replaced by Neural Network?" (Soft Computing 2024)

**Core mechanism:** Instead of running a full forward pass, compute a compact binary hash (128-256 bits) of the input spectrogram. Compare this hash via Hamming distance to 50 pre-computed class prototypes. Classification = argmin of 50 Hamming distances. Hamming distance is a single XOR + popcount instruction.

**Application to our system:**
- Train a small encoder to produce 256-bit hash from mel spectrogram
- Store 50 prototype hashes (one per ESC-50 class)
- At inference: encode input -> 50 XOR+popcount operations -> argmin
- Total: one small encoding + 50 integer comparisons

**Estimated energy reduction:** 50-200x. The hash encoding can be a tiny network (even a single layer). The 50 comparisons are trivially cheap.

**Accuracy impact:** Depends on hash quality. With a well-trained encoder, expect 30-45% (competitive with our SNN). Audio fingerprinting achieves near-perfect matching with 128-bit hashes.

**Can it run on SpiNNaker?** Partially. The hash computation could run on SpiNNaker. Hamming distance comparisons on ARM cores. But better suited for general-purpose hardware.

**Implementation difficulty:** Medium. Could use contrastive learning to train the hash encoder.

**Novelty rating:** High. "Replacing neural network inference with 50 Hamming distance computations" for audio classification is genuinely novel.

---

## IDEA 7: Hierarchical Coarse-to-Fine Cascade

**Source:** "ECHO: Environmental Sound Classification with Hierarchical Ontology" (arXiv:2409.14043, 2024); "Coarse-to-Fine: A hierarchical DNN inference framework for edge computing" (2024)

**Core mechanism:** ESC-50 has 5 super-categories (Animals, Nature, Human, Domestic, Urban) with 10 classes each. Use a tiny first-stage classifier (5-class) followed by a category-specific second-stage classifier (10-class). Most energy is saved because the first stage is much cheaper, and the second stage is 10-class (not 50-class).

**Application to our system:**
- Stage 1: Tiny SNN (maybe 10K params) classifies into 5 super-categories
- Stage 2: 5 specialized tiny SNNs (each ~30K params) for fine classification
- Easy samples may not even need Stage 2 if the super-category is sufficient
- Our per-category analysis shows some categories (Animals, Nature) are much easier than others

**Estimated energy reduction:** 3-10x. Stage 1 is ~5-10x cheaper than full model. Many samples exit early. Stage 2 specialized models are simpler.

**Accuracy impact:** Could actually IMPROVE accuracy. Each specialized sub-model focuses on distinguishing 10 similar sounds rather than 50 diverse ones. Our confusion matrix shows most errors are within categories.

**Can it run on SpiNNaker?** Yes. Two-stage pipeline with conditional routing. SpiNNaker supports dynamic population activation.

**Implementation difficulty:** Low. Train 6 models instead of 1. Route based on Stage 1 output.

**Novelty rating:** Medium. Hierarchical classification exists but hasn't been applied to ESC-50 with SNN + SpiNNaker.

---

## IDEA 8: Reservoir Computing / Liquid State Machine

**Source:** "EARL: Energy-Aware Optimization of Liquid State Machines" (arXiv:2601.05205, 2026); "Deep Echo State Network for Environmental Sound Classification" (Springer 2024); LSM on SpiNNaker (validated)

**Core mechanism:** Replace the trained convolutional layers with a RANDOM, FIXED recurrent spiking network (the "reservoir" or "liquid"). Only train a linear readout from the reservoir state to the 50 classes. The reservoir separates inputs through its complex dynamics -- no backpropagation needed.

**Application to our system:**
- Random recurrent SNN of ~500-2000 spiking neurons (fixed weights)
- Feed mel spectrogram spikes into reservoir
- Read out reservoir state at the end
- Train only a 2000->50 linear layer (100K parameters, no backprop through reservoir)
- The random reservoir can be HARDCODED in SpiNNaker routing tables

**Estimated energy reduction:** 5-20x. No gradient computation for reservoir. Random weights = no weight memory updates. Only the tiny readout needs learning.

**Accuracy impact:** Deep ESN on ESC-10 achieved competitive results. For ESC-50 (harder), expect 15-30% accuracy. With optimized reservoir (EARL framework), potentially higher.

**Can it run on SpiNNaker?** Perfectly. LSMs are one of SpiNNaker's native use cases. The random recurrent connectivity maps naturally to SpiNNaker's routing. Several published implementations exist.

**Implementation difficulty:** Low. sPyNNaker has built-in support for random connectivity. EARL framework provides optimization.

**Novelty rating:** High. "We achieved X% accuracy on ESC-50 with NO TRAINING of the spiking layers" is a strong narrative.

---

## IDEA 9: Extreme Pruning to 95-99% Sparsity (Lottery Tickets for SNN)

**Source:** "Exploring Lottery Ticket Hypothesis in Spiking Neural Networks" (ECCV 2022, arXiv:2207.01382); "QP-SNN: Quantized and Pruned SNN" (ICLR 2025)

**Core mechanism:** Our pruning experiment shows SNN retains 93.2% accuracy at 90% pruning. The lottery ticket hypothesis suggests there exists a tiny subnetwork (~1-5% of original) that performs as well as the full network. Find it via iterative magnitude pruning.

**Application to our system:**
- Current: 622K parameters, 90% pruning retains ~38% accuracy (93.2% of baseline 40.5%)
- Push to 95%: expect ~35% accuracy (based on our curve)
- Push to 99%: ~6.2K parameters remaining. Accuracy may drop to 25-30% but energy savings are massive
- Combined with quantization (QP-SNN approach): 2-4 bit weights + 99% pruning = extreme compression

**Estimated energy reduction:** 10-100x at 99% sparsity. Only 6.2K of 622K weights are non-zero. SpiNNaker can skip all zero-weight synaptic operations.

**Accuracy impact:** At 95%: ~35%. At 99%: ~25-30%. Still useful for many applications.

**Can it run on SpiNNaker?** Yes. Sparse connectivity is native to SpiNNaker. Fewer synapses = fewer routing table entries = less packet traffic = less energy.

**Implementation difficulty:** Medium. Iterative magnitude pruning + fine-tuning. Need to adapt for SpiNNaker deployment.

**Novelty rating:** Medium. LTH for SNN exists, but ESC-50 application with SpiNNaker deployment is new.

---

## IDEA 10: Predictive Coding -- Only Process the Unexpected

**Source:** "Predictive Coding Light" (Nature Communications 2025); "Predictive coding is a consequence of energy efficiency" (Cell Patterns 2023); "Predictive Coding with Spiking Neural Networks: a Survey" (arXiv:2409.05386)

**Core mechanism:** The brain doesn't process every incoming signal. It maintains a prediction of what it expects and ONLY processes the prediction error (the "surprise"). For audio: if the spectrogram at time t is similar to time t-1, don't process it. Only transmit the difference.

**Application to our system:**
- Environmental sounds have high temporal redundancy (a dog barking has repeating patterns)
- Build a simple prediction model of the spectrogram's next frame
- Only encode and process frames that significantly differ from prediction
- This naturally produces SPARSE spike trains (only "surprise" frames generate spikes)

**Estimated energy reduction:** 5-20x depending on sound type. Continuous sounds (engine, rain) are highly predictable = very sparse. Transient sounds (gunshot) produce brief spike bursts.

**Accuracy impact:** Minimal if prediction model is good. The "surprise" signal contains all classification-relevant information by definition.

**Can it run on SpiNNaker?** Yes. Prediction can be implemented as a simple recurrent connection. Error computation is subtraction (excitatory vs inhibitory spikes).

**Implementation difficulty:** Medium. Need to design prediction mechanism for spectrograms. Delta encoding (which we already have!) is a simple form of this.

**Novelty rating:** High. Predictive coding SNN for audio on neuromorphic hardware would be genuinely novel.

---

## IDEA 11: Spike Latency Coding with Delay Learning

**Source:** "Delay learning based on temporal coding in SNNs" (Neural Networks 2024); "Learnable axonal delay in SNNs improves spoken word recognition" (Frontiers 2023); "A Latency Coding Framework for Deep SNNs with Ultra-Low Latency" (arXiv:2603.23206, 2026)

**Core mechanism:** Instead of rate coding (many spikes = large value), use TIME-TO-FIRST-SPIKE coding (earlier spike = larger value). Each neuron fires at most ONE spike. Information is in the timing, not the count. This is maximally sparse: exactly N spikes for N neurons.

Combined with delay learning: instead of learning weights, learn DELAYS on synaptic connections. A spike arriving with the right delay at the right time is equivalent to a weight -- but uses zero multiply-accumulate operations.

**Application to our system:**
- Convert to TTFS coding: each input neuron fires exactly once (timing encodes mel bin value)
- Learn optimal delays on connections (replaces weight multiplication)
- Total spikes per inference: exactly N_input + N_hidden + N_output (deterministic, minimal)
- With 256 hidden neurons: ~2600 + 256 + 50 = ~2906 total spikes per inference

**Estimated energy reduction:** 5-15x. Exact spike count is known and minimal. No rate accumulation needed. Delays can be hardcoded.

**Accuracy impact:** TTFS-based SNNs achieve comparable accuracy to rate-coded SNNs on vision benchmarks. For audio, recent work shows improved spoken word recognition with learnable delays.

**Can it run on SpiNNaker?** Partially. SpiNNaker supports synaptic delays natively (configured per synapse). TTFS decoding requires precise timing, which SpiNNaker's 1ms timestep may limit. SpiNNaker 2 has finer temporal resolution.

**Implementation difficulty:** Medium-High. Requires retraining with TTFS encoding and delay learning.

**Novelty rating:** High. Delay-learning SNN for audio classification on neuromorphic hardware is unexplored.

---

## IDEA 12: Binary / Ternary Weight SNN

**Source:** "Ternary Weight Networks" (arXiv:1605.04711); "xTern: Energy-Efficient Ternary Neural Network Inference on RISC-V" (2024); "1 Bit is All We Need" (arXiv:2509.07025, 2025)

**Core mechanism:** Constrain all weights to {-1, 0, +1}. Multiply-accumulate becomes add/subtract/skip. With binary weights {0,1}, MAC becomes a single AND gate. With ternary {-1,0,+1}, MAC becomes conditional add/subtract.

**Application to our system:**
- Our SNN already has binary spikes (0/1). If weights are also ternary, then:
  - Synaptic operation = add, subtract, or skip (no multiplication at all)
  - 16x memory reduction (ternary weights need 2 bits vs 32-bit float)
- Train with ternary quantization-aware training
- SpiNNaker already uses fixed-point weights; ternary is a natural extension

**Estimated energy reduction:** 5-16x. Elimination of all multiplications. 16x less weight memory.

**Accuracy impact:** Ternary typically loses 1-5% vs full precision. Our SNN accuracy (47.15%) might drop to 42-45%.

**Can it run on SpiNNaker?** Yes. SpiNNaker's synaptic weights can be stored as 2-bit values. The synaptic current computation simplifies to conditional add/subtract.

**Implementation difficulty:** Low-Medium. Ternary quantization-aware training is well-established. Need to verify snnTorch compatibility.

**Novelty rating:** Medium. Ternary SNNs exist for vision. First application to audio on SpiNNaker is new.

---

## IDEA 13: Spectrogram Subsampling -- Classify from Less

**Source:** Our own temporal ablation data; "How Low Can You Go? Reducing Frequency and Time Resolution" (arXiv:1911.04824)

**Core mechanism:** We use 64 mel bins x 216 time frames = 13,824 features. But do we need all of them? Our saliency maps show the SNN attends to specific regions. What if we:
- Use 16 mel bins instead of 64 (4x fewer)
- Use first 1 second instead of 5 seconds (5x fewer time frames, ~43 frames)
- Total: 16 x 43 = 688 features (20x reduction)

**Application to our system:**
- Retrain with 16 mel bins x 43 time frames
- Much smaller input = much smaller model = much less computation
- Feature selection: use our saliency maps to identify which mel bins matter most
- Per-class feature importance: maybe only 8 bins matter for "dog_bark" but 32 for "engine"

**Estimated energy reduction:** 5-20x (proportional to input size reduction, cascading through all layers)

**Accuracy impact:** Significant but quantifiable. 48 mel bins performs nearly as well as 128 in published work. First 1-2 seconds likely sufficient for many ESC-50 sounds. Expect 30-40% accuracy with aggressive subsampling.

**Can it run on SpiNNaker?** Yes. Smaller input = fewer input neurons = simpler SpiNNaker deployment.

**Implementation difficulty:** Low. Just change mel spectrogram parameters and retrain.

**Novelty rating:** Low-Medium. Well-studied but systematic ESC-50 ablation with SNN energy analysis would be useful.

---

## IDEA 14: Knowledge Distillation -- Large Teacher, Tiny Student

**Source:** "Dynamic Activation with Knowledge Distillation for Energy-Efficient SNN Ensembles" (arXiv:2502.14023, 2025); "Efficient Speech Command Recognition via SNN and Knowledge Distillation" (arXiv:2412.12858, 2024)

**Core mechanism:** Train a large, accurate ANN teacher. Distill its knowledge into a tiny SNN student (10-50x smaller). The student learns to mimic the teacher's soft output distribution, which contains more information than hard labels.

**Application to our system:**
- Teacher: PANNs CNN14 (81M params, 92.5% on ESC-50)
- Student: Tiny SNN (10-50K params)
- The student SNN learns the "dark knowledge" -- which classes are similar, which features matter
- Deploy only the tiny student on SpiNNaker

**Estimated energy reduction:** 10-60x (from model size reduction). KD can reduce timesteps by 60% and energy by 54.8%.

**Accuracy impact:** KD student typically retains 85-95% of teacher accuracy. From 92.5% teacher, expect 78-88% student. Or from 47.15% SNN teacher, a 5x smaller student might achieve 40-44%.

**Can it run on SpiNNaker?** Yes. The tiny student SNN is easier to deploy than our current model.

**Implementation difficulty:** Medium. KD framework exists for SNNs. Need to handle SNN-specific aspects (membrane potential matching, spike rate matching).

**Novelty rating:** Medium. KD for SNN exists but application to audio on neuromorphic hardware is new.

---

## IDEA 15: Hyperdimensional Computing (HDC)

**Source:** "A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing" (ISLPED 2016); "Hierarchical Hyperdimensional Computing for Energy Efficient Classification" (DAC 2018)

**Core mechanism:** Represent each feature and class as a very high-dimensional binary vector (e.g., 10,000 bits). Encode inputs by binding and bundling hypervectors. Classify by measuring cosine similarity to class prototype hypervectors. All operations are bitwise (XOR, majority vote).

**Application to our system:**
- Encode each mel spectrogram bin as a 10,000-bit hypervector
- Bind time and frequency positions using XOR
- Bundle across time using majority vote
- Compare to 50 class prototypes via Hamming distance
- Total: bitwise operations only, no floating point

**Estimated energy reduction:** 20-100x. 39.4 nJ/prediction demonstrated for multi-channel physiological data. Could potentially classify audio at <100 nJ.

**Accuracy impact:** HDC achieves 93-96% on simple tasks. For ESC-50 (harder), expect 25-40% with raw features, potentially higher with pre-extracted features.

**Can it run on SpiNNaker?** Partially. Hypervector operations are bitwise, which SpiNNaker's ARM cores can execute. But SpiNNaker isn't optimized for this paradigm. Better on FPGA.

**Implementation difficulty:** Medium. HDC libraries exist (torchhd). Need to design encoding for spectrograms.

**Novelty rating:** High. HDC for environmental sound classification is unexplored territory.

---

## IDEA 16: Event-Driven Processing -- Skip Silence and Redundancy

**Source:** "Exploiting neuro-inspired dynamic sparsity for energy-efficient intelligent perception" (Nature Communications 2025); Speech2Spikes (NICE 2023)

**Core mechanism:** Environmental sounds are not continuous. A 5-second clip of "dog bark" might have 0.5s of barking and 4.5s of silence/background. Process ONLY the event-rich portions.

**Application to our system:**
- Compute energy envelope of audio
- Only generate spikes for frames where energy exceeds threshold
- Skip all silence/low-energy frames (zero computation)
- For ESC-50: many sounds are sparse events (clock tick, door knock, coughing)
- Result: 80-95% of frames can be skipped for many classes

**Estimated energy reduction:** 2-10x depending on sound class. Sparse sounds (clock tick): 10x. Continuous sounds (rain): 1.5x. Average across ESC-50: ~4x.

**Accuracy impact:** Minimal. Classification-relevant information is in the events, not the silence.

**Can it run on SpiNNaker?** Perfectly. SpiNNaker is event-driven by design. No input spikes = no computation.

**Implementation difficulty:** Very Low. Simple energy-based gating on input spike generation.

**Novelty rating:** Low-Medium. Conceptually simple but systematic analysis for ESC-50 would be useful.

---

## IDEA 17: Stochastic Computing -- Multiply with AND Gates

**Source:** "Stochastic Computing CNN Architecture Reinvented for FPGA" (MDPI 2024); "Hardware-Efficient Stochastic Computing-Based Neural Networks with SNN-Isomorphic LIF Activation" (2025)

**Core mechanism:** Represent numbers as random bitstreams where the probability of a 1 encodes the value (e.g., 0.7 = bitstream with 70% ones). Multiplication becomes a single AND gate. Addition becomes a multiplexer. Entire neural network layers become simple combinational logic.

**Application to our system:**
- Already have binary spikes -- natural fit for stochastic representation
- Convert SNN weights to stochastic bitstreams
- Synaptic operation: AND(spike, weight_bit) instead of multiply
- 4.5x more energy efficient and 4x more area efficient than conventional FPGA designs

**Estimated energy reduction:** 3-5x per operation. Combined with spike sparsity: 5-15x overall.

**Accuracy impact:** Stochastic noise adds ~1-3% accuracy loss. Longer bitstreams = higher precision but slower.

**Can it run on SpiNNaker?** Not natively. SpiNNaker uses deterministic computation. Better for FPGA or custom hardware.

**Implementation difficulty:** High. Requires custom hardware design.

**Novelty rating:** Medium. SC for SNN has been proposed but not widely implemented for audio.

---

## IDEA 18: L-Mul -- Replace Multiplications with Integer Additions

**Source:** "Addition is All You Need for Energy-efficient Language Models" (arXiv:2410.00907, 2024, ICLR 2025)

**Core mechanism:** The L-Mul algorithm approximates floating-point multiplication using only integer addition by cleverly manipulating the exponent and mantissa. One integer addition costs ~37x less energy than one float32 multiplication. Applied to ALL multiplications in a neural network.

**Application to our system:**
- Replace all MAC operations in our SNN inference with L-Mul
- 95% energy reduction in element-wise operations
- 80% energy reduction in dot products
- Applicable to our ANN baseline too (for the software energy comparison)

**Estimated energy reduction:** 3-10x for the compute portion of energy. Note: our SNN energy is dominated by synaptic operations (accumulate-and-fire), which are already additions. Bigger impact on the ANN comparison.

**Accuracy impact:** L-Mul with 4-bit mantissa matches float8 precision. For audio classification, effectively lossless.

**Can it run on SpiNNaker?** Partially applicable. SpiNNaker's synaptic ops are already integer-based. But the principle reinforces our existing SNN efficiency story.

**Implementation difficulty:** Low for software. L-Mul can be applied as a drop-in replacement.

**Novelty rating:** Medium. Recent paper, hot topic, but application to audio SNN is novel.

---

## IDEA 19: Neuromorphic Cochlea Frontend -- Bypass Mel Spectrograms

**Source:** "Neuromorphic acoustic sensing using adaptive MEMS cochlea" (Nature Electronics 2023); "FPGA-based Neuromorphic Cochlea" (arXiv:2405.15923, 2024); Speech2Spikes (NICE 2023)

**Core mechanism:** Replace the entire mel spectrogram pipeline (FFT, mel filterbank, log compression) with a neuromorphic cochlea that directly produces spike trains from raw audio. The cochlea operates in analog domain, consuming microwatts. It produces sparse, event-driven output that feeds directly into the SNN.

**Application to our system:**
- Current pipeline: audio -> FFT -> mel -> encode -> SNN (digital throughout)
- Proposed: audio -> silicon cochlea (analog) -> spike trains -> SNN
- Silicon cochlea: ~120 frequency channels, event-driven, microwatt power
- Eliminates ALL digital preprocessing energy

**Estimated energy reduction:** Preprocessing savings: 100-1000x for the frontend. Total system: 2-5x (preprocessing is not the dominant cost currently).

**Accuracy impact:** Bio-inspired preprocessing often IMPROVES robustness to noise. Speech2Spikes showed competitive results with event-driven encoding.

**Can it run on SpiNNaker?** Yes. Spike-based cochlea output feeds directly into SpiNNaker. This is the intended neuromorphic pipeline.

**Implementation difficulty:** High (requires hardware cochlea). Could simulate in software as proof-of-concept.

**Novelty rating:** High. Full neuromorphic audio pipeline (cochlea + SpiNNaker SNN) for ESC-50 would be a first.

---

## IDEA 20: Forward-Forward Algorithm -- No Backpropagation

**Source:** "Backpropagation-free Spiking Neural Networks with the Forward-Forward Algorithm" (arXiv:2502.20411, 2025; Scientific Reports 2026)

**Core mechanism:** Replace backpropagation with two forward passes (positive pass with correct label, negative pass with wrong label). Each layer learns locally. No backward pass = no storing intermediate activations = 60%+ memory reduction.

**Application to our system:**
- Train our SNN using FF instead of backpropagation
- Each layer learns independently (fully local learning rule)
- On-device learning becomes possible (no backward pass = neuromorphic-compatible training)
- Memory reduction: <40% of BP training cost

**Estimated energy reduction:** Training: 3-10x. Inference: similar to BP-trained model, but enables on-chip learning on SpiNNaker for adaptation.

**Accuracy impact:** FF-SNNs achieve ~92% on MNIST (vs 99% BP). For ESC-50, expect 5-15% accuracy drop vs BP training.

**Can it run on SpiNNaker?** Perfectly for inference. Also enables ON-CHIP training since no backward pass is needed. This is the key advantage for SpiNNaker deployment.

**Implementation difficulty:** Medium. snnTorch has a Forward-Forward tutorial. Would need adaptation for audio.

**Novelty rating:** High. FF-SNN for audio on SpiNNaker with on-chip learning capability is genuinely novel.

---

## IDEA 21: Dendritic Computation -- Fewer But Smarter Neurons

**Source:** "Dendritic Integration Inspired Artificial Neural Networks" (NeurIPS 2024); "Introducing the Dendrify framework" (Nature Communications 2022)

**Core mechanism:** Standard LIF neurons are "point neurons" -- they sum all inputs linearly. Real neurons have dendrites that perform nonlinear computation WITHIN the neuron (e.g., AND, OR operations on different dendritic branches). One dendritic neuron can compute what takes 10-100 point neurons.

**Application to our system:**
- Replace 256 hidden LIF neurons with 25-50 dendritic neurons
- Each dendritic neuron has 5-10 branches processing different feature subsets
- Same computational power, 5-10x fewer neurons
- Fewer neurons = fewer spikes = less routing on SpiNNaker

**Estimated energy reduction:** 3-10x (proportional to neuron reduction).

**Accuracy impact:** Dendritic models achieve comparable or better accuracy (91% vs 88% in digit classification). For our SNN, expect maintained or slightly improved accuracy.

**Can it run on SpiNNaker?** Partially. SpiNNaker supports multi-compartment neuron models (e.g., Izhikevich). Dendritic branches can be modeled as separate compartments. Requires more computation per neuron but far fewer neurons overall.

**Implementation difficulty:** High. Need custom neuron model for SpiNNaker. Limited sPyNNaker support for multi-compartment models.

**Novelty rating:** Very High. Dendritic SNN for audio classification on neuromorphic hardware is unexplored.

---

## IDEA 22: Inference Caching / Memoization

**Source:** CacheNet (arXiv:2007.01793); "Energy-efficient acceleration of CNNs using computation reuse" (2022)

**Core mechanism:** Many environmental sounds are repetitive (e.g., a factory environment has the same 5-10 sounds recurring). Cache recent classification results. If a new input is "similar enough" to a cached input (measured by simple feature distance), reuse the cached result. Zero inference energy for cache hits.

**Application to our system:**
- Maintain a small cache of K most recent (feature_hash, class_label) pairs
- For new input: compute simple hash, check cache
- Cache hit: return cached label (near-zero energy)
- Cache miss: run full SNN inference, update cache
- For streaming audio: high cache hit rate due to temporal continuity

**Estimated energy reduction:** Depends on repetition rate. In real environments: 5-50x (80-98% cache hit rate for stationary sound sources). For one-shot ESC-50 evaluation: 1x (no benefit).

**Accuracy impact:** Zero if similarity threshold is set correctly. The cache stores verified classifications.

**Can it run on SpiNNaker?** Yes. Cache in SpiNNaker SRAM. Simple hash comparison before triggering SNN computation.

**Implementation difficulty:** Low. Straightforward engineering. Hash function + small lookup table.

**Novelty rating:** Medium. Novel in the neuromorphic context. "SpiNNaker with inference caching for real-time audio" is a practical contribution.

---

## IDEA 23: Membrane Potential Quantization (Ultra-Low Bit)

**Source:** "SpQuant-SNN: Ultra-low precision membrane potential" (Frontiers 2024); "QUEST: Quantized Energy-Aware SNN Training" (arXiv:2504.00679, 2025)

**Core mechanism:** Our SNN uses 32-bit floating-point membrane potentials. Quantize to 2-4 bits. This reduces memory by 7-13x and eliminates floating-point arithmetic entirely. Combined with binary spikes and ternary weights: the entire SNN runs on 2-4 bit integers only.

**Application to our system:**
- Weights: ternary {-1,0,+1} = 2 bits
- Membrane potentials: 4-bit integer
- Spikes: 1-bit binary (already)
- Total model memory: ~10 KB (down from ~2.5 MB)
- All operations: integer add/subtract with 4-bit accumulator

**Estimated energy reduction:** 7-13x from memory reduction alone. Plus elimination of floating-point ops.

**Accuracy impact:** 4-bit membrane: ~2-5% accuracy drop. 2-bit membrane: ~10-15% drop. With quantization-aware training: losses can be mitigated.

**Can it run on SpiNNaker?** Yes. SpiNNaker already uses fixed-point internally. Lower precision = less computation per neuron update.

**Implementation difficulty:** Medium. Quantization-aware training with snnTorch. Need to verify SpiNNaker fixed-point compatibility.

**Novelty rating:** Medium. SpQuant-SNN is recent but the specific application to audio + SpiNNaker is new.

---

## IDEA 24: Self-Powered Neuromorphic Audio Sensor

**Source:** "Self-aware artificial auditory neuron with triboelectric sensor" (Nano Energy 2023)

**Core mechanism:** Use a triboelectric nanogenerator (TENG) as both the audio sensor AND the power source. Sound waves hitting the TENG generate both the sensing signal and the electricity to power the neuromorphic classifier. Net external energy: ZERO.

**Application to our system:**
- TENG converts sound waves to electrical spikes (natural spiking interface)
- Harvested energy powers a small SNN classifier
- For environmental monitoring: completely self-powered, maintenance-free
- Total external energy consumption: 0 Joules per inference

**Estimated energy reduction:** Infinite (self-powered). Net energy is zero or negative (energy positive).

**Accuracy impact:** TENG bandwidth is limited. May not capture full spectral detail. Accuracy will be lower than microphone-based system. Expect 15-25% for simplified classification.

**Can it run on SpiNNaker?** No (SpiNNaker draws too much power for energy harvesting). But a tiny custom neuromorphic chip could be self-powered. SpiNNaker 2 with DVFS could potentially work for very intermittent operation.

**Implementation difficulty:** Very High. Requires hardware TENG + custom neuromorphic circuit.

**Novelty rating:** Extreme. Self-powered audio classification with zero external energy.

---

## IDEA 25: Phi Framework -- Pattern-Based Hierarchical Sparsity

**Source:** "Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency SNNs" (ISCA 2025, Duke University)

**Core mechanism:** Binary spike activations contain repeating patterns across neurons and timesteps. Phi identifies these patterns via k-means clustering, pre-computes results for each pattern, and skips redundant computation. Two-level hierarchy: (1) vector-wise pattern matching, (2) intra-pattern zero skipping.

**Application to our system:**
- Analyze spike activation patterns in our trained SNN
- Cluster into representative patterns
- Pre-compute partial results for each pattern
- At inference: match input pattern -> lookup pre-computed result
- 3.45x speedup and 4.93x energy improvement demonstrated on SNN accelerators

**Estimated energy reduction:** 3-5x with the Phi optimization alone. Combined with existing sparsity (73.6%): potentially 5-8x.

**Accuracy impact:** Pattern-aware fine-tuning recovers accuracy lost from pattern quantization. <1% degradation reported.

**Can it run on SpiNNaker?** Partially. Pattern matching could be implemented in SpiNNaker's ARM cores. Full benefit requires dedicated hardware support (as in the Phi accelerator).

**Implementation difficulty:** Medium-High. Need to extract patterns from our model, implement matching logic.

**Novelty rating:** Medium-High. Very recent (June 2025). First application to audio SNN would be novel.

---

## SYNTHESIS: MOST PROMISING COMBINATIONS

The most powerful approach combines multiple ideas:

### "The Fly-Brain Audio Classifier" (Ideas 3 + 8 + 16)
- Silicon cochlea or event-driven encoding (skip silence)
- Random sparse projection (fixed, no training, like mushroom body Kenyon cells)
- Winner-take-all sparsification
- Tiny trained readout
- **Total energy: ~50-200 nJ. Accuracy: 20-35%. Novelty: extreme.**

### "The One-Shot SNN" (Ideas 1 + 2 + 12)
- Multi-level neurons at T=1-3
- Temporal early exit for easy samples
- Ternary weights
- **Total energy: ~50-100 nJ. Accuracy: 30-40%. Novelty: high.**

### "The Cascaded Lookup Classifier" (Ideas 4 + 6 + 7)
- Coarse super-category classification via simple hash
- Fine classification via class-specific DWN lookup tables
- **Total energy: ~30-80 nJ. Accuracy: 35-45%. Novelty: extreme.**

### "The Pruned Predictive SNN" (Ideas 9 + 10 + 23)
- 95-99% pruning (lottery ticket)
- Predictive coding (only process surprises)
- Membrane quantization to 4 bits
- **Total energy: ~50-150 nJ. Accuracy: 35-42%. Novelty: high.**

### "The Practical Deployable Improvement" (Ideas 2 + 7 + 13 + 16)
- Temporal early exit (easy sounds exit at T=5)
- Hierarchical coarse-to-fine cascade
- Spectrogram subsampling (first 2 seconds, 32 mel bins)
- Event-driven silence skipping
- **Total energy: ~100-300 nJ. Accuracy: 40-47%. Novelty: medium. Practicality: highest.**

---

## DATA TABLES

### Energy Comparison (Estimated)

| Approach | Energy (nJ) | Accuracy (%) | Hardware | Novelty |
|----------|-------------|-------------|----------|---------|
| Current SNN (T=25) | 968 | 47.15 | SpiNNaker | Baseline |
| Current ANN | 454 | 63.85 | GPU | Baseline |
| Multi-level T=1 SNN | 50-100 | 30-40 | SpiNNaker* | High |
| Temporal early exit | 200-400 | 45-47 | SpiNNaker | Medium |
| Fly olfactory circuit | 50-200 | 20-35 | SpiNNaker | Very High |
| Weightless NN (DWN) | 30-100 | 25-40 | FPGA | Extreme |
| Logic gate network | 1-10 | 30-50 | FPGA | Extreme |
| Hash-based classification | 20-50 | 30-45 | Any | High |
| Hierarchical cascade | 100-300 | 42-48 | SpiNNaker | Medium |
| Reservoir/LSM | 200-500 | 15-30 | SpiNNaker | High |
| 99% pruned SNN | 50-150 | 25-30 | SpiNNaker | Medium |
| Predictive coding SNN | 100-300 | 40-45 | SpiNNaker | High |
| TTFS + delay learning | 100-200 | 35-45 | SpiNNaker* | High |
| Ternary weight SNN | 200-400 | 42-45 | SpiNNaker | Medium |
| HDC (10,000-bit) | 50-100 | 25-40 | FPGA | High |
| Stochastic computing | 100-300 | 40-44 | FPGA | Medium |
| KD tiny student SNN | 100-300 | 40-44 | SpiNNaker | Medium |
| SpQuant (4-bit membrane) | 200-500 | 42-45 | SpiNNaker | Medium |
| Phi pattern sparsity | 200-400 | 45-47 | Custom | Medium-High |
| Combined best | 30-100 | 35-45 | Various | Extreme |

*with modifications

### Difficulty vs Impact Matrix

| | Low Impact (1-3x) | Medium Impact (3-10x) | High Impact (10-100x) | Extreme (100x+) |
|---|---|---|---|---|
| **Easy** | Silence skip, Subsampling | Temporal early exit, Ternary | 99% pruning | -- |
| **Medium** | Predictive coding | KD student, Cascade, LSM | Multi-level T=1, HDC | Hash-based, DWN |
| **Hard** | -- | Delay learning, Phi patterns | Logic gates | Neuromorphic cochlea |
| **Very Hard** | -- | Stochastic computing | Self-powered TENG | -- |

---

## RESEARCH GAPS AND RECOMMENDED FOLLOW-UPS

1. **No published work** applies DWNs (weightless NNs) to audio classification. This is a completely open research direction with enormous potential.

2. **No published work** implements the fly olfactory circuit for environmental sound classification. The 50-class / 50-odor parallel is striking.

3. **No published work** combines temporal early exit with SpiNNaker for audio. The SPARQ framework (March 2026) is brand new and directly applicable.

4. **No published work** distills an audio SNN into logic gate networks. The NeurIPS 2024 oral paper makes this achievable.

5. The **rate-distortion theoretical minimum** for classifying 50 environmental sounds has never been computed. This would provide the fundamental lower bound on energy.

---

## CONFIDENCE ASSESSMENT

- **High confidence:** Temporal early exit, hierarchical cascade, pruning, ternary weights -- these are well-established techniques with predictable outcomes.
- **Medium confidence:** Multi-level T=1, HDC, reservoir computing, hash-based -- demonstrated in other domains, transferability to audio SNN is plausible but unverified.
- **Low confidence (but high novelty):** DWN for audio, fly circuit, logic gate networks, self-powered -- speculative but potentially revolutionary. These are the ideas that could yield a breakthrough paper.

---

## REFERENCES AND SOURCES

### Single-Timestep and Multi-Level SNNs
- [All in One Timestep](https://arxiv.org/html/2510.24637) - Multi-level spiking neurons
- [Scale-and-Fire Neurons](https://arxiv.org/html/2510.23383) - T=1 ANN-to-SNN conversion
- [One Timestep Is All You Need](https://openreview.net/forum?id=swRxhFpK5ds) - Ultra-low latency SNN

### Temporal Early Exit
- [SEENN](https://ar5iv.labs.arxiv.org/html/2304.01230) - Temporal spiking early exit
- [SPARQ](https://arxiv.org/html/2603.14380) - RL-guided spiking early exit (March 2026)
- [Early-Exit Survey](https://dl.acm.org/doi/full/10.1145/3698767) - Comprehensive review

### Biological Circuits
- [Fly Olfactory Algorithm](https://www.quantamagazine.org/new-ai-strategy-mimics-how-brains-learn-to-smell-20180918/) - Quanta Magazine overview
- [Fly-CL](https://arxiv.org/html/2510.16877v1) - Fly-inspired continual learning (2024)
- [Sparse Code for Natural Sound](https://pmc.ncbi.nlm.nih.gov/articles/PMC10749876/) - Auditory cortex sparse coding (2024)

### Weightless and Logic Gate Networks
- [DWN at ICML 2024](https://proceedings.mlr.press/v235/bacellar24a.html) - 56 nJ, 5 ns inference
- [nanoML for HAR](https://arxiv.org/html/2502.12173v1) - 926,000x energy savings
- [Logic Gate Networks NeurIPS 2024](https://arxiv.org/html/2411.04732v1) - 24 ns inference, 86.29% CIFAR-10
- [LogicWiSARD](https://labs.engineering.asu.edu/advent/wp-content/uploads/sites/123/2023/09/LogicWisard_ASAP22_May20.pdf) - 80% energy reduction

### Pruning and Sparsity
- [LTH for SNN](https://arxiv.org/abs/2207.01382) - ECCV 2022, 97% sparsity achievable
- [QP-SNN](https://openreview.net/forum?id=MiPyle6Jef) - Quantized and Pruned SNN (ICLR 2025)
- [Phi Framework](https://arxiv.org/html/2505.10909) - Pattern-based hierarchical sparsity (ISCA 2025)
- [Dynamic Spatio-Temporal Pruning](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1545583/full)

### Energy-Efficient Computation
- [Addition is All You Need](https://arxiv.org/abs/2410.00907) - L-Mul algorithm (ICLR 2025)
- [Neuro-Inspired Dynamic Sparsity](https://www.nature.com/articles/s41467-025-65387-7) - Nature Communications 2025
- [Horowitz 2014](https://ieeexplore.ieee.org/document/6757323) - Energy per operation (ISSCC)

### Reservoir Computing
- [EARL Framework](https://arxiv.org/pdf/2601.05205) - Energy-aware LSM optimization (2026)
- [Deep ESN for ESC](https://link.springer.com/chapter/10.1007/978-981-96-0994-9_7) - Environmental sound with ESN

### Knowledge Distillation
- [SNN Ensemble KD](https://arxiv.org/html/2502.14023v1) - Energy-efficient SNN ensembles (2025)
- [Speech SNN KD](https://arxiv.org/html/2412.12858v1) - Curriculum learning KD for audio SNN

### Predictive Coding
- [Predictive Coding Light](https://www.nature.com/articles/s41467-025-64234-z) - Nature Communications 2025
- [Predictive Coding Survey for SNN](https://arxiv.org/html/2409.05386v1)

### Hyperdimensional Computing
- [HDC Processor](https://ieeexplore.ieee.org/abstract/document/9645008) - 39.4 nJ/prediction
- [Efficient HDC](https://openreview.net/forum?id=9RQh6MOOaD) - Survey

### Neuromorphic Audio
- [Neuromorphic Cochlea](https://www.nature.com/articles/s41928-023-00957-5) - Nature Electronics 2023
- [Speech2Spikes](https://dl.acm.org/doi/fullHtml/10.1145/3584954.3584995) - Efficient audio encoding
- [Neuromorphic KWS with PDM](https://arxiv.org/html/2408.05156v1) - MEMS microphone to SNN

### Quantization and Binary Networks
- [SpQuant-SNN](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full) - Ultra-low precision membrane
- [1 Bit Is All We Need](https://arxiv.org/html/2509.07025v1) - Binary normalized NNs
- [xTern](https://www.aimodels.fyi/papers/arxiv/xtern-energy-efficient-ternary-neural-network-inference) - Ternary on RISC-V

### Delay Learning
- [Delay Learning in SNN](https://www.sciencedirect.com/science/article/abs/pii/S0893608024006026) - Neural Networks 2024
- [Learnable Axonal Delay](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1275944/full) - Spoken word recognition

### Forward-Forward Algorithm
- [FF for SNN](https://arxiv.org/abs/2502.20411) - Backpropagation-free SNN (2025)
- [snnTorch FF Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_forward_forward.html)

### Hierarchical Classification
- [ECHO Framework](https://arxiv.org/pdf/2409.14043) - ESC with hierarchical ontology (2024)
- [Coarse-to-Fine Edge](https://www.sciencedirect.com/science/article/pii/S0167739X24000736) - Energy-efficient cascade

### Thermodynamic Limits
- [Thermodynamic Bounds on DNN Energy](https://arxiv.org/html/2503.09980v2) - Fundamental limits (2025)
- [Landauer Principle](https://en.wikipedia.org/wiki/Landauer%27s_principle) - kT ln 2 per bit

### Self-Powered Sensors
- [Triboelectric Auditory Neuron](https://www.sciencedirect.com/science/article/abs/pii/S2211285523001593) - Self-powered audio SNN

### SpiNNaker Energy
- [SpiNNaker 2 Architecture](https://arxiv.org/pdf/2103.08392) - 10 pJ/synaptic event
- [SpiNNaker 2 for ML](https://arxiv.org/html/2401.04491v1) - Large-scale neuromorphic system
