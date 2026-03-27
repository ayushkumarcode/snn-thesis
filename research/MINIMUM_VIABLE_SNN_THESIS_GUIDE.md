# Minimum viable SNN thesis -- the efficient path to a decent grade

So i worked out the fastest path to a completed, high-quality SNN thesis at Manchester. It's basically a **comparative benchmarking study** using **snnTorch**. Compare SNN performance against a conventional ANN baseline on 1-2 standard datasets (MNIST + one other), measuring accuracy, spike count, and estimated energy efficiency. This works because: (a) snnTorch tutorials give you like 80% of the code already, (b) the research question is clear and naturally produces tables/graphs for the report, (c) you can scale ambition up or down depending on time, and (d) it addresses a real open question (when/why do SNNs outperform ANNs).

**Estimated coding time: 30-50 hours** (not including report writing).
**Minimum datasets: 1 (MNIST), recommended: 2 (MNIST + Fashion-MNIST or N-MNIST).**
**Minimum experiments: 3-4 distinct configurations producing 2-3 results tables.**
**Lit review scope: 20-30 references (15-20 core papers + 5-10 supporting).**

---

## 1. The simplest project that could still get a good grade

### Recommended: "Comparative Analysis of Spiking vs. Conventional Neural Networks for Image Classification"

**Research question:** "How does SNN performance compare to equivalent ANN architectures on standard image classification benchmarks, and under what conditions do SNNs offer advantages?"

Why this is the optimal minimum viable thesis:

1. **It's a genuine research question.** The SNN vs ANN comparison is active research. A 2020 paper from UCSB ("Rethinking the performance comparison between SNNs and ANNs") noted that on ANN-oriented workloads SNNs fail to beat ANNs, while on SNN-oriented workloads SNNs can outperform -- this tension is worth investigating.

2. **The code is mostly pre-written.** snnTorch Tutorial 5 (FC SNN on MNIST) and Tutorial 6 (conv SNN on MNIST) give you the SNN side. A standard PyTorch CNN is the ANN baseline. You're adapting existing code, not building from scratch.

3. **It produces natural report structure.** Background (SNN theory) -> Methodology (architectures, datasets, metrics) -> Results (comparison tables) -> Discussion (why gaps exist, when SNNs win) -> Conclusion.

4. **It scales.** Minimum version: 1 dataset, 2 architectures. Ambitious version: 3 datasets, 4 architectures, hyperparameter sweeps, encoding comparison.

### What makes this a 2:1 vs first:

| Grade | What you need |
|-------|--------------|
| **2:1 (60-69%)** | ANN vs SNN on MNIST with accuracy comparison. Clear report with adequate lit review. Basic analysis of results. |
| **First (70%+)** | Multiple datasets (MNIST + Fashion-MNIST or N-MNIST). Multiple architectures (FC + CNN). Additional metrics beyond accuracy (spike count, training time, estimated energy). Thoughtful discussion of WHY results differ. Encoding scheme comparison (rate vs. latency coding). |

### Alternative minimum viable projects (ranked by efficiency):

| Project | Implementation Effort | Report Writing Effort | Risk Level | Grade Ceiling |
|---------|----------------------|----------------------|------------|---------------|
| **SNN vs ANN comparison (recommended)** | Low (adapt tutorials) | Medium | Low | First |
| **Spike encoding scheme comparison** | Low-Medium | Medium | Low | First |
| **ANN-to-SNN conversion study** | Low (use sinabs or SNN Toolbox) | Medium | Low | 2:1/First |
| **STDP unsupervised learning on MNIST** | Medium (BindsNET) | Medium | Medium | First |
| **Neuron model comparison (LIF vs Izhikevich)** | Medium | Higher (more theory) | Low | First |
| **DVS128 gesture recognition** | Higher | Medium | Higher | First |

---

## 2. How many experiments/results are needed

Based on successful undergrad SNN projects and Manchester's assessment criteria ("the report has to be a complete account... how you evaluated it and with what results"):

### Minimum for a 2:1: 3-4 experiments producing 2 results tables

**Experiment Set A (Baseline):**
1. Train ANN (standard CNN) on MNIST -- record accuracy, training time
2. Train SNN (equivalent architecture using snnTorch) on MNIST -- record accuracy, training time, spike count

**Experiment Set B (Variation):**
3. Vary a key SNN parameter (e.g., number of time steps: 10, 25, 50, 100) and record accuracy vs time steps
4. Test on a second dataset (Fashion-MNIST) to show generalisability

**This produces:**
- Table 1: ANN vs SNN accuracy comparison
- Table 2: Effect of time steps on SNN accuracy
- Figure 1: Training loss curves (ANN vs SNN)
- Figure 2: Accuracy vs time steps plot

### For a first: 6-8 experiments producing 3-4 results tables + 4-6 figures

Add to the above:
5. Compare fully-connected vs convolutional SNN architectures
6. Compare rate coding vs latency coding input encoding
7. Measure and compare spike counts / synaptic operations (energy proxy)
8. Test on N-MNIST (neuromorphic dataset) -- shows SNN advantage on temporal data

**Additional outputs:**
- Table 3: Architecture comparison (FC-SNN vs CSNN vs FC-ANN vs CNN)
- Table 4: Encoding scheme comparison
- Figure 3: Spike raster plots (visualise spiking activity)
- Figure 4: Confusion matrices
- Figure 5: Energy estimation comparison (synaptic operations)

### What examiners actually care about

The Manchester marking criteria say the report needs:
- "An elucidation of the problem and the objectives"
- "An in-depth investigation of the context and literature"
- "A critical appraisal... indicating the rationale for any design/implementation decisions"
- "Evaluation (with hindsight) of the project outcome"

The key insight here: examiners care MORE about your analysis of results than the quantity of results. Three well-analysed experiments with insightful discussion will outscore eight experiments with superficial commentary. For each result, you should explain: what you expected, what happened, and WHY.

---

## 3. How many datasets

### Absolute minimum: 1 (MNIST)

MNIST is universally accepted as the baseline benchmark for SNN research. Diehl & Cook (2015) used only MNIST and has 2000+ citations. Every SNN framework tutorial uses MNIST.

### Recommended: 2 datasets

| Dataset combination | Why this works | Difficulty |
|--------------------|---------------|------------|
| **MNIST + Fashion-MNIST** (recommended) | Same format (28x28 grayscale), harder task. Shows generalisability. Both load identically in PyTorch. | Easy -- literally change one line of code |
| **MNIST + N-MNIST** | Static vs neuromorphic. Shows SNN advantage on temporal data. Uses Tonic library with snnTorch (Tutorial 7). | Medium -- different data loading pipeline |
| **MNIST + CIFAR-10** | Tests on colour images. Good ANN comparison point. | Medium -- requires architecture changes |

### For a first: 2-3 datasets

Adding N-MNIST or DVS128 Gesture as a third dataset shows you understand neuromorphic data and puts SNNs in their natural domain. This is where the SNN advantage over ANNs actually appears.

### Datasets to avoid:
- Custom/obscure datasets (examiners can't verify your claims)
- Datasets requiring significant preprocessing work (time sink)
- Very large datasets (ImageNet, COCO) -- training time makes iteration impossible

---

## 4. Realistic time estimates

### Total coding time: 30-50 hours for 2:1, 50-80 hours for first

| Phase | Hours | What you're doing |
|-------|-------|--------------------|
| **Environment setup** | 2-3 | Install PyTorch, snnTorch, Tonic. Set up Jupyter/Colab. |
| **Work through snnTorch tutorials** | 6-10 | Tutorials 1, 3, 5, 6. This IS your learning AND your code base. |
| **ANN baseline implementation** | 3-5 | Standard PyTorch CNN on MNIST. Hundreds of examples exist online. |
| **Adapt SNN from tutorials** | 5-8 | Modify tutorial code for your specific architecture/parameters. |
| **Second dataset integration** | 2-4 | Fashion-MNIST: change one line. N-MNIST: follow Tutorial 7. |
| **Run experiments** | 4-8 | Training runs. Most take 5-30 mins each on GPU (use Google Colab). |
| **Generate plots and tables** | 3-5 | matplotlib/seaborn for figures. Pandas for tables. |
| **Debugging and iteration** | 5-10 | Things will break. Budget for this. |
| **TOTAL (2:1 path)** | **30-50** | |
| **Additional for first** | **+15-30** | Encoding comparison, N-MNIST, energy analysis, hyperparameter sweep |

### Time-saving strategies:

1. **Use Google Colab** for GPU access. Free tier is enough for MNIST-scale stuff. No local setup needed.

2. **Work through snnTorch tutorials in order.** Tutorials 1-6 progressively build understanding AND code. The tutorial code IS your project code -- you modify it, you don't write from scratch.

3. **Batch your experiments.** Set up all training runs in one session, let them run, then analyse results together.

4. **Use the same architecture for ANN and SNN.** Makes comparison fair AND reduces coding work. Same layers, same hidden dimensions -- just swap activations for LIF neurons.

5. **Automate plotting.** Write one plotting function that works for all experiments. Reuse it.

### Realistic calendar (for someone with other commitments):

| Week | Task | Hours |
|------|------|-------|
| Week 1 | Set up environment, work through Tutorials 1-3 | 8-10 |
| Week 2 | Tutorials 5-6, implement ANN baseline | 8-10 |
| Week 3 | Adapt SNN code, run initial experiments on MNIST | 8-10 |
| Week 4 | Add second dataset, run remaining experiments | 6-8 |
| Week 5 | Generate all plots/tables, start writing results chapter | 6-8 |
| Week 6 (buffer) | Fix issues, re-run experiments if needed | 4-6 |

**Total: 6 weeks at ~8 hours/week = ~48 hours of coding/experimentation.**

Report writing (separate from coding) typically takes another 40-60 hours for a good report. Budget 2-3 weeks for writing.

---

## 5. Existing code and tutorials to adapt (don't build from scratch)

### Primary resource: snnTorch (start here, don't look elsewhere until you've done these)

| Tutorial | What it gives you | Direct thesis use |
|----------|------------------|-------------------|
| [Tutorial 1 - Spike Encoding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html) | Rate coding, latency coding, delta modulation | Understand and implement encoding schemes |
| [Tutorial 3 - Feedforward SNN](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html) | LIF neuron model, building SNN layers | Your SNN architecture foundation |
| [Tutorial 5 - Training SNNs](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html) | FC-SNN on MNIST with backprop through time | **YOUR CORE EXPERIMENT CODE** |
| [Tutorial 6 - Convolutional SNN](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html) | CSNN on MNIST with surrogate gradients | Your convolutional SNN experiment |
| [Tutorial 7 - Neuromorphic Datasets](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html) | Loading N-MNIST with Tonic library | Your neuromorphic dataset experiment |
| [Quickstart](https://snntorch.readthedocs.io/en/latest/quickstart.html) | Complete end-to-end minimal example | Quick reference / sanity check |

All tutorials are available as interactive Google Colab notebooks. You can run them immediately, modify parameters, and see results without any local setup.

### GitHub repos to reference/adapt:

| Repository | What it provides | Stars | Link |
|-----------|-----------------|-------|------|
| **snnTorch** (official) | Complete framework + 18 tutorials | 1800+ | [jeshraghian/snntorch](https://github.com/jeshraghian/snntorch) |
| **Simple SNN STDP** | From-scratch STDP on MNIST (for understanding) | -- | [cowolff/Simple-Spiking-Neural-Network-STDP](https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP) |
| **SNN-MNIST Pure Python** | Unsupervised MNIST classification with STDP | -- | [sujay-pandit/spiking-neural-networks-mnist-classification](https://github.com/sujay-pandit/spiking-neural-networks-mnist-classification) |
| **stdp-mnist** (Diehl & Cook) | Reference implementation of seminal 2015 paper | -- | [peter-u-diehl/stdp-mnist](https://github.com/peter-u-diehl/stdp-mnist) |
| **Norse MNIST** | Alternative framework comparison point | -- | [norse/norse](https://github.com/norse/norse) |
| **Shape Detector SNN** | Manchester BSc example (Ferrari, supervised by Furber) | -- | [filippoferrari/shape_detector_snn](https://github.com/filippoferrari/shape_detector_snn) |

### ANN baseline code (takes 20 minutes to set up):

A standard PyTorch CNN for MNIST is in every deep learning tutorial online. The official PyTorch examples include one. Key point: use the SAME architecture dimensions as your SNN for a fair comparison.

### Tools you need:

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.8+ | Language | Already installed |
| PyTorch | Deep learning framework | `pip install torch torchvision` |
| snnTorch | SNN framework | `pip install snntorch` |
| Tonic | Neuromorphic data loading | `pip install tonic` (only if using N-MNIST) |
| matplotlib | Plotting | `pip install matplotlib` |
| Google Colab | Free GPU | browser-based, no install |

---

## 6. Literature review scope

### For a 2:1: 20-25 references

### For a first: 25-35 references

Manchester's report guidance says you need "an in-depth investigation of the context and literature." The general consensus for UK dissertations is roughly 30-40 references for a 10,000-word report. For a CS project where implementation is a large component, 20-30 well-chosen references is fine.

### Core papers you MUST cite (the essential 12):

**Foundational SNN papers:**
1. Maass, W. (1997). "Networks of spiking neurons: The third generation of neural network models." -- Defines SNNs as the "third generation"
2. Gerstner, W. & Kistler, W. (2002). "Spiking Neuron Models" (textbook) -- LIF model reference
3. Hodgkin, A.L. & Huxley, A.F. (1952). "A quantitative description of membrane current..." -- The biological foundation

**SNN training methods:**
4. Bohte, S.M. et al. (2002). "Error-backpropagation in temporally encoded networks of spiking neurons" -- Early SNN training
5. Neftci, E.O. et al. (2019). "Surrogate gradient learning in spiking neural networks" -- THE key paper on how modern SNNs are trained
6. Eshraghian, J.K. et al. (2023). "Training Spiking Neural Networks Using Lessons From Deep Learning" -- The snnTorch authors' tutorial/survey

**STDP and biological learning:**
7. Bi, G. & Poo, M. (1998). "Synaptic modifications in cultured hippocampal neurons" -- STDP discovery
8. Diehl, P.U. & Cook, M. (2015). "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" -- Seminal STDP-MNIST paper (95% accuracy with 6400 neurons)

**SNN vs ANN comparison:**
9. Deng, L. et al. (2020). "Rethinking the performance comparison between SNNs and ANNs" -- Directly relevant
10. Lemaire, E. et al. (2022). "Are SNNs really more energy-efficient than ANNs?" -- Energy efficiency analysis

**Encoding schemes:**
11. Kim, J. et al. (2021). "Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems" -- Rate vs temporal coding comparison

**Framework:**
12. Eshraghian, J.K. et al. (2023). "snnTorch: A Python package for training spiking neural networks" -- Cite your tool

### Additional references by topic (pick 10-15 based on your focus):

**Neuromorphic computing context:**
- Furber, S.B. et al. (2014). "The SpiNNaker Project" -- Manchester connection
- Davies, M. et al. (2018). "Loihi: A Neuromorphic Manycore Processor" -- Intel's chip
- Merolla, P.A. et al. (2014). "A million spiking-neuron integrated circuit" -- IBM TrueNorth

**Deep SNNs:**
- Wu, Y. et al. (2018). "Spatio-temporal backpropagation for training high-performance spiking neural networks"
- Fang, W. et al. (2021). "Incorporating learnable membrane time constants to enhance learning of spiking neural networks"
- Zheng, H. et al. (2021). "Going deeper with directly-trained larger spiking neural networks"

**ANN-to-SNN conversion:**
- Diehl, P.U. et al. (2015). "Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing"
- Bu, T. et al. (2022). "Optimal ANN-SNN conversion for high-accuracy and ultra-low-latency spiking neural networks"

**Neuromorphic datasets:**
- Orchard, G. et al. (2015). "Converting Static Image Datasets to Spiking Neuromorphic Datasets" -- N-MNIST
- Amir, A. et al. (2017). "A Low Power, Fully Event-Based Gesture Recognition System" -- DVS128 Gesture

**Energy efficiency:**
- Horowitz, M. (2014). "Computing's energy problem (and what we can do about it)" -- MAC vs AC operations
- Yousefzadeh, A. et al. (2022). "An analytical estimation of spiking neural networks energy efficiency"

### How to structure the lit review:

1. **Introduction to neural networks** (1-2 paragraphs, 2-3 refs)
2. **Biological neurons and spiking** (1-2 paragraphs, 2-3 refs)
3. **SNN neuron models** (LIF, Izhikevich) (1-2 paragraphs, 3-4 refs)
4. **SNN training methods** (STDP, surrogate gradients, ANN-to-SNN conversion) (2-3 paragraphs, 4-5 refs)
5. **Encoding schemes** (rate coding, temporal coding) (1 paragraph, 2-3 refs)
6. **SNN vs ANN performance comparisons** (1-2 paragraphs, 3-4 refs)
7. **Tools and frameworks** (1 paragraph, 2-3 refs)
8. **Summary and research gap** (1 paragraph, identifying what your project addresses)

Total: roughly 2000-3000 words for the lit review chapter.

---

## 7. Examples of simple but successful SNN student projects

### Undergrad projects that did well:

**1. Shape Detector SNN -- Filippo Ferrari (Manchester BSc, supervised by Steve Furber)**
- Single-layer SNN for shape detection using pyDVS
- 107 commits, clean code with testing and CI
- GitHub: [filippoferrari/shape_detector_snn](https://github.com/filippoferrari/shape_detector_snn)
- Thesis LaTeX: [filippoferrari/bsc_dissertation](https://github.com/filippoferrari/bsc_dissertation)
- Takeaway: Well-structured, proper software engineering, but moderate in scope

**2. Musical Pattern Recognition in SNNs -- mrahtz (BEng, ~2016)**
- First layer of an SNN for recognising musical patterns in audio
- Used Brian 2 simulator, STDP learning
- Author candidly stated "only a small portion of what was originally intended was actually achieved"
- 49 GitHub stars, showing community interest
- Thesis PDF: [amid.fish/beng_project_report.pdf](http://amid.fish/beng_project_report.pdf)
- Takeaway: Novel application domain. Honest about limitations. This honesty is EXPECTED and REWARDED at undergrad level.

**3. SNN vs CNN Comparison (course project, sofi12321)**
- Compared SNN and CNN on SOCOFing, EMNIST, Fashion-MNIST using snnTorch
- SNN achieved 98% vs CNN 83% on fingerprint dataset
- GitHub: sofi12321/comparison-of-spiking-and-convolutional-NNs
- Takeaway: Simple comparison framework, multiple datasets, clear results tables. Very doable.

**4. Simple SNN with STDP (4-student team, Osnabruck)**
- From-scratch Python implementation, STDP on MNIST
- Accuracy plateaued quickly, team was honest about limitations
- GitHub: cowolff/Simple-Spiking-Neural-Network-STDP
- Takeaway: Even simple implementations with honest analysis of failure modes are acceptable.

**5. SNN Image Classification (HaoyiZhu, course project)**
- Used snnTorch for static and spike-encoded data
- Achieved 99.12% on static MNIST, 97.05% on spike-encoded MNIST
- Takeaway: Near-99% accuracy is achievable with snnTorch tutorials.

### What these projects have in common:
- **Clear, focused scope** (one main thing done well, not five things done poorly)
- **Use of existing frameworks** (nobody built from scratch unless that was the deliberate point)
- **Honest reporting of limitations** (accuracy gaps, convergence issues, scope reductions)
- **Clean presentation** (tables, figures, clear writing)
- **Comparison element** (SNN vs ANN, or different SNN configurations)

---

## 8. Evaluation metrics

### Absolute minimum (for a 2:1): 3 metrics

| Metric | What it measures | How to get it |
|--------|-----------------|---------------|
| **Classification accuracy** | How well the model classifies | `correct_predictions / total_predictions` on test set |
| **Training time** | Computational cost | `time.time()` before and after training loop |
| **Training/test loss curves** | Learning dynamics | Log loss at each epoch, plot with matplotlib |

### For a first: 5-7 metrics

| Metric | What it measures | How to get it | Why examiners like it |
|--------|-----------------|---------------|----------------------|
| **Classification accuracy** | Correctness | Standard | Essential |
| **Training time per epoch** | Computational cost | `time.time()` | Shows awareness of efficiency |
| **Total spike count** | SNN activity level | `spk_rec.sum()` in snnTorch | SNN-specific, shows understanding |
| **Synaptic operations (SOPs)** | Energy proxy | spike_count x connections per layer | Energy estimation without hardware |
| **Confusion matrix** | Per-class performance | `sklearn.metrics.confusion_matrix` | Shows which classes are hard for SNNs |
| **Number of parameters** | Model complexity | `sum(p.numel() for p in model.parameters())` | Fair comparison metric |
| **F1-score** | Balanced accuracy | `sklearn.metrics.f1_score` | Better than accuracy for imbalanced data |

### Energy efficiency estimation (without neuromorphic hardware):

You don't need actual neuromorphic hardware to discuss energy efficiency. The standard approach in the literature is to use **synaptic operations (SOPs)** as a proxy:

- **ANN energy per inference** is proportional to: number_of_MAC_operations x energy_per_MAC
- **SNN energy per inference** is proportional to: number_of_AC_operations x energy_per_AC (only when a spike occurs)

The key insight is that AC operations (additions, triggered by spikes) use roughly 5-10x less energy than MAC operations (multiply-accumulate, happening every forward pass). Combined with spike sparsity (most SNN neurons are silent most of the time), this is the theoretical basis for SNN energy efficiency.

A simple calculation you can include:
```
Energy_ANN = num_parameters x num_layers x E_MAC
Energy_SNN = total_spike_count x avg_connections x E_AC
Ratio = Energy_ANN / Energy_SNN
```

