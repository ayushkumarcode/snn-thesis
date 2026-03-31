# COMP30040 Thesis Screencast Plan

## Project: Spiking Neural Networks for Environmental Sound Classification
## Duration: 7-9 minutes (target 8:00)
## Audience: Third-year CS student who has NOT studied SNNs

---

## FORMAT OPTIONS

### Option A: NotebookLM Podcast / Interview Style

**Concept:** Use NotebookLM or similar to generate two AI "hosts" who discuss the thesis as if on a tech podcast. The student interjects with narration, demos, and corrections. A back-and-forth between the AI hosts asking "wait, so why do SNNs exist?" creates a natural pedagogical structure.

**Pros:**
- Entertaining and novel -- examiners will not have seen this format before
- The "hosts asking questions" format mirrors the audience's perspective (someone who doesn't know SNNs)
- Natural pacing: curiosity-driven, not lecture-driven
- Handles exposition well (explaining LIF, encoding, SpiNNaker) without feeling like a lecture
- Easy to produce: record podcast, then screen-record yourself showing results/demos on top

**Cons:**
- Risk of feeling gimmicky if the AI voices dominate too much
- Hard to control pacing precisely (AI podcast segments have fixed length, hard to trim)
- Audio quality mismatch between AI voices and student narration
- Less personal -- the student's own voice and expertise gets diluted
- NotebookLM output quality is unpredictable; may need many iterations

---

### Option B: Documentary Style -- "The Sound of Spikes"

**Concept:** Narrate over a carefully edited sequence of live terminal demos, SpiNNaker board footage (photo/video), code walkthroughs, animated figures, and result reveals. Think BBC documentary: calm, authoritative narration, clean transitions, ambient music. Open with an actual sound clip (dog bark, thunderstorm), then ask "how would a brain classify this?" Builds through the science, the engineering, the failures, and the results.

**Pros:**
- The most professional-looking option if executed well
- Live demos (training running, spikes visualised, SpiNNaker deploying) are inherently impressive
- Allows showing the ACTUAL codebase, terminal, results files -- demonstrates real engineering work
- Full creative control over pacing, visually rich
- Examiner sees the student as a confident, knowledgeable presenter
- Can incorporate real ESC-50 audio clips to make it visceral

**Cons:**
- Highest production effort: screen recording, audio editing, transitions, timing
- Risk of being boring if the narration is flat or the visuals are just code scrolling
- No interaction / dialogue -- relies entirely on the quality of one voice
- Requires decent microphone and editing software (OBS + DaVinci Resolve / iMovie)

---

### Option C: TED-Talk Style -- Minimal Slides, Storytelling, Suspense

**Concept:** Face-to-camera (or voice-over-slides), structured as a talk with a clear narrative arc. Minimal text on slides, large numbers, one idea per slide. Build suspense: "What if I told you a network that communicates in binary spikes could match a conventional neural network?" Reveal PANNs result as the climax.

**Pros:**
- Clean, easy to produce -- just slides and voice
- Storytelling structure is inherently engaging
- Easy to hit timing marks precisely
- Emphasises the student's communication skills

**Cons:**
- Less "creative" -- every thesis screencast is slides + voice
- No live demos, so the examiner doesn't see the actual system working
- Risk of being indistinguishable from other screencasts
- Doesn't showcase the engineering (50 SpiNNaker deployments, 34 experiments) as viscerally

---

## RECOMMENDATION: Option B -- Documentary Style ("The Sound of Spikes")

**Rationale:** The rubric heavily rewards creativity and professionalism. Option B is the only format that can credibly approach "close to professionally produced" quality. It lets you show actual ESC-50 audio (visceral, memorable), live terminal output (demonstrates real work), architecture diagrams (explains the science), and result reveals with animation. The documentary framing gives you license to tell a story -- which is what separates an outstanding screencast from a competent one.

**Risk mitigation:** Pre-record everything. Write the narration script first, record audio, then screen-record visuals to match the audio. This guarantees timing. Use iMovie or DaVinci Resolve (free) for assembly. Record in a quiet room. Use a USB mic if available.

---

## PRODUCTION TOOLKIT

| Tool | Purpose |
|------|---------|
| OBS Studio | Screen recording (terminal, code, figures) |
| iMovie or DaVinci Resolve | Video editing, transitions, titles |
| Keynote / Google Slides | Animated slides for diagrams and numbers |
| QuickTime | Face-cam recording (optional, for intro/outro) |
| Audacity | Audio cleanup, noise reduction |
| Terminal (iTerm2) | Live demo of training, results, code |
| ESC-50 audio clips | Play actual sounds for the audience |
| USB microphone or AirPods Pro | Clean narration audio |

---

## DETAILED SCRIPT: "The Sound of Spikes"

### STRUCTURE OVERVIEW

| Segment | Time | Duration | Content |
|---------|------|----------|---------|
| 1. Hook | 0:00-0:40 | 40s | Sound clip + question |
| 2. The Problem | 0:40-1:40 | 60s | Why SNNs? Energy + biology |
| 3. What is an SNN? | 1:40-2:40 | 60s | LIF neuron, spikes, encoding |
| 4. The Architecture | 2:40-3:30 | 50s | Model diagram + code |
| 5. Encoding Showdown | 3:30-4:30 | 60s | 7 encodings, bar chart reveal |
| 6. Closing the Gap | 4:30-5:30 | 60s | Rhythm-SNN + PANNs (KEY finding) |
| 7. Real Hardware | 5:30-6:40 | 70s | SpiNNaker deployment + 50 pruned models |
| 8. What SNNs Do Better | 6:40-7:30 | 50s | Adversarial, forgetting, robustness |
| 9. Conclusion | 7:30-8:00 | 30s | Impact + closing |
| **Total** | | **8:00** | |

---

### SEGMENT 1: THE HOOK (0:00 - 0:40)

**[SCREEN: Black screen. Fade in waveform visualisation.]**

**[AUDIO: Play a real ESC-50 clip -- a dog barking. 2 seconds.]**

> "That was a dog."

**[AUDIO: Play a chainsaw clip. 2 seconds.]**

> "That was a chainsaw."

**[AUDIO: Play crackling fire. 2 seconds.]**

> "You just classified three environmental sounds in under six seconds. Your brain did it using about 20 watts of power -- roughly the same as a dim light bulb."

**[SCREEN: Fade to a GPU image, then show: "GPU inference: ~200 millijoules per clip"]**

> "A deep learning model running on a GPU can do the same thing -- but it burns through a thousand times more energy per classification. For a hearing aid, a wildlife sensor, or any battery-powered device, that is a non-starter."

**[SCREEN: Fade to title card -- "The Sound of Spikes: Spiking Neural Networks for Environmental Sound Classification" with name and COMP30040.]**

> "This thesis asks: can we build a neural network that classifies sounds more like a brain? Using binary spikes instead of continuous numbers, event-driven processing instead of dense matrix multiplication? And can we run it on real neuromorphic hardware?"

**[TRANSITION: Quick fade to black, 0.5s]**

---

### SEGMENT 2: THE PROBLEM (0:40 - 1:40)

**[SCREEN: Side-by-side comparison. Left: biological neuron diagram (simple, labelled). Right: artificial neuron diagram. Animate the differences.]**

> "In a conventional neural network, every neuron transmits a continuous floating-point number to every connected neuron. Each connection requires a multiply-accumulate operation -- multiply the activation by the weight, add it to the sum. At 45 nanometres, that costs about 4.6 picojoules."

**[SCREEN: Highlight the ANN side, show "4.6 pJ per MAC" label]**

> "In a spiking neural network, neurons communicate with binary spikes -- on or off, one or zero. No multiplication needed. You just add the weight when a spike arrives. That accumulate-only operation costs about 0.9 picojoules -- roughly five times cheaper."

**[SCREEN: Highlight the SNN side, show "0.9 pJ per AC" label. Show 5x arrow.]**

> "And here is the real advantage: most spiking neurons are silent most of the time. In our network, 74% of neurons produce zero spikes at any given moment. Zero spikes means zero computation. Energy scales with activity, not network size."

**[SCREEN: Show sparsity visualisation -- a grid of neurons, most dark (silent), a few flashing (spiking). Show "74% sparsity" label.]**

> "But there is a catch. Spike emission is a binary threshold -- not differentiable. You cannot run standard backpropagation through it. For years, this made SNNs nearly impossible to train on real tasks. The breakthrough came in 2019 with surrogate gradient descent: use exact binary spikes going forward, but approximate the gradient with a smooth function going backward. This is what makes this entire thesis possible."

**[SCREEN: Show the surrogate gradient concept -- step function with smooth surrogate overlaid. Simple animation.]**

**[TRANSITION: Wipe left]**

---

### SEGMENT 3: WHAT IS AN SNN? (1:40 - 2:40)

**[SCREEN: Animated LIF neuron. Show membrane potential rising with each input spike, leaking between spikes, crossing threshold, emitting output spike, resetting.]**

> "The core building block is the Leaky Integrate-and-Fire neuron. It maintains a membrane potential -- think of it as a bucket of water. Input spikes add water. Between spikes, the bucket slowly leaks. When the water level crosses a threshold, the neuron fires its own spike and the bucket empties."

**[SCREEN: Show the LIF equation appearing: V[t] = beta * V[t-1] + W * S_in[t]. Highlight beta as the "leak rate".]**

> "Mathematically, it is a recurrence: the membrane potential at time t equals the previous potential times a decay factor beta, plus the weighted sum of incoming spikes. Beta of 0.95 means the neuron retains 95% of its charge each timestep -- it has memory. This temporal dynamics is what makes SNNs fundamentally different from feedforward ANNs."

**[SCREEN: Show a mel spectrogram of a dog bark. Overlay: "But how do you turn this image into spikes?"]**

> "Our input is a mel spectrogram -- a 2D image of sound. 64 frequency bins, 216 time frames. To feed this into a spiking network, we need to convert continuous pixel values into binary spike trains. This is called neural encoding -- and it turns out, the choice of encoding changes everything."

**[SCREEN: Quick montage of 3 encoding types. Direct: feed values straight in. Rate: higher value = more spikes. Latency: higher value = earlier spike. Show each as a small animation.]**

> "Direct encoding feeds the raw spectrogram values as input currents. Rate encoding converts each value to a probability of spiking -- louder means more spikes. Latency encoding fires a single spike earlier for stronger inputs. We tested seven different encodings. The results were dramatic."

**[TRANSITION: Zoom into the bar chart]**

---

### SEGMENT 4: THE ARCHITECTURE (2:40 - 3:30)

**[SCREEN: Show the architecture diagram (paper/figures/architecture_diagram.png). Animate layer-by-layer left to right.]**

> "Our network is a spiking convolutional neural network with about 622,000 parameters. Input: a mel spectrogram. Two convolutional layers with batch normalisation and max pooling extract spatial features -- just like a conventional CNN. But after each conv layer, a LIF spiking neuron converts the output to binary spikes."

**[SCREEN: Highlight the conv layers, then the LIF layers. Show "spikes" as binary 0/1 flowing between layers.]**

> "After the convolutional feature extraction, an average pooling layer compresses the spatial dimensions, then two fully-connected layers with LIF neurons produce the final classification. The network processes each input for 25 timesteps -- 25 snapshots of spiking activity -- and the output neuron that fires the most spikes determines the predicted class."

**[SCREEN: Zoom into the code -- show a ~10-line snippet of the forward pass in src/models/snn.py. Highlight the snnTorch LIF call.]**

> "The entire network is implemented in 200 lines of Python using snnTorch, an open-source library built on PyTorch. Training uses surrogate gradients on A100 GPUs. Each fold of five-fold cross-validation takes about 15 minutes."

**[SCREEN: Show a brief terminal clip of training running -- loss decreasing, accuracy improving. Speed it up 4x.]**

**[TRANSITION: Slide left]**

---

### SEGMENT 5: THE ENCODING SHOWDOWN (3:30 - 4:30)

**[SCREEN: Black screen. Text appears: "7 encodings. 35 models. 1 question: does it matter how you convert sound to spikes?"]**

> "We tested seven different ways to encode audio into spikes. This is, to our knowledge, the first systematic encoding comparison for audio SNNs on a 50-class benchmark -- and the first SNN evaluation on the full ESC-50 dataset at all."

**[SCREEN: Reveal the bar chart (paper/figures/encoding_bar_chart.png) progressively -- bars appear one at a time from left to right, with a dramatic pause before the ANN baseline.]**

> "Direct encoding -- feeding raw spectrogram values as continuous input currents -- dominates at 47.15%. Rate and phase coding tie at 24%. Population coding at 19%. Latency at 16%. And at the bottom, delta coding at 7% and burst coding at 6.5% -- barely above random chance for 50 classes, which is 2%."

**[SCREEN: Show the ANN bar appearing. "63.85%". Draw the gap: "16.7 percentage points".]**

> "Our ANN baseline, with the exact same architecture but ReLU instead of LIF, hits 63.85%. So there is a 16.7 percentage point gap between the best SNN and the ANN. The obvious question is: is this gap fundamental, or can we close it?"

**[SCREEN: Text overlay -- "Is the gap fundamental?" with a question mark. Hold 2 seconds.]**

> "We found two ways to close it. One is architectural. The other reveals perhaps the most important finding of the entire thesis."

**[TRANSITION: Quick cut to black, 0.3s]**

---

### SEGMENT 6: CLOSING THE GAP (4:30 - 5:30)

**[SCREEN: Show a simplified Rhythm-SNN diagram -- oscillatory modulation signal modulating LIF membrane potential. Simple sine wave overlay on the neuron.]**

> "First, architecture. We developed 34 experimental configurations, testing bio-inspired techniques like oscillatory modulation, dendritic compartments, learnable delays, and knowledge distillation. The winner: Rhythm-SNN -- inspired by theta and gamma oscillations in the auditory cortex. It modulates the membrane leak rate with a learnable oscillatory signal, giving the network an internal sense of rhythm."

**[SCREEN: Show result reveal -- "Rhythm-SNN: 61.10%" appearing large, then "ANN: 63.85%". Gap shrinks from 16.7pp to 2.75pp. Animate the gap closing.]**

> "Rhythm-SNN reaches 61.10% -- a 14 percentage point improvement over the baseline SNN, closing the gap to just 2.75 points below the ANN. The network learned to pulse its attention in time, just like biological auditory circuits."

**[SCREEN: Transition to PANNs diagram. Show CNN14 (large, greyed out "pretrained") -> 2048-dim embedding -> small SNN head (3 layers, highlighted in blue).]**

> "But the most revealing experiment used transfer learning. We took CNN14 -- a large pretrained audio network trained on two million clips from Google's AudioSet -- froze its weights, and replaced the classifier head with a tiny spiking neural network. Three layers, LIF neurons, surrogate gradients."

**[SCREEN: Show results appearing side by side:]**
**PANNs + SNN: 92.50%**
**PANNs + ANN: 93.45%**
**Gap: 0.95 pp**

> "PANNs plus SNN head: 92.5%. PANNs plus ANN head: 93.45%. The gap collapsed from 16.7 percentage points to less than one."

**[SCREEN: Show text appearing dramatically: "When both get equal-quality features, SNN matches ANN."]**

> "This is the key scientific finding. The 17-point gap in scratch training is not a spiking computation problem -- it is a feature learning problem. When both networks receive high-quality features, the spiking classifier performs essentially identically to the conventional one. The limitation of SNNs is not in classification. It is in representation learning from limited data."

**[SCREEN: Hold on this statement for 2 seconds. Let it land.]**

**[TRANSITION: Slide to SpiNNaker board image]**

---

### SEGMENT 7: REAL HARDWARE (5:30 - 6:40)

**[SCREEN: Photo of SpiNNaker board (from University of Manchester). Show the SpiNNaker pipeline diagram (paper/figures/spinnaker_pipeline.png).]**

> "Results in simulation are one thing. Running on actual neuromorphic hardware is another. We deployed our network to SpiNNaker -- the million-core neuromorphic processor built right here at the University of Manchester."

**[SCREEN: Show the hybrid pipeline diagram animated: audio -> mel spectrogram -> CNN layers (software) -> binary spikes -> FC2 layer (SpiNNaker hardware, highlighted red) -> class prediction.]**

> "The deployment uses a hybrid pipeline. The convolutional layers run in software, extracting features and producing binary spike trains. The final classification layer runs on SpiNNaker hardware, using real spiking neurons communicating via the chip's multicast router."

**[SCREEN: Show terminal output of a SpiNNaker deployment running. Show spikes being sent, results coming back. Brief, 3-4 seconds.]**

> "Getting this to work was the hardest engineering challenge of the project. SpiNNaker's neuron model defaults to a resting potential of negative 65 millivolts -- but our snnTorch model uses zero. That one mismatch meant a 66-millivolt threshold gap, and zero neurons firing. Finding and fixing bugs like this took weeks of systematic debugging."

**[SCREEN: Show the key result: "SpiNNaker: 57.35% | snnTorch: 59.50% | Gap: 2.15 pp". Animate these numbers appearing.]**

> "The final system achieves 57.35% on SpiNNaker, just 2.15 percentage points behind the software simulation. And we did not stop at one model."

**[SCREEN: Show a grid/matrix filling up -- 10 pruning levels x 5 folds = 50 cells, each lighting up green. Show the Pareto frontier plot (accuracy vs energy/sparsity).]**

> "We deployed 50 pruned model variants across 10 sparsity levels and all 5 folds, creating an accuracy-energy Pareto frontier. At 60% pruning, SpiNNaker actually beats the software baseline by half a percentage point -- the hardware prefers sparse, pruned networks. At 95% pruning, we still retain 54% accuracy with 95% fewer active weights. These 50 deployments represent, to our knowledge, the largest SNN audio deployment study on neuromorphic hardware."

**[TRANSITION: Cross-dissolve]**

---

### SEGMENT 8: WHAT SNNs DO BETTER (6:40 - 7:30)

**[SCREEN: Title card: "Beyond Accuracy: Where SNNs Shine"]**

> "Accuracy is not the whole story. SNNs have structural advantages that matter for real-world deployment."

**[SCREEN: Show adversarial robustness comparison. Two bars: SNN 16.55% vs ANN 2.75% under FGSM attack. "6x more robust" label.]**

> "Under adversarial attack -- small, imperceptible perturbations designed to fool the network -- our SNN retains 16.5% accuracy where the ANN collapses to 2.75%. That is six times more robust. Under stronger PGD attacks, the ratio grows to 195 times. The binary spike bottleneck naturally destroys the precise gradient information that adversarial attacks depend on."

**[SCREEN: Show continual learning comparison. SNN forgetting: 69.9%. ANN forgetting: 74.7%. "4.7pp less forgetting" label.]**

> "When trained sequentially on new sound categories, the SNN forgets 4.7 percentage points less than the ANN. Spiking neurons' temporal dynamics act as an implicit regulariser, protecting previously learned representations."

**[SCREEN: Show pruning resilience. At 90% pruning: SNN retains 93.2%, ANN collapses to 36.8%.]**

> "And under extreme pruning -- removing 90% of all weights -- the SNN retains 93% of its original accuracy while the ANN collapses to 37%. Sparse networks built on sparse computation are inherently resilient to further sparsification."

**[TRANSITION: Fade to black]**

---

### SEGMENT 9: CONCLUSION (7:30 - 8:00)

**[SCREEN: Clean slide. Key numbers appearing one at a time:]**
- **First SNN on full ESC-50**
- **7 encodings compared, direct best (47.15%)**
- **Rhythm-SNN: 61.10% (2.75pp from ANN)**
- **PANNs gap collapse: 17pp -> 1pp**
- **SpiNNaker: 57.35%, 50 deployed models**
- **6x adversarial robustness**

> "This thesis makes six contributions. The first convolutional SNN evaluation on the full 50-class ESC-50 benchmark. A systematic seven-encoding comparison revealing that encoding choice matters more than almost any other hyperparameter. A Rhythm-SNN architecture that closes the accuracy gap to under 3 points. A transfer learning result showing the SNN-ANN gap is fundamentally about feature learning, not spiking computation. A 50-model SpiNNaker deployment study. And evidence that SNNs offer structural advantages in robustness, forgetting, and pruning resilience that go beyond energy efficiency."

**[SCREEN: Final slide. Project title, student name, "COMP30040 Third-Year Project". Optional: GitHub URL or QR code.]**

> "The brain doesn't compute with floating point. It computes with spikes -- sparse, binary, and efficient. This thesis shows that for environmental sound classification, spike-based computation is not just viable, but brings unique advantages that conventional networks cannot match."

**[SCREEN: Hold on final slide for 3 seconds. Fade to black.]**

**[END]**

---

## KEY DEMO MOMENTS TO PRE-RECORD

These should be captured as screen recordings before assembling the final video:

1. **ESC-50 audio playback** (Segment 1): Use `aplay` or Python to play 3 clips. Record the waveform visualisation in Audacity or a Python matplotlib animation.

2. **Training in terminal** (Segment 4): Run a training epoch on MPS/CPU (just for visual). Show loss decreasing, accuracy increasing. Speed up 4x in post.

3. **Code walkthrough** (Segment 4): Open `src/models/snn.py` in VS Code. Slowly scroll to the forward method. Highlight the `self.lif1`, `self.lif2` calls.

4. **SpiNNaker deployment** (Segment 7): If you have a recording of a SpiNNaker run, use it. Otherwise, show the terminal output from one of the log files (e.g., `results/spinnaker_results/full_deploy_t3/all_folds_final_log.txt`). Scroll through showing spikes being transmitted.

5. **Results files** (Segment 7): Open `results/MASTER_RESULTS.json` in VS Code. Show the 50 pruned deployment results populating.

---

## SLIDE DESIGNS

Keep slides minimal. Dark background (navy or charcoal), white text, large font. One idea per slide. All figures should be the existing publication-quality figures from `paper/figures/`.

| Slide | Content | Notes |
|-------|---------|-------|
| Title | "The Sound of Spikes" + name + COMP30040 | Full-bleed waveform background |
| Energy | "4.6 pJ (MAC) vs 0.9 pJ (AC)" | Two large numbers, 5x arrow |
| Sparsity | Animated grid of neurons | 74% dark, 26% firing |
| LIF | Membrane potential animation | Rising, leaking, spiking |
| Architecture | `architecture_diagram.png` | Layer-by-layer reveal |
| Encoding question | "Does it matter how you convert sound to spikes?" | White text on black |
| Bar chart | `encoding_bar_chart.png` | Progressive reveal |
| Rhythm-SNN | "61.10%" large, gap closing animation | Before/after |
| PANNs | "17pp -> 1pp" | The money shot |
| Key insight | "When both get equal features, SNN = ANN" | Hold 3 seconds |
| SpiNNaker | Board photo + pipeline diagram | `spinnaker_pipeline.png` |
| 50 models | Grid of 50 green cells filling | Pareto frontier |
| Adversarial | "6x more robust" | Side-by-side bars |
| Contributions | 6 bullet points, appearing one at a time | Clean list |
| Closing | Title + name | Fade out |

---

## NARRATION TIPS

- **Pace:** ~150 words per minute. The script above is approximately 1,200 words = 8 minutes at this pace. Rehearse with a stopwatch.
- **Tone:** Confident but not arrogant. You are explaining YOUR work to a peer. Not lecturing. Not selling.
- **Pauses:** After every major number reveal (47.15%, 61.10%, 92.50%, 57.35%), pause for 1.5 seconds. Let the number land.
- **Emphasis words:** Practise stressing "first", "seven", "six times", "less than one percentage point", "50 deployed models".
- **Avoid:** Jargon without explanation. Never say "surrogate gradient" without having explained it. Never say "LIF" without having defined it. Never say "mel spectrogram" without a visual.

---

## AUDIO PRODUCTION

1. Record narration in one continuous take per segment. Quiet room, USB mic or AirPods Pro close to mouth.
2. Run through Audacity: noise reduction, normalise to -3dB, compress dynamic range slightly.
3. Optional: subtle background music (royalty-free ambient/electronic, very low volume ~-20dB below voice). Suggested: YouTube Audio Library "Ambient" category. Music should be felt, not heard.
4. ESC-50 clips: use the actual .wav files from `data/ESC-50-master/audio/`. Pick clear, recognisable examples (dog bark = `1-100032-A-0.wav` or similar).

---

## EDITING CHECKLIST

- [ ] All audio clips are from ESC-50 dataset (cite Piczak 2015)
- [ ] Architecture diagram matches the actual model (622K params, T=25)
- [ ] All numbers match the 5-fold results (double-check against MASTER_RESULTS.json)
- [ ] SpiNNaker numbers: 57.35% mean, 2.15pp gap (unpruned T=3)
- [ ] PANNs: SNN 92.50%, ANN 93.45%, gap 0.95pp
- [ ] Rhythm-SNN: 61.10% +/- 1.99%
- [ ] Adversarial: 6x (FGSM), 195x (PGD)
- [ ] Forgetting: 69.9% SNN vs 74.7% ANN = 4.7pp less
- [ ] Pruning resilience: 93.2% retained (SNN) vs 36.8% (ANN) at 90% pruning
- [ ] Total duration: 7:00-9:00
- [ ] No copyrighted music
- [ ] Name and COMP30040 visible on title and closing slides
- [ ] Export at 1080p minimum

---

## TIMING CONTINGENCY

If running **long** (>8:30 after first edit):
- Cut Segment 3 (What is an SNN?) from 60s to 40s -- skip the encoding animation, just mention "seven encodings" and move on
- Shorten Segment 8 (What SNNs Do Better) -- keep only adversarial robustness, drop forgetting and pruning

If running **short** (<7:00 after first edit):
- Expand Segment 7 (Real Hardware) -- add more detail on the SpiNNaker debugging story (v_rest bug)
- Add a brief "what I would do with more time" section before the conclusion (Loihi 2 comparison, larger datasets, real-time demo)

---

## RECORDING SCHEDULE

| Day | Task | Time |
|-----|------|------|
| Day 1 | Write final narration script (refine from above). Rehearse 3x. | 2 hours |
| Day 1 | Record all screen demos (terminal, code, figures) | 1 hour |
| Day 2 | Record narration audio (all 9 segments) | 1 hour |
| Day 2 | Assemble in iMovie: lay audio, add visuals, transitions | 2 hours |
| Day 3 | Review, fix timing issues, re-record any weak segments | 1 hour |
| Day 3 | Final export, watch start-to-finish, submit | 30 min |
| **Total** | | **~7.5 hours** |
