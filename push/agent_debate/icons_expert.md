# ICONS Expert Assessment: SNN-ESC50 Paper
**Role:** ICONS 2019-2025 proceedings expert, review criteria specialist
**Paper:** Convolutional SNN for ESC-50 environmental sound classification
**Date:** 9 March 2026

---

## 1. What ICONS Actually Cares About: Criteria That Matter Most

ICONS is not a general ML conference that rewards accuracy maximisation. It is a community conference for people who build neuromorphic systems — hardware engineers, SNN algorithm researchers, deployment practitioners, and neuroscientists who care about whether spiking computation is useful. This shapes everything about what reviewers value.

### What ICONS rewards, in rough order of weight:

**1. Hardware-grounded work.** The most valued contributions at ICONS are papers that actually touch neuromorphic hardware — SpiNNaker, Loihi, BrainScaleS, Xylo, Akida. The 2025 program had explicit hardware deployment papers in both the full-talk and lightning-talk slots (Arfa et al. SpiNNaker2, Meszaros et al. Loihi 2, Muller-Cleve et al. tactile sensing). A paper without hardware can still be accepted, but hardware deployment is a strong positive signal that says "this person actually had to make things work on real silicon."

**2. First-time benchmark establishment.** ICONS explicitly welcomes "benchmark tasks for neuromorphic computing" in its call for papers. Establishing that a domain has never been benchmarked under SNN conditions — and providing the first rigorous numbers — is a recognised contribution. The field needs to know where SNNs currently stand on a variety of tasks before it can improve.

**3. Algorithmic novelty with clear neuromorphic motivation.** Methods papers need a clear story about why the proposed approach matters for neuromorphic execution — energy, sparsity, hardware compatibility, on-chip learning. The 2025 best paper (turbulence modelling) had no accuracy metric at all; it was accepted because it proposed a novel connection between neuron dynamics and physical systems.

**4. Systematic ablation or comparison studies.** Papers that rigorously compare design choices (encodings, surrogate functions, neuron models, hardware configurations) are valued as reference works for the community. Yarga et al. ICONS 2022 (4-encoding speech comparison) is the canonical example of this accepted type.

**5. Honest negative results with mechanistic explanation.** ICONS accepts papers where the SNN does not win — provided the authors explain mechanistically why, and what the results imply for future design. The community is mature enough to value "we tried this and it failed, here is why" as a genuine contribution.

### What ICONS does NOT primarily reward:

- Raw accuracy numbers relative to ANN SOTA
- Parameter efficiency in isolation
- Beating leaderboards
- Applications that are just ML problems with no neuromorphic angle

---

## 2. What Reviewers Would Focus On Positively

For this specific paper, ICONS reviewers will likely respond strongly to the following:

**A. The "first SNN on full ESC-50" claim.** This is a clean, verifiable novelty claim. There is no prior SNN paper on 50-class ESC-50. Reviewers can check this. Larroza et al. (arXiv:2503.11206) only covers ESC-10 with a fully-connected network. Dominguez-Morales (ICANN 2016) uses pure tones, not real soundscapes. The claim is watertight, and the ICONS community will recognise this as filling a genuine gap.

**B. The 7-encoding systematic comparison.** This is the most comprehensive encoding comparison in the SNN audio literature. Yarga et al. (ICONS 2022) — the only directly comparable prior paper — did 4 encodings on speech digits. This paper does 7 on a harder 50-class task. ICONS reviewers know the Yarga paper and will see this as a meaningful extension.

**C. The SpiNNaker deployment.** Hardware work gets immediate credibility at ICONS. The fact that it is a validated FC2-only hybrid approach with documented root-cause analysis of the FC1 cancellation problem is actually better than a naive full-network deployment, because it shows the authors understand what neuromorphic hardware actually requires and documented a real constraint that others will encounter. The 5-fold cross-validation on SpiNNaker (not just a single run) is methodologically stronger than most hardware deployment papers.

**D. The adversarial robustness finding (14.9x advantage).** This is a striking, counterintuitive result with practical implications. Under FGSM at eps=0.1, the SNN retains 26% versus 1.75% for the ANN. Reviewers from the security side of the neuromorphic community (there is a growing subgroup; ICONS 2025 included "Do Spikes Protect Privacy?" and "Neuromorphic Cybersecurity") will find this immediately compelling.

**E. The PANNs+SNN gap-collapse finding.** The finding that the SNN-ANN gap collapses from 16.7 pp to 0.95 pp with frozen AudioSet-pretrained features reframes the entire paper's narrative from "SNN underperforms" to "the bottleneck is feature learning, not spiking computation." This is a novel, clean scientific insight that generalises beyond the specific benchmark.

**F. NeuroBench compliance.** Using the NeuroBench framework (Yik et al., Nature Communications 2025) to report energy numbers is exactly what the community has been pushing for. It signals methodological alignment with emerging community standards. Reviewers will recognise this positively.

---

## 3. What Reviewers Would Flag as Weaknesses

**A. The hybrid SpiNNaker deployment is not a full-network deployment.** The most technically sophisticated reviewers will note that only FC2 (256→50) runs on SpiNNaker while the convolutional layers and FC1 run in software. This is less than ideal, and reviewers may ask: "what is the hardware contribution if most of the compute is in software?" The paper must pre-empt this by clearly explaining the FC1 cancellation root cause (AvgPool produces fractional, not binary, outputs) and presenting it as a design insight rather than a shortcoming. The Option A threshold sweep (showing that MaxPool replacement achieves fc1_binary_fraction=1.0) demonstrates the authors have thought about the path forward.

**B. The hardware-software accuracy gap (12.8 pp over 5 folds) is not explained with a clean fix.** Reviewers will want to know: is this gap from weight quantisation? From spike-count discretisation? From the FC2-only constraint? The paper should cite the DYNAP-SE comparison (7.1 pp on a simpler task) and Loihi 2 work to contextualise the gap as expected for a first-generation deployment without quantisation-aware training.

**C. Only 1,600 training samples per fold.** A reviewer who is sceptical about the SNN performance numbers may argue that the task is underpowered and that the 47.15% result reflects data scarcity more than any fundamental SNN property. The PANNs+SNN result partially addresses this, but the paper should acknowledge this explicitly rather than leaving the reviewer to raise it.

**D. The surrogate gradient ablation is single-seed on a single fold.** The current 1-seed, fold-1 result is preliminary. If the 3-seed CSF3 run completes before submission, include it. If not, label the surrogate section clearly as "fold 1, n=1 seed, preliminary" and put it in the appendix or a footnote rather than a main result. ICONS reviewers are technically sophisticated and will flag under-replicated ablations.

**E. The energy claim needs careful framing.** The current result (SNN 976 nJ vs ANN 463 nJ in simulation) shows the SNN is 2.1x MORE expensive in software simulation. Reviewers who know the Dampfhoffer et al. (2023) threshold analysis will recognise that the 25.8% spike rate far exceeds the ~6-8% break-even point. The paper must not claim energy efficiency in simulation; it must frame the hardware advantage argument carefully (ACs cost 5.1x less than MACs on physical hardware, but the total operation count is still higher). Some reviewers will push hard on this.

**F. The continual learning result is thin for a main contribution.** SNN forgetting 74.4% vs ANN 81.3% is interesting but the 6.9 pp advantage is modest and was measured on only one fold with pretrained features. At 8 pages, this may be hard to include with sufficient rigour. Consider reducing it to one sentence in the conclusion rather than a table.

---

