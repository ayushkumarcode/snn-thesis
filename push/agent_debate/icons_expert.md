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

