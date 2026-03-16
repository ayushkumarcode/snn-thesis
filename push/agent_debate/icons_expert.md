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

## 4. Comparison to Yarga et al. ICONS 2022 — Is This Paper More or Less Strong?

**Yarga et al. (ICONS 2022):** "Efficient Spike Encoding Algorithms for Neuromorphic Speech Recognition" (Yarga, Rouat, Wood — Universite de Sherbrooke)
- 4 encoding methods (Send-on-Delta variants, TTFS, LIF, BSA)
- Speaker-independent digit classification (10 classes)
- Key result: Send-on-Delta matched CNN baseline while reducing spike bit rate
- No hardware deployment
- No adversarial/continual/energy analysis

**Comparison:**

| Dimension | Yarga et al. 2022 | This paper |
|-----------|------------------|------------|
| Encodings compared | 4 | 7 |
| Task difficulty | 10-class digits | 50-class environmental sounds |
| Hardware deployment | None | SpiNNaker 5-fold |
| Energy analysis | None | NeuroBench-compliant |
| Adversarial analysis | None | FGSM + PGD, 7 eps values |
| Transfer learning | None | PANNs+SNN, gap-collapse finding |
| Surrogate ablation | None | 8 surrogates |
| Continual learning | None | 5-task sequential |
| Negative results documented | Partial | Full (burst, population, augmentation) |
| Novel insight | Encoding efficiency trade-off | Feature-learning bottleneck hypothesis |

**Verdict:** This paper is substantially more comprehensive than Yarga et al. on every dimension — more encodings, harder task, hardware deployment, and multiple additional analyses. Yarga et al. was accepted as a full paper at ICONS 2022. By the paper-quality bar of ICONS 2022, this paper is stronger. By the 2025 bar (the conference has grown and the quality threshold may have risen), this paper is still competitive if the hardware story is well-framed.

The key question is not whether this paper is better than Yarga et al. — it clearly is. The key question is whether the 8-page limit allows the paper to present all contributions at sufficient depth, or whether reviewers will feel each individual contribution is too shallow. This is the primary tension.

---

## 5. Is 47.15% Accuracy "Significant" at ICONS? What Is the Right Framing?

**Short answer:** 47.15% is entirely acceptable at ICONS when properly contextualised. It is not the number that will make or break the paper.

**The correct framing for an ICONS audience:**

First, ICONS reviewers understand that 47.15% on a 50-class dataset with a random baseline of 2% (random = 2%) represents a 45.15 pp improvement over chance. They will not compare it to 98.25% ANN SOTA without context. They will ask: "given the architecture, the dataset size, and the training approach, what does this tell us about the current capability of directly-trained convolutional SNNs on a challenging audio benchmark?"

Second, the ICONS community's point of reference for SNN accuracy is not ImageNet or ESC-50 SOTA — it is other SNN papers on comparable tasks. In that context:
- Yarga et al. (ICONS 2022) matched CNN baseline on 10-class digits with Send-on-Delta
- The ICONS 2024 audio paper (Schmitt et al.) focused on real-time capability, not absolute accuracy
- ICONS 2025 paper #27 (Vasilache et al. on vibration predictive maintenance) focused on demonstrating SNN feasibility in a new domain

47.15% is not "low" in this context — it is the first SNN result on this task. There is no prior bar to beat.

Third, the PANNs+SNN result (92.50%) is the decisive rehabilitation. It proves that the 47.15% gap is not inherent to spiking computation. ICONS reviewers will understand that the 16.7 pp gap in scratch training — and its near-total collapse to 0.95 pp with pretrained features — is the scientific contribution. This is a cleaner and more interesting story than simply achieving high accuracy.

**The framing that will land best at ICONS:** Frame the 47.15% as "establishing a baseline for future SNN audio work" and the gap-collapse finding as "demonstrating that the SNN formalism is not the bottleneck." Do not apologise for the number. Do not compare it to ANN SOTA without explanation. Do contextualise it against the 2% random baseline, the 81.3% human baseline, and the PANNs+SNN result.

**What "significant" means at ICONS:** Significance at ICONS is not statistical significance or accuracy significance — it is novelty significance. The first SNN result on a well-known benchmark IS significant, regardless of the absolute value.

---

## 6. Does the SpiNNaker Deployment Carry Weight at ICONS?

**Yes, substantially — but the framing matters.**

Hardware deployment papers are first-class citizens at ICONS. The 2025 program included a dedicated SpiNNaker2 paper (Arfa et al., "Hardware-Aware Fine-Tuning of Spiking Q-Networks on the SpiNNaker2 Neuromorphic Platform") in the lightning-talk slot. The 2024 program included "Towards Efficient Deployment of Hybrid SNNs on Neuromorphic and Edge AI Hardware" (Seekings et al.). The community actively publishes work that grapples with real hardware constraints.

The FC2-only hybrid approach is defensible at ICONS — and the key is to frame the FC1 cancellation problem as a research finding, not a failure. The root cause (AvgPool produces fractional outputs that cannot serve as binary spike inputs to SpiNNaker's IF neurons) is a genuinely non-obvious hardware-software co-design constraint. Publishing this analysis helps future researchers avoid the same problem. The Option A result (MaxPool replacement → fc1_binary_fraction=1.0 for all thresholds) shows the path to full deployment.

The 5-fold SpiNNaker validation (2,000 total hardware inferences) is methodologically stronger than most hardware deployment papers, which typically report a single run. The fold-level variance (F1=29.0%, F2=32.0%, F3=36.5%, F4=43.0%, F5=25.2%) is reported honestly.

**What carries weight:**
- 33.1% ± 6.9% on SpiNNaker (first ESC-50 hardware result)
- 8.25 pp gap on 400-sample fold-4 validation with 64.5% agreement rate
- Root-cause analysis of FC1 cancellation
- Option A path to full deployment

**What may be questioned:**
- "If only FC2 is on hardware, is this really a deployment paper?" — Answer: FC2 executes 256→50 classification on live neuromorphic hardware. That is real deployment. The hybrid approach is the current practical reality given SpiNNaker's binary input constraint.
- "Why is the gap 12.8 pp?" — Need to contextualise against DYNAP-SE (7.1 pp on simpler task), Loihi 2 (near-zero with QAT), and the SpiNNaker 1 platform's 2012-era 130nm design.

At ICONS, 33.1% on SpiNNaker for 50-class environmental sound classification — the first such deployment — carries meaningful weight. It is not 91.12% on 12-class speech commands (SpiNNaker2), but it is a more complex task on older, more constrained hardware.

---

## 7. What the Paper Should Emphasise and De-emphasise for ICONS

### Emphasise:

**Primary message:** "We establish the first SNN baseline on ESC-50 (50-class), deploy it on SpiNNaker hardware, and demonstrate that the SNN-ANN accuracy gap is a feature-learning problem, not a spiking computation problem."

**Secondary messages (in order of ICONS impact):**
1. SpiNNaker deployment: FC1 cancellation root cause, FC2-only hybrid validated at 5-fold, path to full deployment via Option A
2. 7-encoding systematic comparison: direct >> rate/phase >> population > latency >> delta/burst, with mechanistic explanations for each failure mode
3. Adversarial robustness: 14.9x advantage under FGSM eps=0.1, binary thresholding as natural gradient masking
4. PANNs+SNN gap collapse: 16.7 pp → 0.95 pp, feature-learning bottleneck confirmed
5. NeuroBench energy: honest reporting (SNN 2.1x worse in simulation, but AC-only hardware advantage)
6. Surrogate bimodal split: spike_rate_escape/fast_sigmoid/atan succeed; STE/sigmoid/sfs/triangular fail

### De-emphasise or remove:

- Absolute accuracy comparisons to ESC-50 SOTA (98.25% or OmniVec2 99.1%) — these are not the relevant comparison class for an ICONS audience
- Continual learning result (74.4% vs 81.3% forgetting) — modest effect, one fold, limited depth possible at 8 pages; reduce to one sentence
- Data augmentation negative result (40.75% ± 16.03%) — useful for thesis but takes space in a conference paper; move to appendix or mention in one sentence
- Over-claiming on energy efficiency — do not say "SNNs are more energy efficient" without the hardware qualifier

### Structural note for 8 pages:

Given what ICONS values, the hardware section (SpiNNaker) and the encoding comparison should get the most space. The adversarial result and PANNs result should each get a compact table + 2-3 paragraphs. Surrogate ablation: one table, one paragraph. Continual learning: one sentence in conclusion or omit from main paper. Energy: one table, honest interpretation.

---

## 8. Is This Paper Publishable at ICONS As-Is, With Revisions, or Likely Rejected?

**Assessment: Publishable with moderate revisions. Most likely outcome: acceptance with minor-to-moderate revisions. Not likely to be rejected outright.**

**Rationale:**

The paper has two things that ICONS cannot ignore: (1) a genuine first — first SNN on full ESC-50, first SpiNNaker deployment for environmental sound classification — and (2) depth — 7 encodings, hardware deployment, adversarial, PANNs, NeuroBench, surrogate ablation. No other paper in the 2022-2025 ICONS proceedings has this breadth of contribution on a single task.

The weaknesses (hybrid deployment, energy framing, single-seed surrogate ablation, modest continual learning result) are real but not fatal. They are the kinds of issues that get addressed in revision, not rejection reasons.

**Scenarios:**

- **Best case:** Accept as full paper, 20-minute talk. Probability: ~35%
- **Middle case:** Accept as lightning talk / short paper (4 pages). Probability: ~40%
- **Revision case:** Major revision requested, resubmit for poster. Probability: ~15%
- **Reject:** ~10%

The primary rejection risk is if reviewers feel the 8-page full paper format is too compressed to do justice to each contribution — i.e., each contribution is thin rather than deep. This is a real risk given the breadth. If the paper gets rejected for this reason, it would likely be accept-as-short-paper rather than reject-outright.

**Acceptance probability (full or short paper combined): ~75%.**

---

## 9. Single Most Important Addition to Strengthen Acceptance Odds

**The single most important addition: A clear, quantified "path to full deployment" narrative around Option A (MaxPool SNN).**

Here is why this matters more than any other addition:

The current paper's hardware contribution has a gap: FC2-only deployment is honest but leaves reviewers wondering "so is the SNN actually running on neuromorphic hardware for the hard part of the task, or just the final classification layer?" This is a legitimate concern. The full-network deployment story is essential for the hardware contribution to land as strongly as it should.

The Option A result already exists: threshold=3.0 achieves 43.75% accuracy on fold 4 with fc1_binary_fraction=1.0 (i.e., all FC1 inputs ARE binary). This means full SpiNNaker FC1+FC2 deployment is theoretically validated. What is missing is the actual hardware test of FC1+FC2 on SpiNNaker.

**If there is time to run Option A on SpiNNaker hardware before April 1:** Do it. A result showing FC1+FC2 running on SpiNNaker — even if the accuracy degrades slightly from 43.75% — would transform the hardware contribution from "FC2-only hybrid" to "first full convolutional SNN deployed on SpiNNaker for environmental sound." This single addition would likely move the paper from the lightning-talk tier to the full-talk tier.

**If there is no time to run Option A on hardware:** Add a dedicated subsection (0.5 pages) in the SpiNNaker section explaining: (a) why FC1-only was not feasible (AvgPool constraint), (b) what Option A found (binary fraction=1.0 at threshold≥1.0, 43.75% accuracy), and (c) what the FC1+FC2 deployment would look like in practice. Frame this as "the path to full deployment is validated in simulation; hardware execution is immediate future work." This strengthens the paper without requiring new experiments.

**Secondary priority additions (if space permits):**
1. Complete the 3-seed CSF3 surrogate ablation — replace "fold 1, 1 seed" with "fold 1, 3 seeds" for tighter variance estimates
2. Add one sentence on SA-PGD (Wang et al. 2025) to acknowledge that standard PGD may underestimate SNN robustness at high eps — shows reviewer-level awareness
3. Update ESC-50 SOTA reference from 98.25% to OmniVec2 99.1% (CVPR 2024)

---

## 10. What "Advancing State of the Art or Novel/Original Approaches" Means at ICONS

This is the key framing question for positioning the paper.

**At ICONS, "advancing state of the art" does not mean "higher accuracy than the previous best."** That is the ICLR/NeurIPS definition. At ICONS, it means:

1. **Advancing the set of tasks where neuromorphic systems have been benchmarked.** ESC-50 was previously uncharted territory for SNNs. Establishing the first benchmark results on this task advances the state of the art in the sense that the community's knowledge of what neuromorphic systems can do has been extended.

2. **Advancing understanding of design trade-offs.** Establishing that direct encoding dominates for audio spectrograms (and explaining why: continuous integration allows the LIF membrane to perform its own rate computation) advances what the community knows about encoding choice. Establishing the surrogate bimodal split (3 surrogates learn, 4 fail) advances what the community knows about surrogate gradient selection.

3. **Advancing hardware deployment methodology.** The FC1 cancellation analysis — showing why AvgPool breaks SpiNNaker's binary input assumption, and how MaxPool replacement restores binary compatibility — advances the community's understanding of hardware-software co-design constraints for convolutional SNNs.

**Is 47.15% advancing SoTA?** Yes, trivially — it is the first result on this task. There is no prior SoTA for SNN on ESC-50 to compare against. The relevant prior SoTA is "undefined" (no prior work), and 47.15% > undefined. The paper should state this explicitly.

**Is the deployment novel enough?** Yes. The deployment is novel on three levels: (1) first SpiNNaker deployment for environmental sound classification, (2) first neuromorphic hardware deployment for any 50-class audio task, and (3) first published analysis of FC1 cancellation due to AvgPool in convolutional SNNs on SpiNNaker. Each of these would be a novel contribution on its own; having all three together makes the novelty robust even if one point is contested.

**The critical distinction for ICONS:** The conference is not asking "does your SNN beat the ANN?" It is asking "does your paper advance the community's ability to build and understand neuromorphic systems?" The answer here is clearly yes — on encoding, on hardware deployment, on robustness properties, and on transfer learning.

---

## Summary Assessment Table

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Novelty of contribution | **Strong** | First on ESC-50, first SpiNNaker audio classification since 2016, first adversarial on audio spectrograms |
| Hardware grounding | **Strong** | SpiNNaker 5-fold validated; FC1 cancellation analysis; Option A path forward |
| Technical depth | **Moderate** | Breadth is exceptional; depth per contribution is limited at 8 pages |
| Results quality | **Moderate-Strong** | 47.15% is appropriate; gap-collapse finding is the strongest scientific result |
| Comparison to prior work | **Strong** | Clearly supersedes Yarga et al. ICONS 2022; novelty claim is watertight |
| Community fit (ICONS) | **Very Strong** | SpiNNaker, encoding comparison, NeuroBench, adversarial — all ICONS-core topics |
| Energy analysis | **Moderate** | NeuroBench-compliant but must avoid overclaiming; framing is delicate |
| Writing/framing risk | **Moderate** | Paper must not apologise for 47.15%; must pre-empt FC2-only objection |
| **Overall publishability** | **Accept with revision** | ~75% combined probability of acceptance as full or short paper |

---

## Specific ICONS Norms Not Covered Elsewhere

- **Rebuttals are permitted at ICONS 2026 (due May 25).** This is favorable. If reviewers raise the FC2-only objection or the energy framing issue, there is an opportunity to respond. Prepare a rebuttal strategy in advance.
- **Full papers (8 pages) get priority for 20-minute talks.** Short papers (4 pages) get 10-minute talks. There is no shame in a short paper submission if the 8-page version feels compressed. Consider whether a tightly argued 4-page version with the core contributions (encoding comparison + SpiNNaker + gap collapse) might actually be stronger than a diluted 8-page version.
- **ICONS 2026 is fully open access** via ACM Open. University of Manchester is a participating institution. Zero APC.
- **ORCID required for all authors.** Check this early.
- **The ICONS community is small and interconnected.** Reviewers will likely include people who know SpiNNaker, people who know snnTorch/surrogate gradient work, and people who know audio SNNs. This means reviewers will be well-positioned to evaluate technical details — do not overstate anything.
- **Lightning talks are not second-class.** ICONS 2025 had ~20 lightning talk papers alongside 11 full-talk papers. A lightning talk acceptance is a genuine published paper in the ACM proceedings with a DOI. For an undergraduate thesis, this would be an outstanding outcome.

---

*Assessment based on: ICONS 2022-2025 full paper lists (DBLP, conference schedules), ICONS 2025 best paper (Taylor et al. turbulence), Yarga et al. ICONS 2022 (direct precedent), ICONS 2024 audio paper (Schmitt et al.), ICONS 2025 hardware papers (Arfa SpiNNaker2, Meszaros Loihi 2), NeuroBench community norms (Yik et al. 2025), and survey of SNN audio literature confirming novelty claims (Basu et al. 2502.15056, Kim & Lee 2024, Larroza et al. 2503.11206).*
