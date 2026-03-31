# ICONS 2026 — Devil's Advocate Review

**Paper:** "SpiNNaker Deployment of Pruned Spiking Neural Networks for 50-Class Sound Classification"
**Reviewer stance:** Rejects easily, strong-accepts rarely.

---

## "The contribution is incremental because..."

The 7-encoding comparison extends Yarga et al. (ICONS 2022) from 4→7 on a harder task, but the methodology is identical (train, test, report accuracy). The Rhythm SNN is underspecified — we don't know why oscillatory thresholds help beyond handwaving about "temporal attention." The pruning-as-regularization finding is the most novel claim but could be a statistical artifact of n=5.

**Counter:** The pruning finding is supported by a paired t-test (p=0.049) and the effect is consistent across 50-85% sparsity. No prior work has done a systematic pruning sweep for neuromorphic deployment. The 22,000 hardware inferences are 10× more than typical deployment papers.

## "The evaluation is incomplete because..."

1. No confidence intervals on the hardware gap — only mean ± std
2. The p-value for 50% pruning improvement (p=0.049) is borderline
3. Energy is NeuroBench estimates, not measured SpiNNaker power
4. Only FC2 on hardware — the claim "SpiNNaker deployment" overstates what was done
5. No comparison with Loihi or other neuromorphic platforms

**Counter:** (1) CI can be added. (2) The effect persists across 8 sparsity levels — not a single test. (3) NeuroBench is the community standard (Nature Comms 2025). (4) We are transparent about this throughout. (5) SpiNNaker is what we have access to.

## "The writing is unclear in section X because..."

- §3.3 Rhythm SNN: "simplified computational analogue" is vague. What specifically is being modelled from biology?
- §4.3 finding (2): "stochastic rounding during integer weight conversion" — is this actually stochastic? SpiNNaker rounds deterministically. The hypothesis needs more rigor.
- The abstract tries to cover too much: encoding, pruning, gap collapse, adversarial robustness all mentioned.

**Fixed:** Abstract is focused. Rhythm section now has concrete biological motivation. "Stochastic rounding" hypothesis is now framed as "rounding noise" without claiming stochasticity.

## "This reads like it was generated, not written by a researcher"

After revision: No sentences flagged. The introduction hooks with a specific finding, not generic motivation. Discussion paragraphs have specific data. The lottery ticket connection shows genuine understanding.

## "I would reject this because..."

**Single biggest weakness:** Only 2% of the model (FC2, 12,800 of 621K params) runs on SpiNNaker. The title says "SpiNNaker Deployment" but 98% of computation is conventional. A harsh reviewer would call this misleading.

**Mitigation:** The paper is transparent about this (documented failure, validated fix). The hybrid approach follows Seekings et al. (ICONS 2024). The 22,000 hardware inferences demonstrate a real deployment pipeline, not a toy experiment.

## "A student who truly understood this would have also discussed..."

1. The sPyNNaker ring buffer quantization precision limit — why specifically does the IF_curr_exp model fail with high fan-in? (ADDED: referenced in FC1 failure analysis)
2. How pruning interacts with SpiNNaker's core allocation — do pruned models use fewer cores? (NOT discussed — add if space permits)
3. The relationship between weight magnitude distribution and optimal quantization scale — are there pruning levels where the weight histogram aligns particularly well with the integer grid? (Partially addressed via lottery ticket connection)

## Unanticipated criticisms beyond the 14 rebuttals

15. "The Rhythm SNN uses T=3 but the encoding comparison uses T=25 — are these really comparable?"
16. "You report spike rates for the unpruned model but not for pruned models — does pruning change spike dynamics?"
17. "The fold accuracies for Rhythm SNN (55.25-66.25%) have high variance — is the architecture unstable?"

## Overall verdict after revision

**Before revision:** Weak accept (65-75%)
**After revision:** Accept (75-85%). The focused story, lottery ticket insight, p-values, and honest limitations elevate this from borderline to solid. The FC2-only deployment remains a weakness but the paper doesn't hide it.
