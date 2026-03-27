# SpiNNaker / Neuromorphic Engineering Extension Ideas

ranked by potential impact for the ICONS paper.

## Priority 1: SpiNNaker Energy Measurement (2-3 days) -- fills the biggest gap

use sPyNNaker provenance data (SQLite database auto-generated after each run) + published per-chip power figures to get REAL hardware energy numbers. right now the NeuroBench analysis is theoretical (nJ per operation). actual measurements would be gold at ICONS.
- query: synaptic events processed, spikes transmitted, dropped packets, timer durations
- SpiNNaker 1: ~1W active, ~255mW idle, ~5.9 uJ per synaptic event
- for FC2: estimated single-digit millijoules per inference

## Priority 2: Full Deploy via IF_cond_exp + MaxPool (2-3 days) -- could close accuracy gap

switch from `IF_curr_exp` (current-based) to `IF_cond_exp` (conductance-based). in conductance-based models, inhibition is **shunting** -- synaptic current is `g_syn * (V - E_rev)`, so inhibitory inputs can't drive membrane below reversal potential. this prevents the catastrophic cancellation that killed full deployment. combine with MaxPool retraining (Option A, already done).

## Priority 3: Spike Drop Robustness Analysis (1-2 days) -- explains the hardware gap

SpiNNaker drops spike packets non-deterministically. artificially drop X% of input spikes and measure accuracy degradation. if network degrades gracefully -> fault tolerance finding. also explains part of the 8.25pp hardware gap. check provenance data for actual drop counts.

## Priority 4: WTA Lateral Inhibition (1-2 days)

add inhibitory connections between 50 output neurons. forces single winner, sharpens output. could improve SpiNNaker accuracy where output spikes spread across classes.

## Priority 5: On-Chip STDP Learning (5-7 days) -- headline if it works

deploy FC2 with STDP enabled, fine-tune on-chip using teacher signals. sPyNNaker supports SpikePairRule, Vogels2011Rule, etc. would be the first on-chip learning for audio on SpiNNaker.

## Priority 6: Liquid State Machine on SpiNNaker (5-7 days) -- most novel

500-1000 LIF reservoir with random recurrent connections, only readout trained. runs entirely on SpiNNaker. no LSM for ESC-50 exists. avoids cancellation problem (random weights, chosen distribution).

---

## Other Ideas (Lower Priority)

| # | Idea | Time | Novelty |
|---|------|------|---------|
| 7 | Poisson noise robustness (SpikeSourcePoisson background) | 1-2d | Medium |
| 8 | SpiNNaker 2 readiness via NIR export + 8-bit quantization | 2-3d | Medium |
| 9 | Izhikevich resonator neurons as neuromorphic filterbank | 3-5d | High |
| 10 | Structural plasticity (grow/prune connections on-chip) | 4-5d | High |
| 11 | Loihi comparison via NIR + Lava simulator | 3-5d | Medium-High |
| 12 | Real-time audio streaming via SpynnakerLiveSpikesConnection | 3-4d | Medium |
| 13 | Sub-millisecond timestep precision | 2-3d | Medium |
| 14 | Multi-chip ensemble (parallel classifiers, majority vote) | 2-3d | Medium-High |

---

## Technical Details

### IF_cond_exp Parameters
```python
lif_params_cond = {
    "cm": 1.0, "tau_m": 20.0, "tau_refrac": 0.1,
    "v_reset": -65.0, "v_rest": -65.0, "v_thresh": -50.0,
    "tau_syn_E": 5.0, "tau_syn_I": 5.0,
    "e_rev_E": 0.0, "e_rev_I": -80.0,  # prevents cancellation!
}
```

### Energy Provenance Query
```sql
SELECT description, the_value
FROM provenance_data
WHERE description LIKE '%dropped%' OR description LIKE '%synaptic%';
```

### WTA Implementation
```python
wta_conns = [(i, j, wta_weight, 1.0) for i in range(50) for j in range(50) if i != j]
sim.Projection(output_pop, output_pop, sim.FromListConnector(wta_conns),
    receptor_type="inhibitory")
```
