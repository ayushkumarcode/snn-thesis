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
