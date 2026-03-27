# SpiNNaker / Neuromorphic Engineering Extensions
*Generated: 15 March 2026*

## Top Ideas — Ranked by ICONS Impact

### Priority 1: SpiNNaker Energy Measurement (2-3 days) — FILLS BIGGEST GAP
Use sPyNNaker provenance data (SQLite database auto-generated after each run) + published per-chip power figures to get REAL hardware energy numbers. Current NeuroBench analysis is theoretical (nJ per operation). Real measurements are gold at ICONS.
- Query: synaptic events processed, spikes transmitted, dropped packets, timer durations
- SpiNNaker 1: ~1W active, ~255mW idle, ~5.9μJ per synaptic event
- For FC2: estimated single-digit millijoules per inference

### Priority 2: Full Deploy via IF_cond_exp + MaxPool (2-3 days) — COULD CLOSE ACCURACY GAP
