# SpiNNaker debugging notes

notes i wrote so i don't forget how to debug the zero-spike problem next time.
i kept getting 0 spikes so here's what i tried and what each test checks.

## background: why spikes die

snnTorch `snn.Leaky` and sPyNNaker `IF_curr_exp` differ in one critical way:

| Property | snnTorch snn.Leaky | sPyNNaker IF_curr_exp |
|---|---|---|
| Synaptic current decay | None (instantaneous) | Exponential, tau_syn_E ms |
| Update equation | `mem = beta*mem + I` | `dI/dt = -I/tau_syn`, `dV/dt = -(V-Vrest)/tau_m + I/cm` |
| Current delivered | 100% of weight per spike | `(1 - exp(-dt/tau_syn))` fraction per spike |
| With tau_syn=1, dt=1 | 100% | 63.2% (36.8% lost per step) |

With `tau_syn_E=1.0` ms and `dt=1.0` ms, 36.8% of each spike's current is lost
to synaptic decay before reaching the membrane. Trained weights were optimised
assuming 100% delivery. This mismatch is the primary suspect.

A secondary issue is that with 2304 input neurons across 25 timesteps, up to
several hundred simultaneous spike packets can overflow the UDP buffer on the
SpiNNaker link (OSError: No buffer space available, errno 55).

---

## Debug Scripts: What Each One Tests

### debug_01_can_fire.py
**Question:** Can IF_curr_exp fire at all with our LIF parameters?

Tests one input neuron firing every timestep into one output neuron, at three
weight values: 5.0 (very generous), 1.0 (our scale), 0.1 (small).

Prints the full voltage trace (all 25 values) and PASS/FAIL for each weight.

**Use this first.** If weight=5.0 does not fire, the board or neuron model
is broken. Nothing else will work until this passes.

### debug_02_tau_syn.py
**Question:** Does reducing tau_syn_E (faster current injection) allow firing?

Tests the same 1-neuron setup at weight=1.0 with tau_syn_E in
{0.1, 0.5, 1.0, 5.0} ms. Prints full voltage trace for each.

A result where 0.1 passes but 1.0 fails confirms that synaptic current
decay is the primary cause of the zero-spike problem.

### debug_03_two_layer.py
**Question:** If layer 1 fires, does layer 2 fire?

10 inputs -> 5 hidden -> 3 output, all-to-all, weight=2.0 excitatory.
Records and prints per-timestep spike counts for all three layers.

Shows whether the hidden-to-output pathway propagates correctly,
independent of weight magnitude issues.

### debug_04_real_weights.py
**Question:** With our actual trained weights, does any voltage accumulate?

Uses the real fc1_connections.npy but only for the first 20 hidden neurons
(to avoid UDP overflow). Prints per-neuron spike counts, max voltages, and
connection weight statistics.

If any of the 20 neurons fire: weights are sufficient, the problem is scale.
If none fire but voltage > 0: we can see the scale factor needed directly.
If voltage = 0: current is not reaching the membrane at all (tau_syn issue).

### debug_05_weight_scale.py
**Question:** What multiplier makes our weights work?

Same 20-neuron setup as debug_04, sweeps `weight_scale` in {1, 2, 5, 10, 20, 50}.
Prints spike count and firing neurons for each scale. Reports `FIRST WORKING SCALE`.

The output directly tells you what `--weight-scale` to pass to run_on_spinnaker.py.

---

## How to Run Them in Order

All scripts must be run from the project root (snn-esc50 directory).

```bash
cd /path/to/snn-esc50
source .venv-spinnaker/bin/activate

# Step 1: Verify basic neuron firing
python spinnaker/debug_01_can_fire.py

# Step 2: Check tau_syn effect (run if debug_01 shows weight=5 fires but weight=1 does not)
python spinnaker/debug_02_tau_syn.py

# Step 3: Check two-layer propagation (run if debug_01 passes)
python spinnaker/debug_03_two_layer.py

# Step 4: Test with real trained weights (run after debug_01 confirms basic firing)
python spinnaker/debug_04_real_weights.py

# Step 5: Calibrate weight scale (run if debug_04 fails or shows partial activity)
python spinnaker/debug_05_weight_scale.py
```

Each script is self-contained. You do not need to run them all if an earlier
one gives a conclusive answer.

---

## Decision Tree: Interpreting Results

```
debug_01: weight=5.0 fires?
│
├── NO  --> Hardware or config broken.
│           Check VPN to Manchester, ping spinnaker.cs.man.ac.uk
│           Check ~/.spynnaker.cfg has correct server address
│           Try rebooting the spalloc job: salloc --machine SpiNNaker1...
│
└── YES --> debug_01: weight=1.0 fires?
            │
            ├── YES --> Weights at our scale are sufficient.
            │           Go to debug_04 to confirm with real weights.
            │           If debug_04 fails: apply mild scaling (debug_05).
            │
            └── NO  --> debug_02: which tau_syn fires with weight=1.0?
                        │
                        ├── tau=0.1 fires, tau=1.0 does not:
                        │   tau_syn IS the problem.
                        │   Fix: run_on_spinnaker.py --tau-syn 0.1
                        │   Better: run_on_spinnaker.py --neuron-model IF_curr_delta
                        │
                        ├── None fire at any tau_syn with weight=1.0:
                        │   Weight too small AND tau_syn may compound.
                        │   Run debug_05 to find scaling factor.
                        │   Also try --neuron-model IF_curr_delta.
                        │
                        └── debug_03: two-layer propagation?
                            │
                            ├── Both layers fire (weight=2.0): architecture OK.
                            │   Problem is specifically weight magnitude.
                            │   Use debug_05 calibration.
                            │
                            └── Layer 1 fires, layer 2 does not:
                                Hidden-to-output pathway broken.
                                FC2 weights also need scaling.
                                Run debug_05 then check FC2 weights separately.
```

---

## New Flags in run_on_spinnaker.py

### --weight-scale FLOAT (default: 1.0)

Multiplies all FC1 and FC2 weights by this factor before building connection lists.
Use the value found by debug_05_weight_scale.py.

```bash
# Apply 10x scaling (found by debug_05 to be the minimum that fires)
python spinnaker/run_on_spinnaker.py --weight-scale 10

# Debug mode: 20 hidden neurons, 5x scale, 3 samples
python spinnaker/run_on_spinnaker.py --weight-scale 5 --max-hidden 20 --num-samples 3
```

### --tau-syn FLOAT (default: 1.0)

Sets tau_syn_E and tau_syn_I for IF_curr_exp. Lower values inject current faster,
closer to snnTorch's instantaneous delivery.

```bash
# Near-instantaneous synaptic current (closest to snnTorch)
python spinnaker/run_on_spinnaker.py --tau-syn 0.1

# Combined: faster current AND weight scaling
python spinnaker/run_on_spinnaker.py --tau-syn 0.1 --weight-scale 3
```

Only applies to `IF_curr_exp`. Has no effect with `--neuron-model IF_curr_delta`.

### --neuron-model {IF_curr_exp,IF_curr_delta} (default: IF_curr_exp)

`IF_curr_delta` has no synaptic current decay. Each presynaptic spike is
delivered as a direct voltage step: `V += weight`. This is the closest
sPyNNaker approximation to snnTorch's `snn.Leaky` neuron.

```bash
# Use delta neuron (most like snnTorch)
python spinnaker/run_on_spinnaker.py --neuron-model IF_curr_delta

# Delta neuron with weight scaling
python spinnaker/run_on_spinnaker.py --neuron-model IF_curr_delta --weight-scale 5
```

### --max-hidden INT (default: 256)

Restricts inference to only the first N hidden neurons (out of 256). This
dramatically reduces the number of connections transferred over UDP, which
fixes the `OSError: No buffer space available` error from Run 3.

```bash
# Use 20 hidden neurons (safe: 20*2304 = 46K connections)
python spinnaker/run_on_spinnaker.py --max-hidden 20

# Use 64 hidden neurons (moderate: 64*2304 = 148K connections)
python spinnaker/run_on_spinnaker.py --max-hidden 64
```

Note that accuracy will be degraded when using fewer hidden neurons, since
the network was trained with all 256.

### --prune-threshold FLOAT (default: 0.05)

Remove connections with |weight| below this value before sending to SpiNNaker.
Higher values reduce connection count (and UDP load) at the cost of accuracy.

```bash
# Aggressive pruning: keep only connections with |w| > 0.2
python spinnaker/run_on_spinnaker.py --prune-threshold 0.2

# Combine pruning and hidden neuron reduction
python spinnaker/run_on_spinnaker.py --prune-threshold 0.1 --max-hidden 64
```

---

## Combining Flags: Recommended Debugging Sequence

Start minimal and expand:

```bash
# Stage 1: Minimum viable network (safest, should always work if hardware is up)
python spinnaker/run_on_spinnaker.py \
    --max-hidden 20 --weight-scale 10 --tau-syn 0.1 \
    --neuron-model IF_curr_delta --num-samples 1

# Stage 2: Add more hidden neurons if Stage 1 works
python spinnaker/run_on_spinnaker.py \
    --max-hidden 64 --weight-scale 10 --neuron-model IF_curr_delta \
    --num-samples 3

# Stage 3: Full network with calibrated settings
python spinnaker/run_on_spinnaker.py \
    --weight-scale 10 --neuron-model IF_curr_delta \
    --num-samples 10
```

---

## Querying the Provenance SQLite3 Database

sPyNNaker records internal diagnostics in a SQLite3 database after each run.
run_on_spinnaker.py prints the location at the end of the run.

```bash
# Show all provenance data (first 50 rows)
sqlite3 "/path/to/provenance_data/run.sqlite3" \
    "SELECT * FROM provenance_data LIMIT 50;"

# Show spike-related entries only
sqlite3 "/path/to/provenance_data/run.sqlite3" \
    "SELECT description, the_value FROM provenance_data
     WHERE description LIKE '%spike%';"

# Show all tables in the database
sqlite3 "/path/to/provenance_data/run.sqlite3" ".tables"

# Show schema for provenance_data table
sqlite3 "/path/to/provenance_data/run.sqlite3" \
    ".schema provenance_data"

# Find router saturation and dropped packets
sqlite3 "/path/to/provenance_data/run.sqlite3" \
    "SELECT description, the_value FROM provenance_data
     WHERE description LIKE '%drop%' OR description LIKE '%saturate%';"

# Find memory overflow information
sqlite3 "/path/to/provenance_data/run.sqlite3" \
    "SELECT description, the_value FROM provenance_data
     WHERE description LIKE '%memory%' OR description LIKE '%overflow%';"
```

The provenance database is typically saved to:
- `~/spynnaker_output/<timestamp>/provenance_data/run.sqlite3`
- Or a subdirectory of your current working directory named `spynnaker_output/`

If the database is not found, verify that `SPYNNAKER_DISABLE_PROVENANCE=0` is
set in the environment (run_on_spinnaker.py sets this automatically via `os.environ`).

---

## Common Errors and Solutions

### OSError: [Errno 55] No buffer space available

Too many connections being transferred at once over UDP to SpiNNaker.

Solutions (in order of preference):
1. `--max-hidden 20` (reduces to 46K connections)
2. `--prune-threshold 0.1` or `--prune-threshold 0.2`
3. Both combined: `--max-hidden 64 --prune-threshold 0.1`

### WARNING: Danger of SpikeSourceArray sending too many spikes at the same time

The input layer is sending hundreds of spikes in the same timestep.
This is expected given our input encoding (up to ~1400 simultaneous spikes).
It is a warning, not an error. The simulation will still run, but may have
timing inaccuracies if the router is saturated.

Mitigation: This is inherent to our rate-coded input representation.
If accuracy suffers, consider spreading spikes across more timesteps in
extract_features.py using a sparser encoding.

### SpinnmanTimeoutException / board not responding

1. Verify VPN connection to Manchester university network
2. `ping spinnaker.cs.man.ac.uk`
3. Wait a few minutes and retry (the board may be in use by another job)
4. Check the spalloc job queue

### 0 hidden spikes despite input spikes present

Run the debug scripts in order: debug_01 -> debug_02 -> debug_04 -> debug_05.
The decision tree above covers all cases.
