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

so with `tau_syn_E=1.0` ms and `dt=1.0` ms, 36.8% of each spike's current gets
lost to synaptic decay before it even reaches the membrane. the trained weights
were optimised assuming 100% delivery. this mismatch is the primary suspect.

there's also a secondary issue -- with 2304 input neurons across 25 timesteps,
you can get several hundred simultaneous spike packets that overflow the UDP
buffer on the SpiNNaker link (OSError: No buffer space available, errno 55).

---

## debug scripts: what each one tests

### debug_01_can_fire.py
**question:** can IF_curr_exp fire at all with our LIF parameters?

tests one input neuron firing every timestep into one output neuron, at three
weight values: 5.0 (very generous), 1.0 (our scale), 0.1 (small).

prints the full voltage trace (all 25 values) and PASS/FAIL for each weight.

**run this first.** if weight=5.0 doesn't fire, the board or neuron model
is broken. nothing else will work until this passes.

### debug_02_tau_syn.py
**question:** does reducing tau_syn_E (faster current injection) let it fire?

tests the same 1-neuron setup at weight=1.0 with tau_syn_E in
{0.1, 0.5, 1.0, 5.0} ms. prints full voltage trace for each.

if 0.1 passes but 1.0 fails, that confirms synaptic current decay is
the main cause of the zero-spike problem.

### debug_03_two_layer.py
**question:** if layer 1 fires, does layer 2 fire?

10 inputs -> 5 hidden -> 3 output, all-to-all, weight=2.0 excitatory.
records and prints per-timestep spike counts for all three layers.

shows whether the hidden-to-output pathway propagates correctly,
independent of weight magnitude issues.

### debug_04_real_weights.py
**question:** with our actual trained weights, does any voltage accumulate?

uses the real fc1_connections.npy but only for the first 20 hidden neurons
(to avoid UDP overflow). prints per-neuron spike counts, max voltages, and
connection weight statistics.

if any of the 20 neurons fire: weights are sufficient, problem is scale.
if none fire but voltage > 0: you can see the scale factor needed directly.
if voltage = 0: current isn't reaching the membrane at all (tau_syn issue).

### debug_05_weight_scale.py
**question:** what multiplier makes our weights work?

same 20-neuron setup as debug_04, sweeps `weight_scale` in {1, 2, 5, 10, 20, 50}.
prints spike count and firing neurons for each scale. reports `FIRST WORKING SCALE`.

the output directly tells you what `--weight-scale` to pass to run_on_spinnaker.py.

---

## how to run them in order

all scripts need to be run from the project root (snn-esc50 directory).

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

each script is self-contained. you don't need to run them all if an earlier
one gives you a conclusive answer.

---

## decision tree: interpreting results

```
debug_01: weight=5.0 fires?
|
+-- NO  --> something's broken at the hardware/config level.
|           check VPN to Manchester, ping spinnaker.cs.man.ac.uk
|           check ~/.spynnaker.cfg has the right server address
|           try rebooting the spalloc job: salloc --machine SpiNNaker1...
|
+-- YES --> debug_01: weight=1.0 fires?
            |
            +-- YES --> weights at our scale are fine.
            |           go to debug_04 to confirm with real weights.
            |           if debug_04 fails: try mild scaling (debug_05).
            |
            +-- NO  --> debug_02: which tau_syn fires with weight=1.0?
                        |
                        +-- tau=0.1 fires, tau=1.0 doesn't:
                        |   tau_syn IS the problem.
                        |   fix: run_on_spinnaker.py --tau-syn 0.1
                        |   better: run_on_spinnaker.py --neuron-model IF_curr_delta
                        |
                        +-- none fire at any tau_syn with weight=1.0:
                        |   weight's too small AND tau_syn may be compounding it.
                        |   run debug_05 to find the scaling factor.
                        |   also try --neuron-model IF_curr_delta.
                        |
                        +-- debug_03: two-layer propagation?
                            |
                            +-- both layers fire (weight=2.0): architecture's OK.
                            |   problem is specifically weight magnitude.
                            |   use debug_05 calibration.
                            |
                            +-- layer 1 fires, layer 2 doesn't:
                                hidden-to-output pathway is broken.
                                FC2 weights also need scaling.
                                run debug_05 then check FC2 weights separatley.
```

---

## new flags in run_on_spinnaker.py

### --weight-scale FLOAT (default: 1.0)

multiplies all FC1 and FC2 weights by this factor before building connection lists.
use whatever value debug_05_weight_scale.py finds.

```bash
# Apply 10x scaling (found by debug_05 to be the minimum that fires)
python spinnaker/run_on_spinnaker.py --weight-scale 10

# Debug mode: 20 hidden neurons, 5x scale, 3 samples
python spinnaker/run_on_spinnaker.py --weight-scale 5 --max-hidden 20 --num-samples 3
```

### --tau-syn FLOAT (default: 1.0)

sets tau_syn_E and tau_syn_I for IF_curr_exp. lower values inject current faster,
closer to snnTorch's instantaneous delivery.

```bash
# Near-instantaneous synaptic current (closest to snnTorch)
python spinnaker/run_on_spinnaker.py --tau-syn 0.1

# Combined: faster current AND weight scaling
python spinnaker/run_on_spinnaker.py --tau-syn 0.1 --weight-scale 3
```

only applies to `IF_curr_exp`. has no effect with `--neuron-model IF_curr_delta`.

### --neuron-model {IF_curr_exp,IF_curr_delta} (default: IF_curr_exp)

`IF_curr_delta` has no synaptic current decay. each presynaptic spike gets
delivered as a direct voltage step: `V += weight`. this is the closest
sPyNNaker approximation to snnTorch's `snn.Leaky` neuron.

```bash
# Use delta neuron (most like snnTorch)
python spinnaker/run_on_spinnaker.py --neuron-model IF_curr_delta

# Delta neuron with weight scaling
python spinnaker/run_on_spinnaker.py --neuron-model IF_curr_delta --weight-scale 5
```

### --max-hidden INT (default: 256)

restricts inference to only the first N hidden neurons (out of 256). this
dramatically reduces the number of connections transferred over UDP, which
fixes the `OSError: No buffer space available` error from Run 3.

```bash
# Use 20 hidden neurons (safe: 20*2304 = 46K connections)
python spinnaker/run_on_spinnaker.py --max-hidden 20

# Use 64 hidden neurons (moderate: 64*2304 = 148K connections)
python spinnaker/run_on_spinnaker.py --max-hidden 64
```

accuracy will obviously be worse with fewer hidden neurons since the
network was trained with all 256.

### --prune-threshold FLOAT (default: 0.05)

removes connections with |weight| below this value before sending to SpiNNaker.
higher values reduce connection count (and UDP load) but hurt accuracy.

```bash
# Aggressive pruning: keep only connections with |w| > 0.2
python spinnaker/run_on_spinnaker.py --prune-threshold 0.2

# Combine pruning and hidden neuron reduction
python spinnaker/run_on_spinnaker.py --prune-threshold 0.1 --max-hidden 64
```

---

## combining flags: recommended debugging sequence

start minimal and expand:

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

## querying the provenance SQLite3 database

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

the provenance database usually ends up in:
- `~/spynnaker_output/<timestamp>/provenance_data/run.sqlite3`
- or a subdirectory of your current working directory named `spynnaker_output/`

if the database isn't there, check that `SPYNNAKER_DISABLE_PROVENANCE=0` is
set in the environment (run_on_spinnaker.py sets this automatically via `os.environ`).

---

## common errors and what to do

### OSError: [Errno 55] No buffer space available

too many connections being transferred at once over UDP to SpiNNaker.

if this doesn't work, try these (in order):
1. `--max-hidden 20` (reduces to 46K connections)
2. `--prune-threshold 0.1` or `--prune-threshold 0.2`
3. both combined: `--max-hidden 64 --prune-threshold 0.1`

### WARNING: Danger of SpikeSourceArray sending too many spikes at the same time

the input layer is sending hundreds of spikes in the same timestep.
this is expected given our input encoding (up to ~1400 simultaneous spikes).
it's a warning, not an error. the simulation will still run, but you might get
timing inaccuracies if the router gets saturated.

this is inherent to our rate-coded input representation. if accuracy suffers,
you could try spreading spikes across more timesteps in extract_features.py
using a sparser encoding, but honestly i haven't needed to do that yet.

### SpinnmanTimeoutException / board not responding

1. check your VPN connection to Manchester uni network
2. `ping spinnaker.cs.man.ac.uk`
3. wait a few minutes and retry (the board might be in use by someone else)
4. check the spalloc job queue

### 0 hidden spikes despite input spikes being present

run the debug scripts in order: debug_01 -> debug_02 -> debug_04 -> debug_05.
the decision tree above covers all the cases i've run into.
