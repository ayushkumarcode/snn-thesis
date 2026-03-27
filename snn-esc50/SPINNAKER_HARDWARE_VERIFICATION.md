# spinnaker hardware verification

purpose: prove that our SNN inference ran on real SpiNNaker neuromorphic hardware, not a simulator.

---

## conclusion: confirmed -- real SpiNNaker hardware

the evidence is pretty overwhelming from multiple independent sources. we ran on a real SpiNNaker SpiNN-5 board (48-chip, Version 5) allocated from the million-core machine at UoM's Department of Computer Science, accessed remotely via spalloc.

---

## evidence 1: DNS and network infrastructure

`spinnaker.cs.man.ac.uk` resolves to `130.88.193.57`.

WHOIS confirms:
```
inetnum:      130.88.0.0 - 130.88.255.255
netname:      MANLAN
descr:        The University of Manchester Local Area Network
org-name:     The University of Manchester
address:      Kilburn Building, Oxford Road, M13 9PL, Manchester
```

the Kilburn Building is where CS (and the SpiNNaker machine) is physically housed.

## evidence 2: official docs confirm this server

the official sPyNNaker install guide gives this exact config for the Manchester million-core machine:

```ini
[Machine]
spalloc_server = spinnaker.cs.man.ac.uk
spalloc_user = user.name@email.address
```

our config (`~/.spynnaker.cfg`):
```ini
[Machine]
spalloc_server = spinnaker.cs.man.ac.uk
spalloc_user = r36859ak@manchester.ac.uk
```

## Evidence 3: Hardware Fingerprint -- SC&MP Firmware Version

From our run logs:
```
Found board with version [Version: SC&MP 4.0.0 at SpiNNaker:0:0:0 (built Fri Nov 17 09:35:48 2023)]
```

**SC&MP** = "Spinnaker Control & Monitor Program" -- the firmware running on the monitor core of each SpiNNaker chip. Confirmed by [source code on GitHub](https://github.com/SpiNNakerManchester/spinnaker_tools/blob/master/scamp/scamp-3.c).

This version is obtained via SCP (SpiNNaker Command Protocol) -- a real hardware-level handshake. The virtual transceiver (`Virtual5Transceiver`) returns a hardcoded fake version `SC&MP/SpiNNaker 3.4.2`. Our logs show `SC&MP 4.0.0`, which was read from real hardware.

## Evidence 4: Machine Specifications Match Real SpiNNaker1

From our provenance database (`reports/.../global_provenance.sqlite3`):
```
Detected Machine on 10.11.219.113 with 47 Chips, 836 cores and 114.0 links.
Chips have sdram of 123469792 bytes, router table of size 1023,
between 17 and 18 cores and between 3 and 6 links.
```

| Spec | Detected | SpiNN-5 Board Spec | Match? |
|------|----------|--------------------|--------|
| Chips per board | 47 (one faulty) | 48 | Yes (faults normal) |
| Cores per chip | 17-18 | 18 ARM968E-S (1 monitor) | Yes |
| Total cores | 836 | ~864 max | Yes |
| SDRAM per chip | 123,469,792 bytes (~118 MB) | 128 MB (minus overhead) | Yes |
| Router table | 1,023 entries | 1,024 (1 reserved) | Yes |
| Links per chip | 3-6 | Up to 6 (fewer at edges) | Yes |

The imperfect chip count (47 not 48) is actually **stronger** evidence of real hardware -- a simulator always returns perfect numbers.

## Evidence 5: Spalloc Job Allocation Lifecycle

```
2026-03-03 13:13:03 INFO: Requesting job with 1 boards
Created spalloc job 99729
Job has been queued by the spalloc server.
Waiting for board power commands to complete.     ← real power relays switching
2026-03-03 13:14:12 INFO: Creating transceiver for 10.11.219.177
Attempting to boot machine
Found board with version [Version: SC&MP 4.0.0 ...]
Machine communication successful
```

Different runs allocated different physical boards:
- Run at 13:01: Board `10.11.219.97` (48 chips, 855 cores)
- Run at 13:14: Board `10.11.219.177` (47 chips, 836 cores)
- Run at 13:16: Board `10.11.219.113` (47 chips, 836 cores)

This proves the spalloc server is partitioning a real multi-board machine and assigning different available boards to different jobs.

## Evidence 6: On-Chip Firmware IOBUF Logs

Emergency IOBUF files in `reports/.../provenance_data/system_provenance_data/` contain output from C firmware running on individual ARM968 cores:

```
[INFO] (data_speed_up_packet_gatherer.c: 938): Configuring packet gatherer
[INFO] (data_specification.c: 111): magic = ad130ad6, version = 1.0
```

File naming: `emergency_iobuf_X_Y_P.txt` (chip coordinates X,Y and processor P). Files exist across dozens of chip positions (0_0, 0_1, 1_0, etc.), confirming firmware ran on multiple physical chips.

## Evidence 7: Not Running in Virtual/Simulation Mode

| Check | Virtual Mode | Our Run | Verdict |
|-------|-------------|---------|---------|
| SC&MP version | Hardcoded `3.4.2` | `4.0.0` from real chip | Real |
| Transceiver type | `Virtual5Transceiver` | `Spalloc Old` (real) | Real |
| Board power | Skipped | "Waiting for board power commands" | Real |
| Setup time | Milliseconds | 27-37 seconds (network RTT) | Real |
| Chip count | Always perfect 48 | 47 or 48 (faulty chip) | Real |
| IOBUF files | None | Real ARM968 firmware logs | Real |

## Evidence 8: Timing Characteristics

From provenance timing data:
- **Mapping stage:** 28,124 ms -- placing neurons onto physical chip topology
- **Loading stage:** 25,589 ms -- transferring data over Ethernet to board
- **Running stage:** 1,521 ms -- 25 ms biological time on hardware
- **Simulation on-chip:** 25 ms (25 timesteps at 1 ms each)

The 25+ second loading time is consistent with real Ethernet data transfer to remote hardware.

## Evidence 9: Software Stack

All official SpiNNaker Manchester packages installed:
```
sPyNNaker          1!7.4.1
SpiNNMan           1!7.4.1
SpiNNMachine       1!7.4.1
PACMAN             1!7.4.1
SpiNNFrontEnd      1!7.4.1
spalloc            1!7.4.1
PyNN               0.12.4
```

Published by [SpiNNakerManchester](https://github.com/SpiNNakerManchester) on GitHub/PyPI.

---

## Summary Table

| Evidence | Finding | Confidence |
|----------|---------|------------|
| DNS/WHOIS | `130.88.193.57` registered to UoM, Kilburn Building | Definitive |
| Official docs | `spinnaker.cs.man.ac.uk` is the documented server | Definitive |
| SC&MP firmware | Version `4.0.0` read via SCP from real chip | Definitive |
| Hardware specs | 47-48 chips, 836-855 cores, 118 MB SDRAM match SpiNN-5 | Definitive |
| Spalloc jobs | Multiple board IPs allocated across runs | Strong |
| IOBUF firmware | ARM968 C logs from individual chip coordinates | Strong |
| Timing | 27-37s setup, 25s data loading = real network transfer | Strong |
| Imperfect hardware | 47/48 chips (faulty chip) impossible in simulation | Strong |

---

## References

- [SpiNNaker project (University of Manchester)](https://www.scieng.manchester.ac.uk/tomorrowlabs/spinnaker/)
- [SpiNNaker Wikipedia](https://en.wikipedia.org/wiki/SpiNNaker)
- [Spalloc Documentation](https://spalloc.readthedocs.io/en/stable/index.html)
- [PyNN on SpiNNaker Installation Guide](https://spinnakermanchester.github.io/spynnaker/8.0.0/PyNNOnSpinnakerInstall.html)
- [SpiNNaker Tools SCAMP Source](https://github.com/SpiNNakerManchester/spinnaker_tools/blob/master/scamp/scamp-3.c)
- [sPyNNaker Paper (Frontiers in Neuroscience)](https://www.frontiersin.org/articles/10.3389/fnins.2018.00816/full)
- [SpiNNTools Paper (Frontiers in Neuroscience)](https://www.frontiersin.org/articles/10.3389/fnins.2019.00231/full)
