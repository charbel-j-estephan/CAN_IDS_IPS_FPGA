# Random forest CAN IDS for FPGA

Per-frame intrusion detection on a CAN bus, sized to run on an FPGA inside a
1 ms budget. The flow goes from a raw HCRL car-hacking CSV to synthesizable
Verilog that has been simulated against the Python model frame by frame.

## Status of the dataset

The real HCRL `DoS_dataset.csv` is **not** in this repository and could not be
downloaded from the build sandbox: `ocslab.hksecurity.net` and Kaggle are both
blocked by the network policy, and only GitHub hosts are reachable. Everything
here was therefore developed and measured against `src/make_trace.py`, which
generates traces in the exact HCRL CSV layout and reproduces the documented
properties of the real DoS capture:

* Hyundai YF Sonata style ID set, 27 periodic IDs from 10 ms to 1000 ms
* CAN ID `0x000` flooded every 0.3 ms in bursts of 3 to 5 seconds
* about 16 % of frames injected, ~2000 frames/s, ~3.1 M frames over 1540 s
* a 500 kbit/s arbitration model, so the flood delays the legitimate frames
  around it and those delayed frames stay labelled normal, as in the real capture
* payload behaviour split across counter, static and quantised-sensor IDs, so
  roughly 70 % of consecutive normal frames repeat their payload byte for byte

To run on the real file, drop it in and run one command. Nothing else changes:

```bash
./run_all.sh /path/to/DoS_dataset.csv real
```

The numbers below will be regenerated for the real data. Treat them as a
calibration of the method and the hardware cost, not as results on HCRL.

## How the data is split

Two disjoint contiguous truncations with a discarded gap between them:

```
frames  [0 %, 45 %)   train
        [45 %, 55 %)  discarded
        [55 %, 100 %) test
```

Contiguous, not random. Every feature is sequential, so a random row split would
put the immediate neighbours of each test frame into training. The gap stops the
per-ID state carrying across the boundary. The per-ID baseline is fitted on
clean frames from the training truncation only.

## Features

Twelve integers, all computable in one pass from state the FPGA holds in one
BRAM indexed by the 11-bit CAN ID.

| # | feature | what it is |
|---|---|---|
| 0 | `dt_id` | microseconds since the previous frame of this ID |
| 1 | `dt_id_dev` | `abs(dt_id - mean_interval)` for this ID |
| 2 | `dt_ratio_q6` | `dt_id / mean_interval` in 1/64 steps, 64 means exactly on period |
| 3 | `hd` | Hamming distance to the previous payload of this ID |
| 4 | `hd_dev` | `abs(hd - mean_hamming)` for this ID |
| 5 | `dt_bus` | microseconds since the previous frame on the bus |
| 6 | `burst` | run length of consecutive frames sharing this ID |
| 7 | `id_known` | 1 if the ID appeared in clean training traffic |
| 8 | `can_id` | the raw 11-bit identifier |
| 9 | `dlc` | data length code |
| 10 | `pl_violation` | payload bits that break this ID's learned invariant |
| 11 | `pl_popcount` | population count of the raw payload |

Three of these are worth explaining.

**`dt_ratio_q6`** is the scale-invariant version of `dt_id_dev`. A single
absolute-deviation threshold cannot serve a 10 ms ID and a 1000 ms ID at once,
which is what limits the two-feature reference design. Dividing by the period
fixes that, and it costs one multiply by a stored reciprocal rather than a
divider: one DSP slice, one cycle.

**`pl_violation`** records, per ID, which payload bits never move in clean
traffic and what value they hold, then counts how many of those bits a frame
flips. A frame that flips one is structurally impossible for that ID. One 64-bit
XOR, one AND and a popcount. A bit only counts as invariant with at least 500
clean samples behind it, otherwise a short training window invents invariants
that do not hold and the false positive rate goes up.

**Feature sets.** `full` includes `can_id` and `id_known`. `timing` drops both,
so the model carries no identity information and still works when a flood reuses
a legitimate ID. `timing_only` additionally drops the payload-content features.
`paper` is the reference design's two features, kept as a control.

## Results

Measured on the held-out test truncation. `nodes` counts internal comparator
nodes summed over the forest, which is what the FPGA pays for in area.

### The classic DoS case: flooding CAN ID `0x000`

**One comparator reaches 100 %.** ID `0x000` never appears in clean traffic, so
it has no baseline entry, and every feature derived from that table gives it
away. This is a property of the attack, not a strong model: the real HCRL DoS
capture floods the same ID, so the same thing will happen there. A single
whitelist comparator is a legitimate and very cheap DoS defence, but it is not a
random forest and it stops working the moment the flooded ID is a valid one.

For contrast, the reference paper's two features on the same data reach
**98.87 %** with 26 nodes, against the 98.2 % the paper reports. That agreement
is the sanity check that this pipeline is measuring the same thing the paper did.

### Flooding a legitimate ID (`0x316`)

This removes the whitelist shortcut, so it is the case that actually exercises
the forest.

| feature set | best accuracy | nodes at best |
|---|---|---|
| `timing` | **100.0000 %** | 1 |
| `timing_only` | 99.4629 % | 129 |
| `paper` | 99.2313 % | 111 |

Without payload-content features the model plateaus at 99.46 % with 129 nodes.
The failure mode is specific and worth stating: 4686 of 5553 false positives
were the victim ID's own legitimate frames. Once an attacker floods your ID,
your real frames' inter-arrival times are corrupted too, so the injected frames
and the victim's frames become indistinguishable on every timing feature. Adding
`pl_violation` removes that failure entirely.

**Selected model** (`--policy robust`, target 99.99 %):

```
3 trees, 12 comparator nodes, depth 5, splits on 9 features
accuracy   99.999928 %
precision  100.000000 %   (0 false alarms in 1 396 316 frames)
recall      99.999446 %   (1 missed attack)
```

The smallest model that also scores 100 % is a single comparator on
`pl_violation`. The selector does not choose it. One feature carrying the whole
verdict means an attacker who defeats that one feature defeats the IDS outright,
and a single tree has no vote to lose. Twelve comparators instead of one is a
rounding error in area and buys graceful degradation. Use `--policy smallest` to
override.

### Worst case: a flood that replays structurally valid payloads

`make_trace.py --stealth` injects payloads that are valid for the flooded ID, so
`hd`, `pl_popcount` and `pl_violation` all go blind and only timing is left.

| nodes | accuracy |
|---|---|
| 3 | 99.2651 % |
| 15 | 99.3452 % |
| 29 | 99.4665 % |
| 1147 | 99.5641 % |

**This is the argument for small trees.** Going from 3 comparators to 1147 buys
0.30 percentage points. The accuracy ceiling against an adaptive attacker is set
by the features, not by forest size, so spending area on more trees is close to
worthless. Everything useful happens below about 30 nodes.

## Hardware

```
feature extraction   3 cycles   (latch, per-ID fetch, compute)
forest + voter       1 cycle    (combinational, registered output)
total                4 cycles   = 40 ns at 100 MHz
budget               1 ms per frame
margin               25 000x
```

Latency is fixed, not data dependent: the forest is unrolled combinationally, so
there is no worst-case path to argue about. The detector ceiling is 25 M
frames/s against a CAN 1 Mbit/s ceiling of about 21 k frames/s, so the bus is the
bottleneck by three orders of magnitude.

Memory is 2048 IDs x 97 bits of per-ID state plus 2048 x 172 bits of baseline
ROM, about 67 kB, a handful of 18 kbit BRAMs. Real vehicles use around 30 IDs,
so a smaller direct-mapped table or a CAM shrinks this a lot if needed.

The tree count is kept odd throughout so the majority voter cannot tie, which is
the same constraint the reference hardware design imposes.

## Verification

Two testbenches, both comparing against the Python pipeline frame by frame:

* `tb_rf_forest.v` replays the whole test truncation through the generated
  forest and checks every verdict.
* `tb_can_ids_top.v` feeds raw CAN frames into `can_ids_top` and checks the
  verdicts, so it covers the feature extractor, the per-ID state RAM, the
  baseline ROMs and the forest together.

The trained thresholds convert to integers with zero loss, because every feature
is an integer and `x <= 2.5` is exactly `x <= 2` for integer `x`.
`select_model.py` asserts that the integer tables reproduce sklearn exactly on
both truncations rather than assuming it.

## Layout

```
src/make_trace.py      HCRL-format trace generator (stand-in for the real file)
src/can_data.py        loader for the real HCRL CSV, handles short-DLC rows
src/features.py        streaming integer feature extraction
src/prepare.py         truncation split, baseline fit, feature cache
src/sweep.py           forest geometry sweep
src/select_model.py    pick and freeze a model, assert integer exactness
src/export_rom.py      per-ID baseline ROM contents
src/export_verilog.py  generate the forest RTL and both testbenches
src/hw_report.py       area, latency and throughput budget
src/report.py          markdown report of every sweep
rtl/can_ids_features.v feature extractor
rtl/can_ids_top.v      top level
run_all.sh             CSV in, verified Verilog out
```

## Known limitations

* The accuracy numbers come from generated traces, not from HCRL. Rerun on the
  real file before quoting any of them.
* `pl_violation` assumes the training window contains enough clean traffic to
  observe each ID's real variability. A vehicle state absent from training (a
  gear or mode never engaged) can flip a bit thought to be invariant and cause
  false positives. The 500-sample floor mitigates this but does not remove it.
* The arbitration model in the generator is a queue with priority, not a
  bit-level CAN simulation. It is good enough to reproduce flood-induced delay,
  not to study arbitration corner cases.
* Only DoS-style flooding is modelled. Fuzzy and spoofing captures from the same
  dataset family will load and run through this flow unchanged, but nothing here
  has been tuned for them.
