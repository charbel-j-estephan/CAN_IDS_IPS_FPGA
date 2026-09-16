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
| 12 | `id_rate` | leaky-bucket rate estimate for this ID |
| 13 | `bus_rate` | the same estimate for the bus as a whole |

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

**`id_rate`** is an exponentially weighted rate estimate. Each frame of an ID
adds a fixed credit and leaks a fraction of the bucket's own contents scaled by
elapsed time, giving the fixed point `b = 4096 / dt_ratio_q6`. An ID running at
its nominal period settles at 64; one flooded 64x over rate saturates. It needs
one small multiply, one subtract, one add and 12 bits of state per ID.

This one matters more than it looks. During a sustained flood of one ID, every
frame carrying that ID arrives at the flood period, injected or not, so `dt_id`
separates nothing at all for the victim ID. A bucket integrates over many frames
and still reports that the stream is running far over rate when no single
inter-arrival time does. Note the leak has to be proportional to the bucket's own
contents: a fixed drain has no restoring force and random-walks to saturation on
perfectly normal traffic, which is exactly what the first version of this feature
did before it was measured.

**Feature sets.** `full` includes `can_id` and `id_known`. `timing` drops both,
so the model carries no identity information and still works when a flood reuses
a legitimate ID. `timing_only` additionally drops the payload-content features,
and `no_rate` drops the two rate buckets, so the gain from each group is
measurable rather than asserted. `paper` is the reference design's two features,
kept as a control.

## Scoring matches the hardware voter

`RandomForestClassifier.predict` averages per-tree class *probabilities* and then
takes an argmax. The voter in the RTL cannot do that: it sees one bit per tree
and counts. The two agree on shallow, cleanly separated trees and diverge once
leaves are mixed.

Scoring the sweep with sklearn's own `predict` therefore reports an accuracy the
hardware does not achieve. Everything here is scored with a hard majority vote
instead. This was not caught by reading the code, it was caught by the assertion
in `select_model.py` that the exported integer tables must reproduce the trained
model exactly: on a 9-tree depth-8 forest it reported 1064 mismatches.

## Results

Measured on the held-out test truncation, scored with the hardware majority
voter. `nodes` counts internal comparator nodes summed over the forest.

### The classic DoS case: flooding CAN ID `0x000`

**One comparator reaches 100 %.** ID `0x000` never appears in clean traffic, so
it has no baseline entry, and every feature derived from that table gives it
away. This is a property of the attack, not a strong model: the real HCRL DoS
capture floods the same ID, so the same thing will happen there. A single
whitelist comparator is a legitimate and very cheap DoS defence, but it is not a
random forest and it stops working the moment the flooded ID is a valid one.

For contrast, the reference paper's two features on the same data reach
**99.01 %**, against the 98.2 % the paper reports. That agreement is the sanity
check that this pipeline measures the same thing the paper did.

### Flooding a legitimate ID

This removes the whitelist shortcut. The `timing` set reaches 100 % with a single
comparator on `pl_violation`, because the injected payloads are all zeros and so
break the victim ID's learned payload invariant on every frame.

That is real, but it depends on the attacker being lazy. Which leads to the case
that actually decides the design.

### Worst case: a flood replaying structurally valid payloads

`make_trace.py --stealth` injects payloads that are valid for the flooded ID, so
`hd`, `pl_popcount` and `pl_violation` all go blind and only timing is left.

| trees | nodes | accuracy | recall | FP | FN |
|---|---|---|---|---|---|
| 1 | 3 | 99.5835 % | 99.9828 % | 5785 | 31 |
| 3 | 9 | 99.5851 % | 99.9823 % | 5761 | 32 |
| 1 | 12 | 99.7479 % | 99.2804 % | 2222 | 1298 |
| 3 | 20 | 99.6057 % | 99.9834 % | 5476 | 30 |
| 9 | 277 | 99.7660 % | — | 2309 | 958 |

Two things to read off this.

**The rate buckets moved the ceiling.** Without them the best achievable is
99.5311 %; with them it is 99.7660 %. More to the point, **3 comparators with the
rate feature beat 156 comparators without it.**

**Accuracy is the wrong headline.** The 12-node row scores higher than the
20-node row but misses 1298 attacks instead of 30. For an IDS that feeds a node
exclusion system, a missed attack costs more than a false alarm, so the shipped
model is picked on recall, not accuracy.

### Selected model

3 trees, 20 comparator nodes, depth 3, splitting on 9 features:

```
accuracy   99.6057 %
recall     99.9834 %   (30 missed attacks out of 180 373)
precision  97.0530 %   (5476 false alarms out of 1 215 943 normal frames)
```

`--max-nodes` caps the search, because the robust policy maximises feature
spread and will otherwise happily return a 187-node forest for 0.13 more points.

### Does one burned model cover every attack style?

The FPGA gets one forest and one set of baseline ROM contents. If the attacker
switches tactics, that same burned model has to cope. `cross_eval.py` scores the
model trained on the stealth trace against all three, using the stealth baseline
throughout, because the baseline is ROM:

| trace | accuracy | recall | FP | FN |
|---|---|---|---|---|
| `0x000` flood | 99.9981 % | 100.0000 % | 27 | 0 |
| valid-ID flood | 99.6057 % | 99.9834 % | 5476 | 30 |
| stealth flood | 99.6057 % | 99.9834 % | 5476 | 30 |

Training on the hardest case covers the easier ones. The bottom two rows are
identical, which is not a copy-paste: the two traces differ on exactly the
180 373 injected frames, and `pl_popcount` differs on every one of them, yet the
verdicts come out bit-identical. The model's decision rides entirely on timing
and rate, so it is not leaning on the attacker using degenerate payloads. That
was verified element by element, not inferred from the totals matching.

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

Last run, on the selected 3-tree 12-comparator model:

```
tb_rf_forest     checked 300000 vectors, 0 mismatches   PASS
tb_can_ids_top   checked 200000 frames,  0 mismatches   PASS
integer tables vs sklearn trees: 0 mismatches on train, 0 on test
```

The end-to-end testbench matters most now that the feature extractor holds a
stateful leaky-bucket recurrence per ID. A rate bucket that drifts by one count
between the model and the RTL would silently change verdicts, so it is checked
frame by frame rather than spot-checked.

`run_all.sh` refuses to continue if either truncation ends up with no attack
frames or no normal frames. A short capture whose attack bursts all land in one
half will otherwise train a model that predicts "normal" for everything and
still reports a plausible-looking accuracy.

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
src/cross_eval.py      score one frozen model against other attack styles
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
* The residual false alarms are almost entirely the flooded ID's own legitimate
  frames. During a sustained flood those frames are genuinely ambiguous at the
  per-frame level, since they arrive at the flood period like everything else
  carrying that ID. Driving that number down further needs a verdict at the
  (ID, time window) level rather than per frame, which is what a node exclusion
  system wants anyway.
