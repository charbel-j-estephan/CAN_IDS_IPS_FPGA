# Random forest CAN IDS for FPGA

Per-frame intrusion detection on a CAN bus, sized to run on an FPGA inside a
1 ms budget. The flow goes from a raw HCRL car-hacking CSV to synthesizable
Verilog that has been simulated against the Python model frame by frame.

## The dataset

The real HCRL `DoS_dataset.csv` is in `data/DoS_real.csv`, verified against the
published figures:

| | expected | loaded |
|---|---|---|
| frames | 3 665 771 | 3 665 771 |
| injected | 587 521 | 587 521 |
| normal | 3 078 250 | 3 078 250 |
| injected share | 16.03 % | 16.03 % |
| duration | — | 2832.7 s |
| unique IDs | — | 27 |

The flood uses CAN ID `0x000` with an all-zero payload, as documented. Two
format details the loader handles that are easy to get wrong: the file has CRLF
line endings, and rows with DLC below 8 are shorter, so the `R`/`T` flag does
not sit in a fixed column.

`data/normal_run_data.txt` is HCRL's separate attack-free capture, 988 871
frames in a different whitespace-separated layout, read by
`load_hcrl_normal_txt`.

Generated traces from `src/make_trace.py` are kept as `data/synth_*.csv`. They
are used for the adaptive-attacker analysis below, which the real DoS capture
cannot exercise because it only contains the one crude attack.

To rerun everything on the real file:

```bash
./run_all.sh data/DoS_real.csv real
```

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

Scored with the hardware majority voter, on the held-out test truncation.

### Real HCRL DoS capture

Train truncation 1 649 597 frames at 22.19 % injected, test truncation
1 649 597 frames at 8.96 % injected. The two halves have very different attack
densities, which is a harder test than a balanced split.

| feature set | best accuracy | nodes at best |
|---|---|---|
| `timing` | **100.0000 %** | 2 |
| `timing_only` | 100.0000 % | 2 |
| `no_rate` | 100.0000 % | 2 |
| `paper` (reference 2 features) | 98.3037 % | 208 |

**Two comparators reach 100 %**, on `dt_ratio_q6` and `id_rate`. That is a
property of this attack, not a strong model: ID `0x000` never appears in clean
traffic, so it has no baseline entry, and every feature derived from that table
gives it away. A two-comparator whitelist is a legitimate and extremely cheap
DoS defence, but it is not a random forest and it stops working the moment a
flood reuses a valid ID.

The reference paper's two features reach **98.3037 %** here, against the 98.2 %
the paper reports on the same dataset. That agreement is the check that this
pipeline measures what the paper measured.

### Selected model

7 trees, 23 comparator nodes, depth 3, splitting on 8 of 12 features:

```
accuracy   100.0000 %
recall     100.0000 %   (0 missed attacks)
precision  100.0000 %   (0 false alarms)
```

on 1 649 597 held-out frames. Splits: `dt_ratio_q6` 6, `id_rate` 5,
`pl_popcount` 3, `dt_id` 3, `burst` 2, `dt_id_dev` 2, `dt_bus` 1, `hd` 1.

The smallest 100 % model is 2 comparators. The selector does not pick it:
`--policy robust` spreads the verdict across features so no single one carries
it, and `--max-nodes` keeps that in budget. 23 comparators instead of 2 is a
rounding error in area.

### Adaptive attacker, on generated traces

The real DoS capture contains only the crude `0x000` flood, so it cannot answer
what happens against an attacker who floods a legitimate ID with plausible
payloads. `src/make_trace.py --stealth` builds that case, which blinds `hd`,
`pl_popcount` and `pl_violation` and leaves only timing:

| trees | nodes | accuracy | recall | FP | FN |
|---|---|---|---|---|---|
| 1 | 3 | 99.5835 % | 99.9828 % | 5785 | 31 |
| 3 | 9 | 99.5851 % | 99.9823 % | 5761 | 32 |
| 1 | 12 | 99.7479 % | 99.2804 % | 2222 | 1298 |
| 3 | 20 | 99.6057 % | 99.9834 % | 5476 | 30 |
| 9 | 277 | 99.7660 % | — | 2309 | 958 |

Without the rate buckets the ceiling is 99.5311 %; with them, 99.7660 %. More
usefully, **3 comparators with the rate feature beat 156 comparators without
it.** Going from 3 nodes to 277 buys 0.18 points, so the ceiling is set by the
features, not by forest size. This is the argument for small trees.

Accuracy is also the wrong thing to rank on here. The 12-node row scores higher
than the 20-node row while missing 1298 attacks instead of 30, so the model for
this case is selected on recall.

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

Last run, on the real-data model (7 trees, 23 comparators):

```
tb_rf_forest     checked 300000 real test vectors, 0 mismatches   PASS
tb_can_ids_top   checked 200000 real CAN frames,   0 mismatches   PASS
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
src/make_trace.py      HCRL-format trace generator, for attacks the real
                       DoS capture does not contain
src/can_data.py        loaders for the real HCRL attack CSVs and for the
                       attack-free normal_run_data.txt
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
