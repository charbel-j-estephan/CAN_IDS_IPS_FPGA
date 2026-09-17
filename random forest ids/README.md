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

### Adaptive attacker, on real traffic

The real DoS capture contains only the crude attack, so it cannot answer what
happens against an attacker who floods a legitimate ID. `src/attack_on_real.py`
answers it with the files already here: it takes HCRL's attack-free
`normal_run_data.txt`, 988 871 frames of genuine vehicle traffic, and injects a
flood on top. The background traffic, its jitter, its payload behaviour and its
ID mix are all real; only the attack is synthetic, and the attack is the part
whose parameters are documented (0.3 ms period, 3 to 5 s bursts, ~16 % of
frames).

Three modes, flooding the busiest legitimate ID `0x2c0`:

| mode | best accuracy | nodes | FP | FN |
|---|---|---|---|---|
| `zero-id` (reproduces the real DoS attack) | 100.0000 % | 1 | 0 | 0 |
| `valid-id` (legit ID, all-zero payload) | 99.9996 % | 5 | 0 | 2 |
| `stealth` (legit ID, replayed real payloads) | 99.5386 % | 7 | 2421 | 5 |

On the stealth case every feature set converges to the same 99.5386 %, and the
reason is worth stating precisely. With 3 trees and 6 comparators:

* all 2421 false positives are on `0x2c0`, none on any other ID
* all 5 false negatives are on `0x2c0`
* the other 26 IDs are classified perfectly

The entire residual error is the victim's own frames, 2421 of its 22 160.
During a flood those frames arrive at the flood period like everything else
carrying that ID, so they are genuinely ambiguous per frame. That is a property
of the problem, not of the model, and no amount of extra trees moves it.

### The 100 % model does not survive an adaptive attacker

This is the most important measurement here. Taking the model trained on the
real DoS capture, the one that scores 100 %, and running it unchanged against
the other attacks:

| trace | accuracy | recall | FP | FN |
|---|---|---|---|---|
| real DoS capture (its own data) | 100.0000 % | 100.0000 % | 0 | 0 |
| `zero-id` flood, different drive | 95.7927 % | 100.0000 % | 22 120 | 0 |
| `valid-id` flood | 95.7918 % | 99.9946 % | 22 120 | 5 |
| **`stealth` flood** | 78.0178 % | **0.0000 %** | 22 120 | 93 453 |

**Recall zero.** It misses every single injected frame.

The forest's own thresholds explain it. Two of its seven trees root on
`pl_popcount <= 0`, meaning "the payload is all zeros". Another roots on
`dt_ratio_q6 <= 0`, and a ratio of exactly zero happens only when an ID has no
baseline entry at all. So the model that scores 100 % learned "this ID is not
in the whitelist, and its payload is blank". A flood that reuses a valid ID with
plausible payloads produces ratio 1 and a nonzero payload, and walks straight
through.

That is what a 100 % number on this dataset is worth. It is reported here as a
measurement rather than a caveat because it was easy to state as a caveat and
harder, and more useful, to demonstrate.

### The model to actually deploy

Train on the hardest case, not the easiest. The model selected on the stealth
trace is **3 trees, 14 comparator nodes, depth 3**, and it covers everything:

| trace | accuracy | recall | FP | FN |
|---|---|---|---|---|
| `zero-id` flood | 100.0000 % | 100.0000 % | 0 | 0 |
| `valid-id` flood | 99.5386 % | 99.9946 % | 2421 | 5 |
| `stealth` flood | 99.5386 % | 99.9946 % | 2421 | 5 |
| real DoS capture, **different drive** | 99.4085 % | 93.4007 % | **0** | 9758 |

Nine fewer comparators than the 100 % model, and it is the only one of the two
that works against an attacker who has read the paper.

The last row is a separate finding: the baseline ROM learned from one drive,
applied to a capture recorded eleven days earlier, produced **zero false alarms**
across 3.07 million normal frames. The per-ID timing baseline transfers across
drives cleanly. Recall drops to 93.4 % there because the model was trained
against a flood of a known ID and the DoS capture floods an unknown one, which
argues for training on a mix of attack styles rather than one.

## Reading the trees

`src/show_trees.py` prints any frozen forest as trees you can follow by eye,
with each threshold translated into physical units:

```bash
python3 src/show_trees.py --model results/realatk_stealth/model.json
```

Where the same forest lives, in three forms:

| file | form | for |
|---|---|---|
| `results/<tag>/model.json` | integer node tables | the Verilog generator and the reference predictor |
| `rtl/<tag>/rf_forest.v` | one wire per comparator, nested muxes | synthesis |
| `src/show_trees.py` output | indented tree with units | reading |

### How a verdict is reached

Every tree sees the same features and votes ATTACK or normal. The frame is
flagged when a majority of trees vote ATTACK. Tree counts are always odd so the
vote cannot tie, the same constraint the reference hardware design imposes.

In hardware the trees are not walked one node at a time. Each internal node
becomes one magnitude comparator, all of them evaluate simultaneously, and the
path collapses through a mux tree, so the whole forest resolves in a single
clock regardless of depth. In `rf_forest.v` that reads as one `wire cN_M` per
comparator and one nested conditional per tree.

### The recommended forest in full

3 trees, 8 comparators, majority of 2. `id_rate` is a sustained-rate estimate
where 64 means the ID is running at its normal rate; `dt_ratio_q6` is the
current gap as a fraction of that ID's normal gap, where 64 means exactly on
schedule.

```
TREE 0  (4 comparators)
  id_rate <= 288                    running at most 4.5x normal rate?
    yes: dt_ratio_q6 <= 12          arrived under 20 % into its normal gap?
           yes: ATTACK
           no:  normal
    no:  burst <= 0                 no back-to-back frames of this ID?
           yes: dt_id_dev <= 8289   gap within 8.29 ms of normal?
                  yes: normal
                  no:  ATTACK
           no:  ATTACK

TREE 1  (2 comparators)
  pl_popcount <= 2                  payload almost entirely zero bits?
    yes: dt_id <= 1310              arrived within 1.31 ms of the last one?
           yes: ATTACK
           no:  normal
    no:  normal

TREE 2  (2 comparators)
  pl_popcount <= 2                  payload almost entirely zero bits?
    yes: dt_ratio_q6 <= 9           arrived under 16 % into its normal gap?
           yes: ATTACK
           no:  normal
    no:  normal
```

Read as one sentence: trees 1 and 2 catch a degenerate payload arriving far too
early, and tree 0 catches a flood on timing and rate alone without looking at
the payload at all. Tree 0 is why the forest still works when the attacker
replays valid payloads, because it is the one that does not depend on them.

### Pruning, and an honest note on what it bought

Rendering the trees exposed comparators whose two branches led to the same
verdict. sklearn creates them when a split reduces impurity but both leaves end
up the same majority class. `prune_equivalent` in `select_model.py` collapses
any node whose whole subtree agrees, which is behaviour preserving by
construction and asserted against the unpruned predictions:

| model | comparators before | after |
|---|---|---|
| recommended | 14 | **8** |
| real DoS | 23 | **19** |

Accuracy, false positives and false negatives are all unchanged to the frame.

What it did **not** buy: the synthesized forest is byte identical before and
after, 33 cells and 25 LUTs either way, because yosys and ABC were already
eliminating that logic. So for the fully unrolled combinational forest here the
gain is in model size and readability, and in the node count finally being an
honest number rather than one inflated by up to 43 %.

It would be a real area saving for a memory-central implementation, where the
node table lives in BRAM and is walked one node per cycle. There every node
costs storage and a cycle, so 14 nodes against 8 is a genuine difference. Worth
knowing if the design is ever moved to that style to save LUTs.

## Hardware

```
feature extraction   3 cycles   (latch, per-ID fetch, compute)
forest + voter       1 cycle    (combinational, registered output)
total                4 cycles   = 40 ns at 100 MHz
budget               1 ms per frame
margin               25 000x
```

Latency is fixed, not data dependent: the forest is unrolled combinationally,
so there is no worst-case path to argue about. The detector ceiling is 25 M
frames/s against a CAN 1 Mbit/s ceiling of about 21 k frames/s, so the bus is
the bottleneck by three orders of magnitude.

### Area, measured

Synthesised with yosys 0.33, `synth_xilinx -family xc7`, on the recommended
3-tree model. Parse any run with `src/syn_report.py`:

| module | LUT | FF | RAMB36 | RAMB18 | DSP48 | CARRY4 |
|---|---|---|---|---|---|---|
| `can_ids_features` | 785 | 320 | 11 | 11 | 2 | 58 |
| `rf_forest` | 25 | 2 | 0 | 0 | 0 | 2 |
| **total** | **810** | **322** | **11** | **11** | **2** | **60** |

33 BRAM18 equivalents, plus 8 distributed `RAM256X1S`.

**The forest is 3.1 % of the LUTs.** The trees are nearly free. The feature
extractor and its per-ID memories are the entire design, which is the strongest
form of the argument for keeping the forest small: a bigger forest buys almost
nothing in accuracy and costs almost nothing in area, so the interesting
engineering is all in the features.

Two earlier estimates in this file were wrong and are corrected above: the LUT
count was understated by about 1.9x, and the reciprocal multiply was given as
one DSP48 when the design uses two, the second being the rate bucket.

Synthesise against a real architecture. Generic `synth` has no block-RAM
primitive, so it maps the 2048-entry per-ID arrays to flip-flops and reports
458 128 cells with 221 623 registers. That measures the target, not the design.

### The obvious next optimisation

Memory is dominated by the 2048-entry direct-mapped ID table, 562 kbit of which
the real captures use 27 entries. An ID-to-slot index ROM (2048 x 5 bit) feeding
a 32-entry table would cut that to roughly 2 BRAM18, a 16x reduction, at the
cost of one more pipeline cycle. Not done here: 33 BRAM18 already fits
comfortably on a mid-range part (an Artix-7 35T has 100), and the change would
need the RTL re-verified. Worth doing for a small or cost-sensitive target.

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

Last run, on both frozen models:

```
real DoS model, 7 trees / 23 comparators
  tb_rf_forest     300000 real test vectors, 0 mismatches   PASS
  tb_can_ids_top   200000 real CAN frames,   0 mismatches   PASS

recommended model, 3 trees / 14 comparators   (rtl/recommended/)
  tb_rf_forest     300000 test vectors,      0 mismatches   PASS
  tb_can_ids_top   200000 real CAN frames,   0 mismatches   PASS

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
src/show_trees.py      print a frozen forest as readable trees
src/syn_report.py      parse yosys statistics into a resource table
src/cross_eval.py      score one frozen model against other attack styles
src/attack_on_real.py  inject a flood into HCRL's real attack-free capture
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
