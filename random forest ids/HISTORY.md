# Development history

> **This file is superseded and kept for the record.** It documents the
> trained-forest line of work, which the shipped detector replaced. Its
> conclusions about model size, feature sets and the alarm layer were correct
> for what had been measured at the time and were then overturned by later
> measurements: see README.md for what actually ships and the chain of
> failures that led there.
>
> Specifically, everything below describes a model trained on labelled attacks.
> The shipped detector uses no attack data at all, is three comparators rather
> than eight, and catches attacks this one was blind to. Numbers in this file
> are *not* current.

## The original write-up

Per-frame intrusion detection on a CAN bus. The flow goes from a raw HCRL
car-hacking CSV to a small, frozen random forest whose decision logic you can
read.

The models are deliberately tiny and every feature is an integer, so this
stays portable to a hardware implementation later. The Verilog that did that
was removed on request to keep the project focused on the models; see
[Recovering the RTL](#recovering-the-rtl) if it is wanted back.

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

### A second, independent dataset

`src/syncan_data.py` reads the **SynCAN** benchmark (ETAS GmbH / Robert Bosch),
whose `test_flooding` set is a DoS on a legitimate ID. It is a different vehicle
model, a different format and a different research group, so it is the closest
thing here to an out-of-sample test.

    Hanselmann, Strauss, Dormann, Ulmer, "CANet: An Unsupervised Intrusion
    Detection System for High Dimensional CAN Bus Data", IEEE Access 8, 2020.
    https://github.com/etas/SynCAN

Two differences change what it can measure. It gives **signal values, not
payload bytes**, so a payload is reconstructed by quantising each signal to 16
bits; prefer the `no_payload` feature set there and treat anything
payload-derived as describing the reconstruction. And it labels **every ID
inside an attacked window**, not only the injected frames, so its labels answer
"is the bus under attack now" rather than "was this frame injected".

Its licence permits academic and non-commercial research and requires citation
but **forbids redistributing the data**, so the CSVs stay out of this repo.
Fetch it yourself:

```bash
git clone --depth 1 https://github.com/etas/SynCAN /tmp/syncan
cd /tmp/syncan && unzip -q test_flooding.zip
python3 src/prepare.py --csv /tmp/syncan/test_flooding.csv --format syncan \
    --out results/syncan/cache.npz --baseline-out results/syncan/baseline.json
```

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

Fourteen integers, all computable in one streaming pass from a small amount of
state kept per CAN ID.

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

## Scoring uses a majority vote

`RandomForestClassifier.predict` averages per-tree class *probabilities* and then
takes an argmax. This project scores a hard majority vote instead: each tree
gives one bit and the bits are counted. The two agree on shallow, cleanly separated trees and diverge once
leaves are mixed.

The two agree on shallow, cleanly separated trees and diverge once leaves are
mixed, so the two scoring rules are not interchangeable and the reported numbers
have to match the rule the frozen model actually uses. This was not caught by reading the code, it was caught by the assertion
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

### Model selected on the real capture alone

Kept for the record, and **not** the one to deploy: see the overfitting section
above. 7 trees, 23 comparator nodes, depth 3, splitting on 8 of 12 features:

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

### SynCAN, an independent benchmark

Scored on SynCAN's own held-out slice, with its window labels:

| feature set | best accuracy | recall | comparators |
|---|---|---|---|
| timing and rate only | 97.11 % | 92.53 % | 508 |
| all features | 97.11 % | 93.80 % | 420 |
| reference paper, 2 features | 88.22 % | **59.56 %** | 9 |
| the frozen model, after pruning | 96.74 % | 92.03 % | **8** |

Comparator counts from the sweep are as trained. A frozen model then loses every
comparator whose two branches reach the same verdict, which is why the last row
is far smaller than the rows above it: that model's sweep point was 21 nodes and
it pruned to 8 with identical predictions.

The reference design's two features lose a third of all attacks on data neither
design was tuned against, while timing and rate hold 92 %. That gap is the case
for the extra features, measured out of sample rather than argued.

Note the per-frame recall is capped by the labelling: a frame of some other ID
inside an attacked window is labelled attack, so no per-frame detector can score
100 % against these labels. The bus-rate feature happens to fire for every ID
during a flood, which is why the numbers land as high as they do.

## Alarms, not frames

Per-frame accuracy is the wrong final metric. The residual per-frame error is
the flooded ID's own frames arriving at the flood period, which are ambiguous
one frame at a time and are *scattered singles*. An IDS does not act on one
frame: it raises an alarm on an ID, and a node exclusion system acts on that ID.

`src/eval_windows.py` adds that layer. Per ID, a leaky integrator: a flagged
frame adds one, the score bleeds off with elapsed time, and the ID alarms while
the score is at or above a threshold. A small counter and a timestamp per ID.
Scattered false positives decay before the next one arrives; a flood crosses the
threshold in milliseconds.

### The result

The deployed model is **5 trees, 8 comparator nodes, depth 2**, trained on
mixed-rate floods. At alarm threshold 8:

| trace | windows | detected | worst detect | false alarms |
|---|---|---|---|---|
| real HCRL DoS capture | 73 | **73 / 73** | 19.2 ms | 0 |
| real traffic + `0x000` flood | 5 | 5 / 5 | 5.5 ms | 0 |
| real traffic + valid-ID flood | 5 | 5 / 5 | 7.1 ms | 0 |
| real traffic + stealth on `0x2c0` | 5 | 5 / 5 | 7.1 ms | 0 |
| real traffic + stealth on `0x316` | 5 | 5 / 5 | 7.1 ms | 0 |
| real traffic + 2x low-rate flood | 7 | 7 / 7 | 58.4 ms | 0 |
| real traffic + mixed-rate floods | 43 | **43 / 43** | 60.1 ms | 0 |

**143 of 143 attack windows, zero false alarms in every second of clean traffic
measured**, across three different victim IDs and injection rates from 2x to 33x
the victim's own frame rate.

Two latencies, kept apart. The per-frame verdict is immediate. The *alarm* takes
between a few and sixty milliseconds because it waits for corroborating frames,
and the slower the flood the longer that wait. That wait is what buys the zero
false-alarm rate.

### Detection floor against attack rate

The flood's rate is the attacker's free parameter and the one the detector is
most sensitive to, so it gets measured separately rather than averaged away.
`src/rate_floor.py` labels every window with its own injection rate:

| attack rate | windows | detected | median detect |
|---|---|---|---|
| under 3x | 3 | 3 / 3 | 39.9 ms |
| 3x to 6x | 9 | 9 / 9 | 17.6 ms |
| 6x to 12x | 6 | 6 / 6 | 10.8 ms |
| 12x to 24x | 15 | 15 / 15 | 4.0 ms |
| above 24x | 10 | 10 / 10 | 1.8 ms |

Latency scales with how quiet the attacker is, which is the expected shape: less
evidence per unit time takes longer to accumulate.

### The overfitting this exposed, and a correction

An earlier version of this file recommended a model trained only on 33 %-duty
floods of a single victim ID, `0x2c0`. Tested against the traces above, that
model is far more overfit than its own numbers suggested:

| trace | mixed-trained | the earlier model |
|---|---|---|
| real HCRL DoS | 73 / 73 | 73 / 73 |
| `0x000` flood | 5 / 5 | 5 / 5 |
| valid-ID flood | 5 / 5 | 5 / 5 |
| stealth on `0x2c0` | 5 / 5 | 5 / 5 |
| **stealth on `0x316`** | **5 / 5** | **0 / 5** |
| **2x low-rate flood** | **7 / 7** | **0 / 7** |
| **mixed-rate floods** | **43 / 43** | **0 / 43** |

It is blind to a flood of a *different legitimate ID*, and blind at every rate
band including above 24x. It had effectively memorised `0x2c0`. Both models are
8 comparators, so this cost nothing in size: it is purely what the training set
spanned. The fix was to vary the attacker's rate and victim across bursts
(`attack_on_real.py --mode mixed`) and train on that.

The general lesson, and it applies to the reference paper's numbers as much as
to mine: a CAN IDS benchmark that contains one attack rate against one victim ID
cannot distinguish a detector from a lookup table.

### Two metric bugs of the same family

Both were caught by results that were impossible rather than merely bad, and
both are worth naming because they are easy to write again.

**Frame-counted history.** The first alarm rule used a shift register of the
last N verdicts per ID. It reported **1 of 73 windows detected on a trace where
the model was perfect frame by frame**. An attacker's ID goes silent between
floods, so its register never gets a zero pushed in and the alarm latches
forever. The rule now decays with elapsed time and clears an ID that falls
silent.

**Start-counted detection.** The metric then asked whether an alarm *started*
inside each window. A fast burst saturates the score, which takes over a second
to decay, so a window opening inside that shadow recorded no new start even
though the ID was alarming throughout it. That scored fast floods as *less*
detected than slow ones, which is backwards. It now asks whether the alarm was
*active* during the window, which is also the question an operator is asking.

## Which features actually earn their place

Measured by running the sweep with the group removed:

| trace | with payload features | without any |
|---|---|---|
| real HCRL DoS | 100.0000 % @ 2 nodes | 100.0000 % @ 2 nodes |
| `0x000` flood | 100.0000 % @ 1 node | 100.0000 % @ 1 node |
| **valid-ID flood** | **99.9996 % @ 5** | **99.5386 % @ 8** |
| stealth on `0x2c0` | 99.5386 % | 99.5386 % |
| stealth on `0x316` | 99.5386 % @ 9 | 99.5386 % @ 8 |
| SynCAN flooding | 97.1062 % | 97.1126 % |

Payload features earn their place in exactly one scenario: a flood that reuses a
legitimate ID but with a degenerate, all-zero payload. Against an attacker who
replays valid payloads they contribute nothing measurable.

That is a real design choice with a measured price. Dropping them removes the
64-bit last-payload state and the 135 bits of per-ID payload ROM, about 71 % of
the per-ID memory, and costs 0.46 points on the lazy valid-ID flood alone. The
`no_payload` feature set is there for that trade.

### A feature that did not earn its place

The payload features leave the victim's own in-flood frames unresolved, so a
purpose-built one was tried: a *gated* per-ID payload tracker, which follows the
ID's signal trajectory and folds in only samples already close to its estimate,
on the theory that it stays locked on the smooth legitimate signal and treats
replayed samples as outliers.

Measured inside attack windows, on the flooded ID alone, it separates the
victim's frames from the injected ones at AUC 0.634. For comparison,
`dt_ratio_q6` already there manages 0.929:

| feature | AUC, in-window, on the flooded ID |
|---|---|
| `dt_ratio_q6` | **0.929** |
| `burst` | 0.726 |
| `pl_violation` | 0.663 |
| the gated tracker | 0.634 |
| `hd`, `hd_dev` | 0.554 |

The sweep confirmed it: on the `0x316` flood the front with the tracker and the
front without it are identical to the frame, 99.538571 % and the same 2421 false
positives either way. It was removed. Recorded here because the negative result
is the useful part: the timing features are at the information limit for
per-frame discrimination inside a flood, and no payload feature moves it. What
moves it is the alarm layer above.

One measurement from that experiment is worth keeping: `0x2c0`, the ID the
original stealth trace floods, carries a *single constant payload* in the real
capture. Replaying it is indistinguishable by construction. `0x316` was added as
a second victim precisely because it carries a smoothly varying signal, 50.9 %
unique payloads with a median consecutive step of 2, and it is the realistic
spoofing target. The conclusion holds on both.

## Reading the trees

`src/show_trees.py` prints any frozen forest as trees you can follow by eye,
with each threshold translated into physical units:

```bash
python3 src/show_trees.py --model results/realatk_stealth/model.json
```

Where the same forest lives, in three forms:

| file | form | for |
|---|---|---|
| `results/<tag>/model.json` | integer node tables | the reference predictor, and anything that consumes the model |
| `src/show_trees.py` output | indented tree with units | reading |
| `results/dashboard.html` | charts and tables, all datasets | comparing at a glance |

`results/dashboard.html` is generated from `src/export_viz_data.py` output and
opens in a browser with no server. Every figure on it is read from that JSON, so
re-running a sweep and regenerating the data updates the page.

### How a verdict is reached

Every tree sees the same features and votes ATTACK or normal. The frame is
flagged when a majority of trees vote ATTACK. Tree counts are always odd so the
vote cannot tie, the same constraint the reference hardware design imposes.

Nothing about the evaluation is sequential. Every comparison in every tree
depends only on the feature vector, so all of them can be evaluated at once and
the depth of a tree costs nothing but the mux chain that selects its leaf. That
is what made the original hardware version a single clock cycle, and it is why
depth is close to free while node count is what to keep small.

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

What it did **not** buy, when the RTL still existed: the synthesised forest was
byte identical before and after, 25 LUTs either way, because yosys and ABC were
already eliminating that logic. So the gain is model size, readability, and a
node count that is finally honest rather than inflated by up to 43 %. It would
be a real saving for an implementation that stores the node table and walks it
one node at a time, where every node costs storage and a step.

## Verification

The trained thresholds convert to integers with zero loss, because every
feature is an integer and `x <= 2.5` is exactly `x <= 2` for integer `x`.
`select_model.py` asserts that the exported integer tables reproduce the
trained model exactly on both truncations, against a hard majority vote rather
than sklearn's own `predict`, and it refuses to write a model if they differ.

`prune_equivalent` is asserted the same way: the pruned trees must predict
identically to the unpruned ones, frame for frame.

`run_all.sh` refuses to continue if either truncation ends up with no attack
frames or no normal frames. A short capture whose attack bursts all land in one
half will otherwise train a model that predicts "normal" for everything and
still reports a plausible-looking accuracy.

## Recovering the RTL

The Verilog was removed on request. It is intact in git history at commit
`b915807`, where both testbenches passed at zero mismatches against the Python
pipeline and the design synthesised to 810 LUTs, 322 flip-flops, 33 BRAM18
equivalents and 2 DSP48 on xc7.

```bash
git checkout b915807 -- "random forest ids/rtl"
git checkout b915807 -- "random forest ids/src/export_verilog.py" \
                        "random forest ids/src/export_rom.py" \
                        "random forest ids/src/syn_report.py" \
                        "random forest ids/src/hw_report.py"
```

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
src/report.py          markdown report of every sweep
src/show_trees.py      print a frozen forest as readable trees
src/syncan_data.py     loader for the SynCAN benchmark
src/export_viz_data.py collect every result into one JSON for charting
src/eval_windows.py    score alarms per attack window, not frames
src/rate_floor.py      detection rate against the attacker's injection rate
src/cross_eval.py      score one frozen model against other attack styles
src/attack_on_real.py  inject a flood into HCRL's real attack-free capture
run_all.sh             CSV in, trained forest out
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
