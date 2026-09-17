# Running this on your own machine

Every command below was run end to end before being written down, and the
numbers shown are the ones it actually printed.

## 1. Dependencies

```bash
python3 -m pip install numpy pandas scikit-learn
```

That is all the detector needs. `matplotlib`, `graphviz` and the rest are not
used.

## 2. Get the data

The dataset files are deliberately **not** in this repository. SynCAN's licence
forbids redistributing the data or modified versions, this repository is
public, and the HCRL terms are their own to set. So you download them yourself,
once.

**HCRL Car-Hacking dataset** (https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
gives you two files that matter:

| file | what it is | used for |
|---|---|---|
| `normal_run_data.txt` | attack-free driving, 988 871 frames | **builds the detector** |
| `DoS_dataset.csv` | the real DoS capture | testing |

Put both in `data/`:

```bash
mkdir -p data
cp /path/to/normal_run_data.txt data/
cp /path/to/DoS_dataset.csv     data/DoS_real.csv
```

`data/` is gitignored, so nothing you put there can be committed by accident.

## 3. Build the detector

One command. It uses **no attack data at all** -- the per-ID baseline and both
thresholds come from the clean capture, which is the only thing a real vehicle
can actually supply.

```bash
python3 src/build_detector.py --clean data/normal_run_data.txt
```

```
clean capture  988871 frames over 8.4 min, 27 CAN IDs
baseline       27 IDs, bus interval 240 us  ->  results/detector/baseline.json

  id_rate      clean max 71, threshold > 80  (0 clean frames past it)
  dt_ratio_q6  clean min 32, p0.01 39, threshold <= 39  (105 clean frames past it, 0.01062 %)

the whole detector:
  flag a frame when  id_rate > 80  OR  dt_ratio_q6 <= 39
  that is 2 comparators on 2 features, no attack data used
```

That rule is the entire classifier. Three comparator nodes, two features.

## 4. Test it on the real attack

```bash
python3 src/eval_windows.py \
    --model    results/detector/model.json \
    --baseline results/detector/baseline.json \
    --csv      data/DoS_real.csv \
    --m 1,2,4,8
```

```
test slice 1649597 frames over 1111.4 s
attacks    73 windows, 336.3 s of attack, 775.1 s clean
per frame  91 false positives, 1 missed

 threshold   windows detected  median detect  worst detect  false alarms  per hour
         1        73 / 73              0.0 ms        0.5 ms             0      0.00
         2        73 / 73              1.0 ms        5.3 ms             0      0.00
         4        73 / 73              1.9 ms        6.7 ms             0      0.00
         8        73 / 73              3.8 ms       18.7 ms             0      0.00
```

Takes about 3 minutes, most of it parsing the 190 MB CSV.

## 5. Test it on attacks the HCRL capture does not contain

The real capture only holds the crude attack, a flood of ID `0x000`, which has
no baseline entry and so is separable by construction. These build harder ones
on top of the same real attack-free traffic, so the background jitter, payload
behaviour and ID mix stay real and only the attack is synthetic.

An hour of ordinary traffic with rare short bursts across all 27 victim IDs,
which is the realistic duty cycle and the run that found most of the bugs:

```bash
python3 src/soak.py --mode soak --times 7 \
    --model    results/detector/model.json \
    --baseline results/detector/baseline.json \
    --thresholds 2,4,8
```

The adversarial modes. `phase` is the one to beat: it puts every injected frame
at the exact midpoint of the victim's period, so the arrival pattern stays
perfectly regular and the rate never goes above 2x.

```bash
for mode in phase creep rampup single; do
    python3 src/soak.py --mode $mode --times 3 --seed 5 \
        --model    results/detector/model.json \
        --baseline results/detector/baseline.json \
        --thresholds 2,4
done
```

| mode | what it does |
|---|---|
| `phase` | exact midpoint of the victim's period, 2x rate, no jitter |
| `creep` | 1.2x the victim's rate, the quietest flood that still adds frames |
| `rampup` | ramps 1.2x to 8x, so it opens below any fixed threshold |
| `single` | one injected frame per burst, the limit case |

Expect `single` to score 0. An alarm that waits for corroboration cannot fire
on one frame, and that is a property of the design rather than a bug.

## 6. See the trees

```bash
python3 src/show_trees.py --model results/detector/model.json
```

## Optional: retrain a forest instead of calibrating

The calibrated rule beat every trained forest here, but the training pipeline
is intact if you want to reproduce that comparison. It needs a trace with
labelled attacks, which `src/soak.py --csv-out` will generate:

```bash
python3 src/soak.py --mode soak --times 2 --seed 11 \
    --burst 0.5,3.0 --gap 3,10 \
    --model results/detector/model.json \
    --baseline results/detector/baseline.json \
    --csv-out data/real_soaktrain.csv

python3 src/prepare.py --csv data/real_soaktrain.csv \
    --out results/soak/cache.npz --baseline-out results/soak/baseline.json
python3 src/sweep.py --cache results/soak/cache.npz \
    --out results/soak/sweep.csv --sets per_id,scale_free,no_payload
python3 src/select_model.py --cache results/soak/cache.npz \
    --sweep results/soak/sweep.csv --out results/soak/model_perid.json \
    --feature-set per_id --target 0.985 --max-nodes 10
```

## If something goes wrong

**`no configuration reaches ...`** from `select_model.py` means the accuracy
target is above anything the sweep found. Lower `--target`.

**A truncation with no attack frames** is caught loudly by `prepare.py` rather
than silently producing a model that predicts "normal" for everything at 66 %
accuracy. If you see that error, your trace has its attacks outside the
training span.

**Slow runs.** The CSV parse dominates. `src/cache_eval.py` extracts a trace's
features once into an npz, and it is verified to reproduce `eval_windows.py`
exactly on all seven traces, so use it if you are iterating.
