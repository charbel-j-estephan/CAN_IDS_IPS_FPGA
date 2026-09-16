# Random forest CAN IDS: accuracy versus hardware cost

`nodes` counts internal comparator nodes summed over the forest, which
is what the FPGA pays for in area. Latency is a fixed 4 cycles per
frame in every configuration, so depth does not change the timing
budget, only the area.

## dataset: `dos`

### feature set `timing` — no identity features, payload-content features included

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 100.0000% | 100.0000% | 0 | 0 | dt_ratio_q6 |

### feature set `timing_only` — no identity features, no payload-content features

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 100.0000% | 100.0000% | 0 | 0 | dt_ratio_q6 |

### feature set `paper` — the reference paper's two features only

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 98.4331% | 93.5957% | 1382 | 20497 | dt_id_dev |
| 1 | 3 | 5 | 98.4635% | 93.7123% | 957 | 20497 | dt_id_dev,hd_dev |
| 1 | 3 | 6 | 98.8165% | 95.2581% | 2133 | 14392 | dt_id_dev,hd_dev |
| 1 | 4 | 8 | 98.8344% | 95.3262% | 1884 | 14392 | dt_id_dev,hd_dev |
| 1 | 4 | 11 | 98.8596% | 95.4387% | 2133 | 13790 | dt_id_dev,hd_dev |
| 1 | 5 | 18 | 98.8648% | 95.4614% | 2177 | 13674 | dt_id_dev,hd_dev |
| 1 | 6 | 19 | 98.8652% | 95.4629% | 2178 | 13668 | dt_id_dev,hd_dev |
| 1 | 6 | 26 | 98.8710% | 95.4850% | 2084 | 13680 | dt_id_dev,hd_dev |

### feature set `full` — every feature, including `can_id` and `id_known`

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 100.0000% | 100.0000% | 0 | 0 | can_id |

## dataset: `stealth`

### feature set `timing` — no identity features, payload-content features included

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.2651% | 97.1676% | 5895 | 4366 | dt_id,dt_id_dev,dt_ratio_q6 |
| 1 | 3 | 7 | 99.3166% | 97.3606% | 5177 | 4366 | dt_id,dt_id_dev,dt_ratio_q6,burst,pl_popcount |
| 1 | 4 | 15 | 99.3452% | 97.5081% | 7651 | 1492 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,pl_popcount |
| 1 | 5 | 29 | 99.4665% | 97.9544% | 5425 | 2024 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,dlc,pl_popcount |
| 1 | 6 | 52 | 99.5200% | 98.1503% | 4138 | 2564 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,dlc,pl_popcount |
| 3 | 6 | 156 | 99.5311% | 98.1913% | 3913 | 2635 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,dlc,pl_popcount |
| 5 | 6 | 257 | 99.5315% | 98.1952% | 4132 | 2410 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,dlc,pl_popcount |
| 3 | 8 | 383 | 99.5480% | 98.2686% | 5037 | 1274 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc,pl_popcount |
| 5 | 8 | 637 | 99.5495% | 98.2743% | 5016 | 1274 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc,pl_popcount |
| 9 | 8 | 1147 | 99.5641% | 98.3261% | 4464 | 1622 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc,pl_popcount |

### feature set `timing_only` — no identity features, no payload-content features

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.2651% | 97.1676% | 5895 | 4366 | dt_id,dt_id_dev,dt_ratio_q6 |
| 1 | 3 | 7 | 99.2892% | 97.2578% | 5559 | 4366 | dt_id,dt_id_dev,dt_ratio_q6,burst |
| 1 | 4 | 14 | 99.3196% | 97.3879% | 6225 | 3275 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst |
| 1 | 5 | 28 | 99.3721% | 97.5984% | 6554 | 2214 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst |
| 1 | 6 | 52 | 99.3871% | 97.6552% | 6392 | 2166 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst |
| 3 | 5 | 86 | 99.3991% | 97.6991% | 6166 | 2225 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc |
| 1 | 8 | 112 | 99.4344% | 97.8405% | 6418 | 1479 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc |
| 3 | 6 | 155 | 99.4482% | 97.8931% | 6333 | 1372 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc |
| 3 | 8 | 340 | 99.4771% | 98.0083% | 6561 | 740 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc |
| 5 | 8 | 561 | 99.4776% | 98.0081% | 6387 | 908 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc |
| 9 | 8 | 1009 | 99.4798% | 98.0192% | 6618 | 646 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc |

### feature set `paper` — the reference paper's two features only

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.0056% | 96.1774% | 8185 | 5700 | dt_id_dev,hd_dev |
| 1 | 4 | 12 | 99.0644% | 96.4302% | 9139 | 3925 | dt_id_dev,hd_dev |
| 3 | 4 | 36 | 99.0665% | 96.4374% | 9087 | 3948 | dt_id_dev,hd_dev |
| 1 | 6 | 40 | 99.1735% | 96.8275% | 7273 | 4267 | dt_id_dev,hd_dev |
| 1 | 8 | 87 | 99.2868% | 97.2805% | 7694 | 2264 | dt_id_dev,hd_dev |
| 3 | 8 | 263 | 99.3194% | 97.4006% | 7188 | 2316 | dt_id_dev,hd_dev |
| 5 | 8 | 437 | 99.3207% | 97.4028% | 6970 | 2515 | dt_id_dev,hd_dev |
| 9 | 8 | 796 | 99.3214% | 97.4072% | 7099 | 2377 | dt_id_dev,hd_dev |

## dataset: `validid`

### feature set `timing` — no identity features, payload-content features included

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 100.0000% | 100.0000% | 0 | 0 | pl_violation |

### feature set `timing_only` — no identity features, no payload-content features

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.2848% | 97.1761% | 1435 | 8551 | dt_id,dt_ratio_q6,hd |
| 1 | 3 | 7 | 99.2903% | 97.2619% | 5544 | 4366 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev |
| 1 | 4 | 14 | 99.3489% | 97.4968% | 5760 | 3331 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst |
| 1 | 5 | 27 | 99.4095% | 97.7377% | 5975 | 2270 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst |
| 1 | 6 | 43 | 99.4226% | 97.7637% | 3934 | 4129 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,dlc |
| 3 | 6 | 129 | 99.4629% | 97.9418% | 5553 | 1946 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc |

### feature set `paper` — the reference paper's two features only

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.0056% | 96.1774% | 8185 | 5700 | dt_id_dev,hd_dev |
| 1 | 3 | 6 | 99.1000% | 96.4545% | 3135 | 9432 | dt_id_dev,hd_dev |
| 1 | 4 | 13 | 99.1200% | 96.6378% | 8493 | 3794 | dt_id_dev,hd_dev |
| 1 | 5 | 21 | 99.1650% | 96.8002% | 7641 | 4018 | dt_id_dev,hd_dev |
| 1 | 6 | 34 | 99.1990% | 96.9066% | 6008 | 5177 | dt_id_dev,hd_dev |
| 1 | 6 | 36 | 99.1991% | 96.8850% | 4723 | 6460 | dt_id_dev,hd_dev |
| 3 | 6 | 103 | 99.2057% | 96.9224% | 5364 | 5727 | dt_id_dev,hd_dev |
| 3 | 6 | 111 | 99.2313% | 97.0170% | 4899 | 5834 | dt_id_dev,hd_dev |

### feature set `full` — every feature, including `can_id` and `id_known`

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 100.0000% | 100.0000% | 0 | 0 | pl_violation |

