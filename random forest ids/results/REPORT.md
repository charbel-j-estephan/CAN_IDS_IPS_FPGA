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
| 1 | 8 | 30 | 99.0070% | 96.0751% | 3201 | 10665 | dt_id_dev,hd_dev |
| 1 | 8 | 38 | 99.0140% | 96.1014% | 3072 | 10695 | dt_id_dev,hd_dev |

## dataset: `real`

### feature set `timing` — no identity features, payload-content features included

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | 100.0000% | 100.0000% | 0 | 0 | dt_ratio_q6,id_rate |

### feature set `timing_only` — no identity features, no payload-content features

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | 100.0000% | 100.0000% | 0 | 0 | dt_ratio_q6,id_rate |

### feature set `paper` — the reference paper's two features only

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 93.3957% | 66.2283% | 67903 | 41041 | dt_id_dev,hd_dev |
| 1 | 3 | 5 | 95.0091% | 72.1836% | 41289 | 41041 | dt_id_dev,hd_dev |
| 1 | 4 | 12 | 96.2435% | 80.8050% | 44534 | 17433 | dt_id_dev,hd_dev |
| 1 | 5 | 17 | 97.9135% | 88.1461% | 14525 | 19894 | dt_id_dev,hd_dev |
| 1 | 8 | 71 | 98.3019% | 90.7893% | 18204 | 9808 | dt_id_dev,hd_dev |
| 3 | 8 | 202 | 98.3036% | 90.8051% | 18294 | 9689 | dt_id_dev,hd_dev |
| 3 | 8 | 208 | 98.3037% | 90.8054% | 18293 | 9689 | dt_id_dev,hd_dev |

## dataset: `realatk_stealth`

### feature set `timing` — no identity features, payload-content features included

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.5384% | 98.7181% | 2423 | 4 | dt_id_dev,dt_ratio_q6,id_rate |
| 3 | 2 | 7 | 99.5386% | 98.7186% | 2421 | 5 | dt_id,dt_id_dev,dt_ratio_q6,pl_popcount,id_rate |

### feature set `timing_only` — no identity features, no payload-content features

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.5384% | 98.7181% | 2423 | 4 | dt_id,burst |
| 3 | 2 | 8 | 99.5386% | 98.7186% | 2421 | 5 | dt_id,dt_id_dev,burst |

### feature set `paper` — the reference paper's two features only

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.2032% | 97.8078% | 4184 | 5 | dt_id_dev,hd_dev |
| 1 | 3 | 5 | 99.3784% | 98.2815% | 3263 | 5 | dt_id_dev,hd_dev |
| 1 | 5 | 10 | 99.5184% | 98.6633% | 2527 | 5 | dt_id_dev,hd_dev |
| 1 | 6 | 12 | 99.5386% | 98.7186% | 2421 | 5 | dt_id_dev,hd_dev |

## dataset: `realatk_valid-id`

### feature set `timing` — no identity features, payload-content features included

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 95.1044% | 87.8958% | 25739 | 0 | pl_violation |
| 3 | 2 | 5 | 99.9996% | 99.9989% | 0 | 2 | dt_id,dt_ratio_q6,pl_violation,pl_popcount |

### feature set `timing_only` — no identity features, no payload-content features

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.5390% | 98.6861% | 0 | 2424 | dt_ratio_q6,hd_dev,bus_rate |
| 1 | 3 | 5 | 99.6559% | 99.0370% | 1376 | 433 | dt_ratio_q6,hd_dev,dt_bus,burst,bus_rate |
| 1 | 4 | 8 | 99.6637% | 99.0610% | 1572 | 196 | dt_id,dt_ratio_q6,hd_dev,dt_bus,burst,bus_rate |
| 1 | 5 | 14 | 99.6677% | 99.0703% | 1373 | 374 | dt_id,dt_id_dev,hd,hd_dev,dt_bus,burst,id_rate,bus_rate |
| 1 | 8 | 33 | 99.6721% | 99.0819% | 1298 | 426 | dt_id,dt_id_dev,dt_ratio_q6,hd_dev,dt_bus,burst,id_rate,bus_rate |

### feature set `paper` — the reference paper's two features only

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.2032% | 97.8078% | 4184 | 5 | dt_id_dev,hd_dev |
| 1 | 3 | 5 | 99.3786% | 98.2372% | 843 | 2424 | dt_id_dev,hd_dev |
| 1 | 5 | 11 | 99.5186% | 98.6288% | 107 | 2424 | dt_id_dev,hd_dev |
| 1 | 6 | 13 | 99.5196% | 98.6665% | 2521 | 5 | dt_id_dev,hd_dev |
| 1 | 6 | 14 | 99.5243% | 98.6483% | 313 | 2188 | dt_id_dev,hd_dev |
| 1 | 7 | 16 | 99.5445% | 98.7049% | 207 | 2188 | dt_id_dev,hd_dev |
| 3 | 8 | 58 | 99.5469% | 98.7118% | 194 | 2188 | dt_id_dev,hd_dev |

## dataset: `realatk_zero-id`

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
| 1 | 2 | 2 | 82.2251% | 0.0000% | 0 | 93453 | dt_id_dev,hd_dev |
| 1 | 3 | 3 | 95.4793% | 88.6222% | 22880 | 888 | dt_id_dev,hd_dev |
| 1 | 5 | 6 | 97.7798% | 93.9810% | 9351 | 2322 | dt_id_dev,hd_dev |
| 1 | 6 | 8 | 97.9898% | 94.0200% | 201 | 10368 | dt_id_dev,hd_dev |
| 1 | 7 | 10 | 99.2789% | 97.9366% | 305 | 3486 | dt_id_dev,hd_dev |

## dataset: `stealth`

### feature set `timing` — no identity features, payload-content features included

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.5835% | 98.4131% | 5785 | 31 | dt_id_dev,dt_ratio_q6,id_rate |
| 3 | 2 | 9 | 99.5851% | 98.4193% | 5761 | 32 | dt_id,dt_ratio_q6,id_rate |
| 1 | 4 | 12 | 99.7479% | 99.0267% | 2222 | 1298 | dt_ratio_q6,hd,hd_dev,burst,pl_popcount,id_rate |
| 1 | 5 | 22 | 99.7511% | 99.0391% | 2246 | 1230 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,pl_popcount,id_rate,bus_rate |
| 1 | 8 | 96 | 99.7523% | 99.0450% | 2447 | 1012 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,pl_popcount,id_rate,bus_rate |
| 3 | 6 | 121 | 99.7622% | 99.0833% | 2378 | 942 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,pl_popcount,id_rate,bus_rate |
| 7 | 6 | 277 | 99.7660% | 99.0978% | 2309 | 958 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,pl_popcount,id_rate,bus_rate |

### feature set `timing_only` — no identity features, no payload-content features

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.5835% | 98.4131% | 5785 | 31 | dt_ratio_q6,id_rate |
| 3 | 2 | 9 | 99.5851% | 98.4193% | 5761 | 32 | dt_ratio_q6,id_rate |
| 1 | 4 | 12 | 99.6671% | 98.7159% | 2967 | 1682 | dt_id,dt_ratio_q6,hd,hd_dev,burst,id_rate |
| 1 | 5 | 22 | 99.6702% | 98.7283% | 2991 | 1614 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,id_rate |
| 1 | 6 | 40 | 99.6848% | 98.7846% | 2882 | 1519 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,dlc,id_rate,bus_rate |
| 1 | 6 | 41 | 99.6883% | 98.7980% | 2840 | 1512 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,id_rate |
| 1 | 8 | 98 | 99.6924% | 98.8131% | 2708 | 1587 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,id_rate,bus_rate |
| 3 | 8 | 302 | 99.6966% | 98.8283% | 2546 | 1691 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,id_rate,bus_rate |
| 5 | 8 | 507 | 99.6976% | 98.8327% | 2583 | 1639 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,id_rate,bus_rate |
| 7 | 8 | 707 | 99.6979% | 98.8335% | 2540 | 1678 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,id_rate,bus_rate |

### feature set `paper` — the reference paper's two features only

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.0056% | 96.1774% | 8185 | 5700 | dt_id_dev,hd_dev |
| 1 | 4 | 12 | 99.0644% | 96.4302% | 9139 | 3925 | dt_id_dev,hd_dev |
| 1 | 6 | 40 | 99.1735% | 96.8275% | 7273 | 4267 | dt_id_dev,hd_dev |
| 1 | 8 | 87 | 99.2868% | 97.2805% | 7694 | 2264 | dt_id_dev,hd_dev |
| 3 | 8 | 263 | 99.3050% | 97.3481% | 7439 | 2265 | dt_id_dev,hd_dev |
| 5 | 8 | 437 | 99.3062% | 97.3520% | 7385 | 2302 | dt_id_dev,hd_dev |
| 7 | 8 | 613 | 99.3069% | 97.3550% | 7413 | 2265 | dt_id_dev,hd_dev |
| 9 | 8 | 796 | 99.3078% | 97.3579% | 7362 | 2303 | dt_id_dev,hd_dev |

## dataset: `validid`

### feature set `timing` — no identity features, payload-content features included

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 100.0000% | 100.0000% | 0 | 0 | pl_violation |

### feature set `timing_only` — no identity features, no payload-content features

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.5835% | 98.4131% | 5785 | 31 | dt_ratio_q6,hd_dev,id_rate |
| 1 | 3 | 7 | 99.6994% | 98.8390% | 2514 | 1684 | dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,id_rate |
| 1 | 5 | 19 | 99.7016% | 98.8477% | 2475 | 1691 | dt_id,dt_id_dev,dt_ratio_q6,hd_dev,burst,id_rate |
| 1 | 6 | 29 | 99.7057% | 98.8605% | 1976 | 2133 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,dlc,id_rate |
| 1 | 6 | 30 | 99.7074% | 98.8686% | 2245 | 1841 | dt_id,dt_id_dev,dt_ratio_q6,hd_dev,burst,id_rate |
| 1 | 8 | 71 | 99.7089% | 98.8750% | 2331 | 1734 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,burst,id_rate,bus_rate |
| 3 | 6 | 92 | 99.7097% | 98.8767% | 2099 | 1955 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,id_rate,bus_rate |
| 5 | 6 | 153 | 99.7100% | 98.8769% | 1919 | 2130 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,id_rate,bus_rate |
| 3 | 8 | 206 | 99.7110% | 98.8820% | 2138 | 1898 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,id_rate,bus_rate |
| 7 | 6 | 215 | 99.7110% | 98.8814% | 1996 | 2039 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,id_rate,bus_rate |
| 7 | 6 | 242 | 99.7125% | 98.8878% | 2125 | 1890 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc,id_rate,bus_rate |
| 9 | 6 | 316 | 99.7132% | 98.8914% | 2215 | 1789 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc,id_rate,bus_rate |
| 7 | 8 | 480 | 99.7155% | 98.9002% | 2193 | 1779 | dt_id,dt_id_dev,dt_ratio_q6,hd,hd_dev,dt_bus,burst,dlc,id_rate,bus_rate |

### feature set `paper` — the reference paper's two features only

| trees | depth | nodes | accuracy | F1 | FP | FN | split on |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 99.0056% | 96.1774% | 8185 | 5700 | dt_id_dev,hd_dev |
| 1 | 3 | 6 | 99.1000% | 96.4545% | 3135 | 9432 | dt_id_dev,hd_dev |
| 1 | 4 | 13 | 99.1200% | 96.6378% | 8493 | 3794 | dt_id_dev,hd_dev |
| 1 | 5 | 21 | 99.1650% | 96.8002% | 7641 | 4018 | dt_id_dev,hd_dev |
| 1 | 6 | 34 | 99.1990% | 96.9066% | 6008 | 5177 | dt_id_dev,hd_dev |
| 1 | 6 | 36 | 99.1991% | 96.8850% | 4723 | 6460 | dt_id_dev,hd_dev |
| 1 | 8 | 63 | 99.2360% | 97.0427% | 5327 | 5341 | dt_id_dev,hd_dev |
| 1 | 8 | 69 | 99.2922% | 97.2628% | 5099 | 4784 | dt_id_dev,hd_dev |
| 3 | 8 | 202 | 99.3012% | 97.2738% | 3454 | 6303 | dt_id_dev,hd_dev |
| 5 | 8 | 347 | 99.3034% | 97.2875% | 3792 | 5935 | dt_id_dev,hd_dev |

