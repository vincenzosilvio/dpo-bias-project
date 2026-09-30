| model | usable | any gendered pronoun | first person | non-Latin script | mean tokens | ppl ref | ppl wiki | GAP train/train | GAP train occ/held-out tmpl |
|---|---|---|---|---|---|---|---|---|---|
| base | 57% | 94% | 22% | 0% | 117 | 5.45 | 18.97 | +0.61 [+0.39, +0.80] | +0.66 [+0.42, +0.88] |
| step10 | 27% | 70% | 55% | 0% | 137 | 5.68 | 19.00 | +0.53 [+0.33, +0.73] | +0.46 [+0.16, +0.71] |
| step20 | 4% | 59% | 55% | 0% | 93 | 7.55 | 19.28 | (+0.39 [-0.15, +1.00], n<30) | (+0.38 [-0.23, +0.86], n<30) |
| step40 | 4% | 12% | 73% | 2% | 36 | 9.97 | 21.05 | (+0.14 [+0.00, +0.36], n<30) | (+0.08 [-0.40, +0.60], n<30) |
| step60 | 4% | 9% | 47% | 2% | 54 | 15.35 | 22.41 | (-0.18 [-0.47, +0.00], n<30) | (-0.25 [-1.00, +0.00], n<30) |
| step100 | 0% | 1% | 3% | 39% | 7 | 14.80 | 20.83 | n/a | n/a |
| step153 | 0% | 0% | 2% | 34% | 10 | 16.92 | 21.17 | n/a | n/a |
| base_4bit | 45% | 93% | 22% | 0% | 130 | 6.87 | 21.01 | +0.53 [+0.31, +0.72] | +0.63 [+0.33, +0.86] |

| model | train_occ|train_tmpl | train_occ|heldout_tmpl | heldout_occ|train_tmpl | heldout_occ|heldout_tmpl | parity dist. | control p_f |
|---|---|---|---|---|---|---|
| base | +0.61 [+0.39, +0.80] | +0.66 [+0.42, +0.88] | +0.89 [+0.73, +1.00] | (+0.79 [+0.58, +0.96], n<30) | 0.36 | 0.56 |
| base_4bit | +0.53 [+0.31, +0.72] | +0.63 [+0.33, +0.86] | (+0.64 [+0.24, +1.00], n<30) | (+0.53 [+0.14, +0.86], n<30) | 0.30 | 0.63 |

GAP(4-bit) - GAP(fp32), paired occupation-cluster bootstrap:
- train_occ|train_tmpl: -0.07 [-0.28, +0.11]
- train_occ|heldout_tmpl: -0.03 [-0.24, +0.18]
- train_occ|all_tmpl: -0.06 [-0.22, +0.07]

Usable rate by stereotype group:
- base: balanced 53%, female 55%, male 62%
- base_4bit: balanced 48%, female 43%, male 45%
