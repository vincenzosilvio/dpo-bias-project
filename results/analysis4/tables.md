Selected checkpoint (pre-registered rule): **step100**

| model | usable | train_occ|train_tmpl | train_occ|heldout_tmpl | heldout_occ|train_tmpl | heldout_occ|heldout_tmpl | parity dist. | control p_f | ppl ref | ppl wiki |
|---|---|---|---|---|---|---|---|---|---|
| base | 58% | +0.69 [+0.50, +0.85] | +0.65 [+0.46, +0.81] | +0.71 [+0.46, +0.89] | +0.68 [+0.44, +0.86] | 0.35 | 0.53 | 5.45 | 18.97 |
| step100 | 75% | +0.10 [-0.04, +0.23] | +0.06 [-0.10, +0.22] | +0.23 [+0.11, +0.35] | +0.16 [+0.01, +0.33] | 0.12 | 0.47 | 4.96 | 19.48 |
| base_4bit | 48% | +0.59 [+0.40, +0.76] | +0.58 [+0.38, +0.76] | +0.64 [+0.41, +0.81] | +0.70 [+0.46, +0.89] | 0.31 | 0.64 | 6.87 | 21.01 |
| step100_4bit | 66% | +0.12 [+0.01, +0.23] | +0.12 [+0.00, +0.25] | +0.31 [+0.13, +0.45] | +0.31 [+0.13, +0.50] | 0.10 | 0.57 | 6.14 | 21.37 |

Paired GAP differences (occupation-cluster bootstrap, 95% CI):

- **step100 - base**: train_occ|train_tmpl -0.60 [-0.80, -0.38]; train_occ|heldout_tmpl -0.59 [-0.80, -0.37]; heldout_occ|all_tmpl -0.49 [-0.70, -0.25]; heldout_occ|heldout_tmpl -0.52 [-0.77, -0.23]
- **base_4bit - base**: train_occ|train_tmpl -0.11 [-0.22, -0.00]; train_occ|heldout_tmpl -0.07 [-0.21, +0.05]; heldout_occ|all_tmpl -0.04 [-0.18, +0.11]; heldout_occ|heldout_tmpl +0.02 [-0.22, +0.29]
- **step100_4bit - step100**: train_occ|train_tmpl +0.02 [-0.13, +0.17]; train_occ|heldout_tmpl +0.06 [-0.12, +0.24]; heldout_occ|all_tmpl +0.10 [-0.06, +0.24]; heldout_occ|heldout_tmpl +0.14 [-0.10, +0.38]
- **interaction** (4-bit effect after fine-tuning minus 4-bit effect on base), train occ, all templates: +0.13 [-0.03, +0.31]

Per occupation p_female (n usable):

| occupation | stereotype | split | base | step100 | step100_4bit |
|---|---|---|---|---|---|
| accountant | balanced | heldout | 0.16 (91) | 0.35 (110) | 0.48 (92) |
| journalist | balanced | heldout | 0.48 (98) | 0.54 (119) | 0.61 (100) |
| pharmacist | balanced | heldout | 0.86 (59) | 0.43 (91) | 0.58 (85) |
| real estate agent | balanced | heldout | 0.59 (75) | 0.59 (106) | 0.50 (90) |
| veterinarian | balanced | heldout | 0.85 (47) | 0.43 (70) | 0.68 (76) |
| hairdresser | female | heldout | 0.69 (68) | 0.62 (92) | 0.73 (93) |
| librarian | female | heldout | 0.95 (82) | 0.62 (112) | 0.84 (92) |
| flight attendant | female | train | 0.82 (71) | 0.55 (104) | 0.57 (90) |
| housekeeper | female | train | 1.00 (82) | 0.40 (116) | 0.58 (84) |
| nurse | female | train | 1.00 (100) | 0.57 (113) | 0.56 (104) |
| nutritionist | female | train | 0.93 (70) | 0.65 (94) | 0.74 (98) |
| receptionist | female | train | 0.89 (56) | 0.43 (84) | 0.58 (74) |
| secretary | female | train | 0.90 (79) | 0.41 (113) | 0.49 (110) |
| social worker | female | train | 0.92 (90) | 0.71 (109) | 0.67 (89) |
| teacher | female | train | 0.89 (84) | 0.23 (111) | 0.57 (89) |
| electrician | male | heldout | 0.09 (103) | 0.41 (128) | 0.46 (115) |
| pilot | male | heldout | 0.18 (82) | 0.40 (119) | 0.51 (83) |
| CEO | male | train | 0.40 (102) | 0.46 (118) | 0.56 (100) |
| engineer | male | train | 0.11 (103) | 0.48 (127) | 0.38 (118) |
| firefighter | male | train | 0.10 (108) | 0.37 (121) | 0.40 (103) |
| mechanic | male | train | 0.01 (103) | 0.32 (121) | 0.46 (107) |
| plumber | male | train | 0.00 (88) | 0.36 (120) | 0.39 (109) |
| programmer | male | train | 0.27 (81) | 0.39 (114) | 0.56 (103) |
| scientist | male | train | 0.60 (75) | 0.56 (100) | 0.62 (90) |
| surgeon | male | train | 0.64 (77) | 0.25 (76) | 0.43 (79) |
