# Fine-Tuning a Small LLM against Gender-Occupation Bias: DPO Failures, a Working Fix, and What It Costs

**TL;DR.** On Qwen2.5-0.5B-Instruct, with 446 counterfactual pairs and 21
minutes / 23.5 Wh of LoRA training on one T4, **supervised fine-tuning on
gender-swapped texts cut the gender-occupation gap in generated text from
+0.61 to +0.04.** It also generalised: on occupations never trained on,
the gap fell by −0.49 (95% CI [−0.69, −0.28]), and on prompt templates
never trained on by −0.61 [−0.93, −0.27]. Text quality held: usable texts
rose from 57% to 76%, and WikiText perplexity moved +2.7%. **Plain DPO
on the same pairs failed twice.** Run 1 collapsed the model, which
stopped writing gendered third-person texts. Run 2 (DPO + NLL) drifted
toward first-person narration and was stopped by the early-stop rule.
In both, every offline DPO metric looked healthy; only the pre-registered
check on generated text caught it. **4-bit NF4 quantization** of the
debiased model moved the gap back by +0.10 [−0.11, +0.30]: not
significant at this sample size, but in the direction that matters for
deploying small debiased models.

## Motivation

Small, quantized LLMs are what actually gets deployed under a compute
budget. Two questions follow: do efficiency techniques change a model's
social biases, and can lightweight alignment correct those biases at a
small fraction of the training cost? This project builds a measured,
reproducible pipeline to test both questions on one model. It reports
cost next to bias and text quality, because a debiasing method that
breaks the model, or an efficiency gain that loses the output, is not a
gain.

## Method

1. **Data.** The base model writes short single-character texts
   ("Write about a nurse finishing a long shift. Begin with the nurse's
   first name ..."). Prompts never mention pronouns (lesson #10). Each
   text is labelled by its pronouns only when the text has exactly one
   named character whose name agrees with the pronouns (`text_utils.py`,
   spaCy + names corpus).
2. **Direction from the model, not from stereotypes.** A calibration
   half of the samples measures each occupation's p_female. Occupations
   with |p_female − 0.5| ≥ 0.20 get pairs, pushed toward the minority
   gender. This rule was fixed before the data existed (`build_pairs.py`).
3. **Counterfactual pairs.** Rejected = the model's majority-gender
   text. Chosen = the same text with the character's gender swapped
   (pronouns, name, title, role nouns; `counterfactual.py`). 3,600
   completions gave 446 pairs across 14 occupations (211 male→female,
   235 female→male). There are 42 unit and regression tests.
4. **Training** (`train_dpo.py`, `trl` 1.14, LoRA r=16 on all linear
   layers, 8.8M trainable parameters, 1.75%). Three objectives on the
   same pairs:
   - run 1: plain DPO (β 0.1, lr 5e-5)
   - run 2: DPO + NLL on chosen (lr 1e-5 and 3e-5)
   - run 3: supervised fine-tuning on the chosen texts only (lr 5e-5)

   Each runs 3 epochs (153 steps), with checkpoints every 10 steps. From
   run 2 on, a generation monitor runs every 10 steps with an early-stop
   rule.
5. **Evaluation** (`evaluate.py`). Each model writes 1,200 fresh texts
   (150 prompts × 8, new seed), on the grid {train, held-out
   occupations} × {train, held-out templates}. Metrics: GAP =
   p_female(female-coded) − p_female(male-coded), usable rate,
   perplexity, tokens/s, GPU energy (NVML). **Pre-registered selection
   rule:** among checkpoints whose usable rate is at most 10 points below
   the base model's, take the smallest |GAP| on train occupations ×
   train templates. The selected checkpoint and the base model are also
   evaluated in 4-bit NF4.
6. **Analysis** (`analyze_eval.py`, `analyze_finetune.py`, CPU only).
   Occupation-cluster bootstrap CIs, paired comparisons with the same
   occupation draws, per-occupation shifts, and figures.

## Results: run 3, supervised fine-tuning on counterfactual texts

![trajectory](results/analysis3/fig4_finetune_trajectory.png)

The rule selected **step 100**. Every run-3 checkpoint passed the
quality guard.

| model | usable | GAP train occ / train tmpl | GAP train occ / held-out tmpl | GAP held-out occ / train tmpl | GAP held-out occ / held-out tmpl | per-occupation distance from 50/50 | WikiText ppl |
|---|---|---|---|---|---|---|---|
| base | 57% | +0.61 [+0.39, +0.80] | +0.66 [+0.42, +0.88] | +0.89 [+0.73, +1.00] | +0.79 [+0.58, +0.96]* | 0.36 | 18.97 |
| **step 100 (selected)** | **76%** | **+0.04 [−0.11, +0.17]** | **+0.05 [−0.16, +0.25]** | **+0.38 [+0.12, +0.62]** | **+0.32 [+0.00, +0.61]*** | **0.13** | **19.48** |
| step 100, 4-bit | 63% | +0.14 [−0.01, +0.28] | +0.13 [−0.08, +0.32] | +0.37 [+0.14, +0.61] | +0.48 [+0.18, +0.74]* | 0.10 | 21.35 |

Brackets are 95% occupation-cluster bootstrap CIs. \* = fewer than 30
usable texts per side in the base model. Full table for all checkpoints:
`results/analysis3/tables.md`.

**Paired change, step 100 − base** (same occupation draws for both):

| cell | ΔGAP [95% CI] | how clean a test |
|---|---|---|
| train occupations, train templates | −0.57 [−0.78, −0.36] | optimistic: the selection cell |
| train occupations, **held-out templates** | **−0.61 [−0.93, −0.27]** | clean for new prompts |
| **held-out occupations**, all templates | **−0.49 [−0.69, −0.28]** | clean for new occupations (2 per side) |
| held-out occupations, held-out templates | −0.47 [−0.84, −0.09] | clean, small |

![per occupation](results/analysis3/fig5_per_occupation.png)

- **The bias generalised beyond the training data.** Pilot went 0.04 →
  0.38 female and electrician 0.05 → 0.29; librarian went 0.96 → 0.69
  and hairdresser 0.85 → 0.71. None of these were trained on. The control
  occupations moved toward parity too (veterinarian 1.00 → 0.50,
  pharmacist 0.81 → 0.44). The model weakened the occupation → gender
  association in general, not only for the 14 trained occupations.
- **Some overshoot.** Teacher went 0.79 → 0.13 female, past parity, and
  receptionist 0.88 → 0.33. CEO moved *against* its training direction
  (0.45 → 0.29). At about 30–40 texts per occupation, single-occupation
  values carry ±0.15 of noise. Still, GAP near 0 on average does not mean
  every occupation sits at 50/50.
- **Quality.** The usable rate went *up*, 57% → 76% (fewer first-person
  and multi-person texts), and names became more varied (71 → 92
  distinct; "John" fell from 32% to 7% of texts). Two costs: WikiText
  perplexity rose 2.7% (18.97 → 19.48), and **texts drifting into
  Chinese** rose from 0.4% to 4.3% (e.g. "herding all客户的头发…"). The
  usable filter does not catch the latter, so it is a real quality loss
  that the headline numbers do not show.
- **Same-prompt examples** (base vs step 100, held-out occupations):
  `results/analysis3/examples.md`.

**4-bit quantization after fine-tuning**

![quantization](results/analysis3/fig6_quantization_after_finetune.png)

| paired change (4-bit − fp32) | train occ / train tmpl | train occ / held-out tmpl | held-out occ / held-out tmpl |
|---|---|---|---|
| base model | −0.07 [−0.30, +0.12] | −0.03 [−0.25, +0.19] | −0.26 [−0.70, +0.11] |
| fine-tuned (step 100) | +0.10 [−0.11, +0.30] | +0.08 [−0.20, +0.35] | +0.16 [−0.29, +0.58] |

Interaction (the 4-bit effect after fine-tuning minus the 4-bit effect on
the base model), train occupations, all templates: **+0.15 [−0.07, +0.40]**.
On the base model, quantization did not increase the gap. On the
debiased model, the point estimates go back toward the original bias,
but no CI excludes 0. This is the question to settle with more samples
and more models. Usable texts fell 76% → 63%. The 4-bit model ran slower
(42 vs 113 tok/s) and used 75% more GPU energy per 1k tokens (0.284 vs
0.162 Wh). Part of that is because the LoRA adapter stays unmerged on
NF4 weights, so this is not a clean deployment comparison.

**Cost of run 3** (`results/run3/resource_report.json`, NVML, one T4):
the full 153 steps took 83.4 min and 83.1 Wh, of which the generation
monitor accounted for 62.4 min and 59.6 Wh. **Training alone: 21.0 min
and 23.5 GPU-Wh**, with peak memory 7.4 GB. Carbontracker, which counts the
monitor and applies its default PUE of 1.58, reports 126 Wh and 48 gCO2eq.
The selected checkpoint (step 100) needed about two thirds of the
training. For comparison, evaluating one model (1,200 generations plus
perplexity) costs about 20–25 min and 23 Wh, about as much as training
it. The monitor log (`results/run3/monitor.jsonl`) matches the
evaluation: GAP +0.69 → about 0 by step 100, with usable texts at 75–89%
throughout.

## Run 1 in detail: plain DPO collapsed

### 1. DPO collapsed the model before it reduced the bias

![collapse](results/analysis/fig1_collapse.png)

| model | usable | any gendered pronoun | first person | mean tokens | ppl ref | ppl wiki | GAP train/train | GAP train occ / held-out tmpl |
|---|---|---|---|---|---|---|---|---|
| base | 57% | 94% | 22% | 117 | 5.45 | 18.97 | +0.61 [+0.39, +0.80] | +0.66 [+0.42, +0.88] |
| step 10 | 27% | 70% | 55% | 137 | 5.68 | 19.00 | +0.53 [+0.33, +0.73] | +0.46 [+0.16, +0.71] |
| step 20 | 4% | 59% | 55% | 93 | 7.55 | 19.28 | *(n < 30)* | *(n < 30)* |
| step 40 | 4% | 12% | 73% | 36 | 9.97 | 21.05 | *(n < 30)* | *(n < 30)* |
| step 60 | 4% | 9% | 47% | 54 | 15.35 | 22.41 | *(n < 30)* | *(n < 30)* |
| step 100 | 0% | 1% | 3% | 7 | 14.80 | 20.83 | n/a | n/a |
| step 153 | 0% | 0% | 2% | 10 | 16.92 | 21.17 | n/a | n/a |

GAP values carry 95% occupation-cluster bootstrap CIs.

![gap](results/analysis/fig2_gap.png)

- **No checkpoint passes the quality guard** (base 57%; best checkpoint
  27%), so the pre-registered rule selects nothing. The trained model
  is not debiased. It is broken.
- **What the collapse looks like.** At step 10 the model starts writing
  in the first person (22% → 55% of texts, e.g. *"My name is Emily. I'm
  a software engineer..."*). By step 40 it writes one-line greetings
  (*"Greetings, I'm the hairdresser; thank you!"*). From step 100 it
  writes fragments, a third of them in non-Latin script (*"宣告"*,
  *"öhne"*). The share of texts with any he/she pronoun falls from 94%
  to 70%, then 12%, then 0%. The model learned to **avoid the format
  the pairs were written in**, not to change which gender it picks.
- **The offline metrics missed it.** Held-out reward accuracy stayed at
  0.91–0.98 throughout. Reference perplexity moved only 5.45 → 5.68 at
  step 10, while the usable rate had already halved. The one warning
  sign in training was that the chosen texts' log-probability also fell
  (−196 → −252; lesson #14). That is the known *likelihood displacement*
  failure of DPO on near-identical pairs: Pal et al. 2024 (DPO-Positive)
  and Razin et al. 2025 show that when chosen and rejected differ by a
  few tokens, DPO lowers both and moves probability mass to unrelated
  outputs. A counterfactual gender swap is exactly that kind of pair.
- **Step 10's GAP is not evidence of reduction.** +0.53 vs +0.61 is well
  inside the CIs. It is also measured on a self-selected 27% of texts.
  Where movement exists, it runs one way only. On texts with pure
  pronouns (any name), male-coded occupations moved toward female
  (p_female 0.26 → 0.40) while female-coded ones did not move
  (0.90 → 0.93). The control group jumped from 0.56 to 0.85 female: the
  model's overall female skew grew rather than the gap closing.

### 2. 4-bit quantization: no detectable bias change, and not an energy saving

![quantization](results/analysis/fig3_quantization.png)

| base model | usable | GAP train/train | GAP train occ / held-out tmpl | tok/s | GPU Wh / 1k tokens | GPU Wh / usable text | weights | peak GPU mem |
|---|---|---|---|---|---|---|---|---|
| fp32 | 57% | +0.61 [+0.39, +0.80] | +0.66 [+0.42, +0.88] | 115 | 0.161 | 0.033 | 1,976 MB | 2.16 GB |
| NF4 4-bit | 45% | +0.53 [+0.31, +0.72] | +0.63 [+0.33, +0.86] | 78 | 0.208 | 0.060 | 451 MB | 0.61 GB |

- **Bias.** GAP(4-bit) − GAP(fp32) on train occupations, all templates,
  is **−0.06 [−0.22, +0.07]** (paired occupation-cluster bootstrap). No
  detectable change. With this sample the data cannot rule out shifts of
  about ±0.2, so this is an absence of evidence, not evidence of
  absence. Per occupation (figure) the movement looks mostly like noise:
  about 20–28 usable texts per occupation. The largest shift is housekeeper,
  1.00 → 0.65 female.
- **Quality.** The usable rate fell 12 points (57% → 45%). The most
  common reason was more texts that mix he and she for one character
  (142 → 214 of 1,200). The drop was uneven: male-coded occupations
  lost 17 points (62% → 45%), female-coded 12 (55% → 43%). Reference
  perplexity rose from 5.45 to 6.87.
- **Cost.** NF4 cut weights by 77% and peak memory by 72%. On a T4 at
  batch size 8 it was also 32% slower and used 29% more GPU energy per
  generated token (dequantization overhead on a small model). Per usable
  text, the unit that counts here, energy went up 84%. The memory saving
  is real; an energy saving would need other hardware or kernels, and
  this run does not show one.

### Held-out occupations

On held-out occupations (pilot, electrician, hairdresser, librarian) the
base model's GAP is +0.89 [+0.73, +1.00] on train templates. This cell has
only two occupations per side, so its CI rests on two clusters and should
not be over-read. The held-out × held-out cell has fewer than 30 usable
texts per side. The DPO checkpoints are not reported there: none passed
the guard.

### Training cost

24 min on one Tesla T4, 26.0 Wh GPU energy from the NVML hardware counter.
Carbontracker reports 41.2 Wh, which is 26.0 Wh × its default PUE of
1.58, so the two measurements agree: 15.8 gCO2eq at 384 g/kWh. Peak memory
was 7.0 GB. GPU only: CPU energy counters are not readable on Kaggle. The
evaluation costs more than the training: 23–32 Wh per full-length model
to generate 1,200 texts, and about 2.3 h of generation and perplexity
for all eight models.

## What the project shows, and what it doesn't

- ✅ A small, cheap fine-tune on counterfactual texts **reduced
  gender-occupation bias in generation, and the effect generalised** to
  unseen prompts and unseen occupations. The checkpoint was chosen by a
  rule fixed before the results, and quality was guarded.
- ✅ **Plain DPO on minimal-edit counterfactual pairs failed in a
  specific, repeatable way:** generations drifted away from the format
  the pairs share, while reward accuracy stayed above 0.9. A
  generation-based monitor caught it. Judge alignment for bias by what
  the model generates.
- ✅ Measured costs: training in minutes and tens of Wh. NF4 cut memory
  by 77% but did not save energy on a T4 for this model size.
- ⚠️ A hint, not a finding, that 4-bit quantization partly undoes
  debiasing. The interaction is +0.15, with a CI that includes 0.
- ❌ No working DPO variant yet. The first-token explanation (lesson #16)
  is a hypothesis.
- ❌ One model (0.5B), one seed, one family of short templates.

## Limitations

- **One model, one seed, one hyperparameter setting per run.** Runs 2
  and 3 were designed after seeing run 1 (and run 3 after run 2). The
  order is recorded in this README, and the held-out cells were never
  used for any choice.
- **Post-hoc analysis choices.** The occupation-cluster bootstrap, the
  30-texts-per-side threshold, the parity metric and the pronoun
  diagnostics were chosen after the results were seen. They are
  reported as exploratory. Only the selection rule and the metric
  definitions in "Evaluation design decisions" were fixed in advance.
- **GAP uses human stereotype labels, but training followed the model's
  own calibration.** For this model all 14 targeted occupations were
  pushed against their stereotype label, so the two agree here. In
  general they need not: for this model scientist and surgeon skew
  female (lesson #12), and pushing such an occupation toward parity
  would raise GAP.
- **Label noise.** Pronoun labels rely on single-character texts. The
  manual audit of the 25-pair sample is still pending.
- **Small cells.** Held-out occupations have two occupations per
  stereotype. The 4-bit comparison can only detect GAP changes of
  about 0.2 or more.
- **Binary gender.** The metric counts he vs she only. Texts using
  "they" are excluded, not counted as neutral. Some of the DPO models'
  "they" texts may be a reasonable outcome that GAP cannot credit.
- **Language drift.** Fine-tuned models sometimes switch to Chinese
  mid-text (4.3% at step 100 vs 0.4% at base). The usable filter does
  not exclude such texts.
- **The aggregate hides per-occupation overshoot** (teacher 0.79 → 0.13).
  GAP rewards the averages crossing, not each occupation reaching parity.
- **Energy figures** are GPU-only NVML readings from one run per model
  on shared cloud hardware.
- Long-form transfer, WinoBias/BOLD and a larger model were planned and
  **not done**.

## Run 2 (fixed 2026-09-30, before it was run)

Run 2 changes the training and nothing else. Same 446 pairs, same
evaluation, same seed, same selection rule. The new settings were chosen
in response to the quality collapse, not to any GAP value. There are two
variants that differ only in learning rate: 1e-5 is the cautious one,
3e-5 moves the model further. **Both are reported**, each with its own
selection under the unchanged rule. Neither is dropped because of its
result.

| | run 1 | run 2a | run 2b |
|---|---|---|---|
| loss | DPO (sigmoid) | DPO + NLL on chosen, weight 1.0 | same as 2a |
| learning rate | 5e-5 | 1e-5 | 3e-5 |
| generation monitor | none | every 10 steps: 24 train-cell prompts × 4 samples | same as 2a |
| early stop | none | usable rate > 10 points below step 0 at 2 consecutive checks | same as 2a |
| everything else | β 0.1, LoRA r 16, 3 epochs, effective batch 8, checkpoints every 10 | unchanged | unchanged |

The monitor uses only train occupations × train templates, never the
held-out cells. Its GPU time and energy are reported separately in
`resource_report.json` (`monitor`), so the training cost can be read
without it.

The evaluation reuses the base-model and base-4-bit results of run 1:
same code, seed and prompts. For each variant it evaluates checkpoints
10, 20, 40, 60, 100 and 153, or those that exist plus the last one saved
if training stopped early. Two variants mean two selections on the
train/train cell, so the selected train/train GAP is optimistic. The
held-out cells stay the clean test. If no checkpoint passes, that is the
result, and there is no third run before the application deadline.

### Run 2 outcome (from the training monitor)

Both variants **stopped early at step 20** under the pre-registered rule.
Monitor: 96 train-cell texts per check, same seed each time. Data in
`results/run2/`.

| step | 2a (lr 1e-5): usable / first person / GAP (n) | 2b (lr 3e-5): usable / first person / GAP (n) |
|---|---|---|
| 0 | 66% / 8% / +0.69 (63) | 66% / 8% / +0.69 (63) |
| 10 | 55% / 20% / +0.75 (53) | 41% / 36% / +0.68 (39) |
| 20 | 44% / 40% / +0.54 (42) | 38% / 26% / **+0.15 (36)** |

- The texts drifted toward first person again, faster at the higher
  learning rate.
- **Unlike run 1**, the chosen texts' log-probability *rose* on the
  held-out pairs (2b: −171 → −168), so the extra fine-tuning term did
  its job in aggregate. Still, generations changed.
- 2b at step 20 shows the first large GAP drop (+0.69 → +0.15). It rests
  on 36 texts (CI roughly ±0.3), at a checkpoint that fails the quality
  guard, so it is a hint, not a result.
- The monitor cost more than the training: about 12 of 15 min and 11.6
  of 14.9 Wh per variant (reported separately).

**Hypothesis (lesson #16), not yet tested.** The pair contrast starts at
the very first token, the name. Rejected texts open with the model's few
favourite names. Chosen texts open with names drawn uniformly from
21–44 (a lesson #12 fix). Pushing down a few frequent opening tokens
and spreading the push-up over many rare ones frees probability at the
first token, and "I" / "As" / "The" can absorb it. The fine-tuning term
does not hold this back at weight 1: it is a per-token mean (about 1/100
per token), while the DPO term acts on the sequence sum (β = 0.1 per
token). A test would compare first-token distributions before and after
training, or rerun with the name tokens masked out of the DPO loss.

## Run 3 (fixed 2026-09-30, after run 2 stopped, before run 3 was run)

Plain supervised fine-tuning on the 402 chosen (gender-swapped) texts:
counterfactual data augmentation, with no DPO term (`--sft-only`). It
only raises the likelihood of in-distribution third-person texts, so
the first-token effect hypothesised above has no DPO push-down to act on. Everything else is as in
run 2: lr 5e-5 (run 1's; stable for LoRA), 3 epochs, the same monitor
and early-stop rule, the same evaluation, selection rule and quality
guard. This run was decided **after** runs 1 and 2 failed, and it is
reported as such. The held-out occupations and templates were never
used for any of these choices.

## Next steps

1. **Settle the quantization interaction** (+0.15, CI includes 0): more
   samples per prompt, a merged-adapter 4-bit model (the realistic
   deployment case), and GPTQ / 8-bit next to NF4.
2. **DPO done the standard way:** start DPO from the run-3 checkpoint
   (SFT → DPO), and test the first-token hypothesis with same-name pairs
   ("Alex… his" vs "Alex… her").
3. Fix the language drift (a language filter, or a Chinese-token check
   in the usable rule), per-occupation overshoot control, a second seed,
   a second model size.
4. Report bias reduction per joule (performance-per-resource) instead
   of bias and energy side by side.

## Reproduce

Everything runs as committed Kaggle notebooks (details in
`kaggle_setup.md`). Each stage takes the previous notebook's output as
input.

```bash
pip install -r requirements.txt && pip uninstall -y torchao && python -m pytest tests/ -q
python src/generate_dataset.py --template-set short --samples-per-prompt 24
python src/build_pairs.py && python src/review_sample.py --sample-size 25
CUDA_VISIBLE_DEVICES=0 python src/train_dpo.py                                 # 24 min
python src/evaluate.py --run-dir runs/dpo_qwen0.5b --raw data/short/raw_completions.jsonl --out-dir results/eval   # ~2.5 h
python src/analyze_eval.py --eval-dir results/eval --out results/analysis      # run 1, CPU, ~1 min
python src/analyze_finetune.py --eval-dir results/eval3 --out results/analysis3 # run 3, CPU
```

`results/` holds the evaluation outputs from the run reported above
(completions, metrics, training log, pairs), so `analyze_eval.py` can be
run without a GPU.

## Iteration history / lessons learned (read this before changing the pipeline)

Several rounds of small-batch review turned up real problems, each fixed in
code. Know these before you re-derive them:

1. **Refusals / "as an AI language model" disclaimers** (~60% of early
   completions) — fixed with a system message in `generate_dataset.py`
   instructing direct, non-refusing creative responses.
2. **Model defaults to "they" or first-person, avoiding any gendered
   pronoun** — worst on the "typical day" and "dialogue" templates. Fixed by
   explicitly demanding third-person narration and banning "I"/"my".
3. **A real, named public figure appeared** (Mark Zuckerberg, for a "CEO"
   prompt) — fixed with an explicit "fictional name, not a real well-known
   person" instruction *and* a `REAL_PERSON_BLOCKLIST` safety net (now in
   `text_utils.py`) (instructions alone aren't reliable enough).
4. **"a engineer" / "a electrician"** — grammatical article bug, fixed with
   `article_for()` in `occupations.py`.
5. **Degenerate pairs**: when no completion in a group was genuinely
   counter-stereotypical, the script was pairing "less stereotyped" against
   "more stereotyped" and calling it a preference pair — this teaches
   nothing about bias. Manual review found this in ~1 in 5 groups. Fixed:
   `label_completions.py` now requires the chosen completion's score to be
   **strictly positive** (genuinely counter-stereotypical), not just higher
   than the rejected one.
6. **Pronoun misattribution (KNOWN, UNRESOLVED LIMITATION)**: the pronoun
   heuristic counts every he/she in a completion without knowing which
   character it refers to. In scenes with a colleague or visitor, a pronoun
   belonging to that secondary character can get wrongly credited to the
   target occupation. Mitigated for the "colleague" template (colleague must
   stay unnamed and ungendered) and the "walked into the room" template
   (banned second characters/narrators outright), but this is a heuristic
   limitation, not something prompt engineering fully closes. Now ALSO
   mitigated in code: completions with both he- and she-family pronouns
   are excluded (see #7). **Every manual spot-check pass must still check
   this**: a single-gender completion can have pronouns that belong to
   someone else. Record the verdicts in `data/manual_review.csv`.
7. **Code review, 2026-09-23 (before the first full post-fix run):**
   - The "genuine contrast" filter (#5) only required `best_score > 0`, so
     two counter-stereotypical completions (+5 vs +1) still formed a pair:
     the same degenerate pair, mirrored. Fixed by pairing by *class*
     (counter vs stereo), never by score difference.
   - The score `counter - stereo` let a mixed completion ("5 she + 4 he")
     count as counter-stereotypical, which is precisely the misattribution
     case of #6. Fixed: mixed completions are excluded.
   - "Balanced" occupations scored `-(m+f)`, always <= 0, so after #5 they
     could never produce pairs, silently. Now explicit: they are a
     held-out control group.
   - `max_new_tokens=120` truncated stories mid-sentence and nothing
     recorded it; DPO on truncated text teaches abrupt endings. Now 320,
     truncation is recorded and truncated samples are excluded.
   - Pairs stored the user prompt without the system message used at
     generation: at training time Qwen's default system prompt would have
     been injected, so policy/reference log-probs would be computed on a
     context that never produced the data. Pairs now store the full
     message list (TRL conversational format).
   - Refusal residue and first-person narration (outside dialogue) are now
     filtered in code, not only by prompt instructions.
   - No seed; `review_sample.py` referenced in docs but missing from the
     repo; scripts at the repo root while docs said `src/`. All fixed.
   (That labelling pipeline, `legacy/label_completions.py`, was retired in lesson #9; its checks live on in `text_utils.py` and `tests/test_text_utils.py`.)
8. **Generation length and dtype (Kaggle smoke tests, 2026-09-23):** at
   `max_new_tokens=320`, 5/6 stories were truncated. Measured instead of
   guessed: 40 samples with a 1024 cap, 0 truncated, median 379 / p95 679 /
   max 718 tokens -> default cap set to 800, templates unchanged. Same run
   showed `torch.cuda.is_bf16_supported()` returns True on a T4 (it counts
   emulated bf16); dtype is now chosen by compute capability (>= 8.0 ->
   bf16, else float32).

9. **First full long-form run (2026-09-23, Kaggle T4): natural pairs abandoned.**
   1000 completions, 1% truncated, but only **25 pairs** (21 from
   male-coded occupations, 4 from female-coded), for three reasons:
   - *Yield/asymmetry:* in 49/76 train groups the model never wrote a
     counter-stereotypical completion; for female-coded occupations it
     wrote "he" in ~5 of 202 usable completions. More sampling can't fix it.
   - *Label noise:* reading the 20-pair sample, 7 had a serious attribution
     problem and 2 were outright invalid (the rejected text's pronouns
     never referred to the occupation-holder). All 4 sampled pairs from
     the colleague template were affected; the model also ignored the
     "only one person" instruction. Counting pronouns cannot tell who they
     refer to in long multi-character stories.
   - *Wrong direction for this model:* "scientist" (labelled male-coded)
     was 81% female among single-gender completions; the model also skews
     female overall (68% on balanced occupations; "Sarah"/"Emily"
     dominate). Pairs defined by human stereotype labels would push some
     occupations *away* from parity.
   Baseline from this run (share of she/her among single-gender
   completions): female-coded 96%, balanced 68%, male-coded 31%.
   **Redesign:** short single-character templates (pronoun labels become
   reliable); counterfactual pairs (yield and symmetry solved, contrast
   isolated to gender); training direction decided per occupation by the
   model's measured bias on a separate calibration half, with a rule fixed
   before the first short-form run (`build_pairs.py`); long-form templates
   kept only to test transfer. Found while building it: spaCy tags
   sentence-initial "Emily" as an adverb and "Hope filled the room" as a
   proper noun, so names are detected with a names corpus plus POS/NER
   guards (regression tests in `tests/test_text_utils.py`).

10. **Short-form smoke test (48 samples, engineer + CEO): the prompt's
    pronoun order decided the gender.** Templates alternated "he or she"
    and "she or he" to balance order effects. Instead, the model used the
    first-mentioned pronoun almost every time: engineer 12/12 male vs 12/12
    female, CEO 11/12 male vs 10/10 female, depending only on the order.
    Any measurement with those prompts would have measured word order, not
    occupational bias -- and the alternation would have hidden the real
    effect. Fix: prompts never name pronouns (enforced by
    `tests/test_occupations.py`). Also found: 27% of texts had no name
    (fix: "Begin the text with the first name"); a quoted slogan ("Always
    Happy") detected as a name (fix: ignore quoted text); "Mr. Johnson"
    discarded (fix: gender from the title, swap Mr <-> Ms, keep surname);
    single-pronoun texts discarded (min pronoun count now 1, safe because
    the name must agree with the pronouns); length p95 191 / max 271 tokens,
    cap set to 300. Not fixable by filters: a named character who is *not*
    the occupation-holder ("Mary ... the CEO who had made all the
    difference"); the manual audit measures how often this happens.
    Methodological note for the write-up: for this 0.5B model a surface cue
    in the prompt dominates the occupational association, so bias
    measurements are only meaningful with prompts that don't mention the
    measured attribute.

11. **Short-form smoke test #2 (neutral templates, 48 samples, engineer +
    CEO):** usable 77% (was 40%), 0 texts without a name (was 27%), 0
    "they"-only texts, 0 truncated at 300 tokens (max 237). The per-template
    gender flip is gone. Engineer 4/19 female, CEO 5/18 female among usable
    texts (both male-leaning, n too small to compare them). All 4 engineer
    texts for held-out template t5 ("walking home") were female vs 0/15 in
    the other templates -- possibly chance, possibly a scene effect; the
    held-out template split measures exactly this. Reading all 37 usable
    texts: pronouns referred to the named character in every one. One wrong
    exclusion fixed: a quoted nickname ('Elisabeth "Betty" Rogers') split the
    name and "Rogers" counted as a second person.

12. **First full short-form run (2026-09-27, Kaggle T4, 2400 completions).**
    Usable 56% (every template 44-62%), 205 pairs (94 male->female, 111
    female->male). Baseline GAP +0.65, the same on held-out occupations and
    held-out templates. Calibration disagreed with the stereotype labels as
    expected (scientist, surgeon, pharmacist skew female). Reading the 25-pair
    sample found six problems in the swap, all fixed with regression tests:
    - **Replacement names:** "John" was 79/111 male replacements (71%), because
      names were sampled by model frequency: DPO could learn "prefer John".
      Now uniform over names the model produced at least twice, and
      `build_pairs.py` warns if one name is over-represented.
    - **Surnames as first names:** "Mrs. Thompson" put "Thompson" in the
      female name pool (pair 55: John -> Thompson). The pool now keeps only
      corpus first names of that gender.
    - **Middle names:** "Ms. Sarah Jane Smith" -> "Mr. John Jane Smith"
      (pair 136). Middle names are dropped.
    - **Other people's titles:** "Dear Miss Jones" -> "Dear Mr Jones" (pair
      129). Titles are swapped only before the protagonist's name/surname.
    - **Generic and kinship nouns:** "a woman in a suit" (another person)
      became "a man" (pair 130); "his son" would become "her daughter". The
      noun may denote the protagonist or someone else and no rule tells them
      apart, so texts with man/woman/son/sir... are rejected (`person_noun`).
      Role nouns (repairman, waitress) are still swapped.
    - **"her" after double-object verbs:** "allowed her time" -> "allowed his
      time" (pair 124); spaCy tags it possessive. Rejected
      (`ambiguous_her`) after verbs like give/allow/offer; the first list
      also held "leave", which rejected the clearly possessive "leave her
      classroom" (pair 149), so it was narrowed.
    Also: "Mrs. Smith ... her" was excluded as a name-pronoun mismatch because
    the corpus lists Smith (and Patel) as male first names; a gendered title
    followed by one word is now a surname with the title's gender. A real
    person got through (pair 79: Michael Jackson, in a firefighter text);
    blocklist extended. Re-running the fixed swap on the 25 sampled texts:
    22 swapped, 3 rejected (1 person_noun, 2 ambiguous_her before the verb
    list was narrowed). Rejections cost ~5-10% of pairs, so the next run
    uses 24 samples per prompt instead of 16: calibration still uses sample
    ids 0-7 (rule unchanged), and the pair-source half doubles (8 -> 16).
    Considered and NOT adopted: requiring the occupation word in the text.
    In the sample it would have dropped good texts ("John works at the auto
    repair shop" for mechanic; "software developer" for programmer).

13. **Second full short-form run (2026-09-28, Kaggle T4, 3600 completions,
    24 per prompt).** Usable 57%, truncated 3.8%, baseline GAP +0.68 (first
    run +0.65). 446 pairs (211 male->female, 235 female->male); the new
    rejections cost 24 pairs (19 person_noun, 5 ambiguous_her). Checking
    ALL pairs, not only the sample, found four more problems, fixed with
    regression tests and the pairs rebuilt from the same completions:
    - **"Miss Alice" -> "Mr Alice"** (5 pairs): lesson #12's rule "title +
      one word = surname" also caught first names. Now a word after a
      title is a surname only if it is in a list of common surnames
      (`COMMON_SURNAMES`, minus those that are also common first names) or
      is not a first name of the title's gender.
    - **"Sarah" -> "Smith"** (28 pairs): "Mr. Smith" put Smith (a male
      first name in the corpus) in the male replacement pool. Surnames are
      now excluded from the pool.
    - **Possessive "her" tagged as object** (~12 pairs): "worked her last
      hour" -> "worked him last hour", "checks her watch" -> "checks him
      watch". spaCy tags these "her" as PRP. Fix: "her" followed by a noun
      (after optional modifiers) is possessive. All 83 remaining "him" in
      chosen texts were read: all objects.
    - **Blocklist false positives** (37 texts): substring matching found
      "jack ma" in "Mechanic Jack made"; "John Doe"/"Jane Doe" are
      placeholders, not real people. Now whole-word matching, placeholders
      removed.
    Reading 14 pairs flagged for an other-person noun: in all 14 the
    pronouns referred to the named character. Remaining known imperfection:
    unisex corpus names are kept (Alex, Tommy, Sam: 24 pairs), which is
    correct for the swap but "Tommy" reads as male to most readers.
    **Calibration is noisy at this n:** with 12-26 usable calibration texts
    per occupation, two decisions flipped between runs (CEO 0.48 -> 0.27,
    now targeted; scientist 0.72 -> 0.54, now near parity). The rule was
    not changed; the decision is taken from this run's data. With n ~ 20
    the standard error of p is ~0.1, so the 0.20 margin is only ~2 SE: a
    limitation for the write-up.

14. **DPO training run (2026-09-29, Kaggle T4, `train_dpo.py` defaults).**
    446 pairs -> 402 train / 44 held-out (stratified by occupation). LoRA
    r=16 on all linear layers: 8.8M trainable parameters (1.75% of 503M),
    fp32, beta 0.1, lr 5e-5 cosine, effective batch 8, 3 epochs = 153 steps.
    Cost: 23.9 min, peak GPU memory 7.0 GB, **GPU energy 26.0 Wh (NVML)**.
    Carbontracker reports 41.2 Wh = 26.0 Wh x its default PUE of 1.58, so
    the two measurements agree; 15.8 gCO2eq at 384 g/kWh. Held-out pairs:
    reward accuracy 0.93-0.98 from step 10 (the contrast is a few gender
    words, so accuracy says little), margin 0.35 -> 4.4, loss 0.54 -> 0.095,
    no divergence. **Warning sign:** the log-probability of the CHOSEN texts
    also fell (-196 -> -252 on train batches) and entropy rose (2.2 -> 2.6):
    DPO pushed both texts down, rejected faster (a known DPO behaviour).
    Later checkpoints may therefore write worse text, so the evaluation
    measures quality (usable rate, perplexity) next to bias, and the
    selection rule has a quality guard.

15. **Evaluation run (2026-09-30, Kaggle T4): DPO collapsed; no checkpoint
    selected.** The usable rate went 57% (base) → 27% (step 10) → 4%
    (steps 20–60) → 0% (steps 100, 153). Texts with a gendered pronoun
    went 94% → 70% → 12% → 0%. The model first switched to first-person
    narration, then to one-line greetings, then to fragments, some in
    non-Latin script. The held-out reward accuracy (0.91–0.98) and
    reference perplexity (5.45 → 5.68 at step 10) did not show it. The
    pre-registered usable-rate guard did, and it rejected every
    checkpoint, so no 4-bit DPO model was evaluated. Diagnosis:
    likelihood displacement on near-identical pairs (chosen and rejected
    differ only in gendered words). The chosen log-probability falling
    in lesson #14 was the early sign. Lessons: (a) monitor generated
    text during DPO training, not only reward accuracy; (b) with
    minimal-edit pairs, add an NLL term on chosen or use a lower lr and
    fewer steps. The first attempt to run the evaluation failed after
    30 s because the training notebook's output was not attached as
    input (`/kaggle/input` glob empty): attach the committed version
    whose Output tab shows `runs/dpo_qwen0.5b/checkpoint-*`.
    The base-vs-NF4 comparison is unaffected. It shows no detectable
    GAP change (−0.06 [−0.22, +0.07]), 12 points fewer usable texts
    (mostly mixed he/she texts), and 29% more GPU energy per token on
    the T4.

## Evaluation design decisions (fixed before any post-fix data)

Everything in this section was fixed before the evaluation results. The
analysis choices in `analyze_eval.py` (occupation-cluster bootstrap,
30-texts-per-side threshold, parity distance, pronoun diagnostics) were
added afterwards and are exploratory.


- **Held-out occupations** (`split="heldout"`): pilot, electrician,
  hairdresser, librarian never enter training. Balanced occupations are a
  control group, also never trained on. Effects are reported separately for
  train / held-out / control: an effect only on train occupations is
  memorisation, not debiasing.
- **Held-out templates** (short set, templates 4-5) never produce pairs,
  so results are reported on the grid {train, held-out occupations} x
  {train, held-out templates}, plus long-form stories (transfer).
- **Headline metric:** GAP = p_female(female-coded) - p_female(male-coded)
  among single-gender completions. Overall gender skew (p_female on the
  balanced control group) is reported separately: shrinking the gap is the
  goal, moving the overall skew is a side effect to report, not hide.
- **Overshoot is a failure, not a success.** Trained long enough, DPO will
  not stop at parity: it will make nurses male and engineers female. A
  reversed gap is a failure. Checkpoints are evaluated along training.
- **Checkpoint selection (fixed 2026-09-29, before any post-training
  generation was looked at):** fresh generations (seed 1234, 8 per prompt,
  same sampling settings) from the base model and checkpoints 10, 20, 40,
  60, 100, 153. Among checkpoints whose usable rate is at most 10 points
  below the base model's, pick the smallest |GAP| on train occupations x
  train templates; ties go to the earlier step. Held-out cells are never
  used for selection. The selected checkpoint and the base model are also
  evaluated in 4-bit (NF4). Reported for every model: GAP with bootstrap
  95% CIs on the four cells, control-group p_female, usable rate and
  exclusion reasons, perplexity on 200 base-model texts for held-out
  occupations and on WikiText-2, and generation speed and GPU energy per
  1k tokens.
- **Training direction comes from the model, not from stereotype labels**
  (pre-registered rule in `build_pairs.py`: |p_female - 0.5| >= 0.20 on the
  calibration half, n >= 12). Stereotype labels only group the results.

## Repo structure

```
dpo-bias-project/
├── README.md
├── requirements.txt
├── kaggle_setup.md
├── resources/names/          # Kantrowitz names corpus (see credits)
├── src/
│   ├── occupations.py        # occupations, splits, short + longform templates
│   ├── generate_dataset.py   # samples completions (--template-set)
│   ├── text_utils.py         # filters, pronoun counts, name detection
│   ├── counterfactual.py     # gender swap for single-character texts
│   ├── build_pairs.py        # calibration rule + counterfactual DPO pairs
│   ├── review_sample.py      # funnel, calibration, baseline, audit sample
│   ├── train_dpo.py          # DPO + LoRA, checkpoints, energy measurement
│   ├── evaluate.py           # bias / quality / cost of base, checkpoints, 4-bit (GPU)
│   ├── analyze_eval.py       # cluster CIs, diagnostics, figures (CPU)
│   └── analyze_finetune.py   # paired comparisons for a run that passed (CPU)
├── results/                  # outputs of the reported run
│   ├── eval/, eval3/         # run 1 / run 3: <model>/completions.jsonl, metrics.json, summary
│   ├── analysis3/            # run 3 tables, examples, fig4-6
│   ├── run2/, run3/          # training reports + monitor logs
│   ├── analysis/             # analysis.json, tables.md, fig1-3
│   ├── resource_report.json  # training cost + full training log
│   ├── split.json, dpo_pairs.jsonl
├── tests/                    # unit + regression tests (pytest)
├── legacy/                   # first labelling pipeline (lessons #1-8), not used
└── data/<template-set>/      # created by the scripts
```

## Credits and references

- Names corpus: Mark Kantrowitz, Names Corpus v1.3 (with additions by Bill
  Ross), redistributed in `resources/names/` with its README, as its
  license requires.
- Energy measurement: Anthony, Kanding & Selvan (2020), "Carbontracker:
  Tracking and Predicting the Carbon Footprint of Training Deep Learning
  Models".
- Counterfactual data augmentation: Lu et al. (2020), "Gender Bias in
  Neural Natural Language Processing"; Zmigrod et al. (2019),
  "Counterfactual Data Augmentation for Mitigating Gender Stereotypes in
  Languages with Rich Morphology".
- DPO failure mode: Pal et al. (2024), "Smaug: Fixing Failure Modes of
  Preference Optimisation with DPO-Positive"; Razin et al. (2025),
  "Unintentional Unalignment: Likelihood Displacement in Direct
  Preference Optimization" (ICLR).
- Occupation groupings follow Zhao et al. (2018), WinoBias. They are
  qualitative labels, used only to group results; they are not labour
  statistics (see `src/occupations.py`).
