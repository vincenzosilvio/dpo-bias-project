# DPO Fine-tuning for Gender-Occupation Bias Reduction

## Hypothesis

Direct Preference Optimization (DPO) on a purpose-built preference dataset
reduces stereotyped gender-occupation associations in a small language
model's generations, while preserving general helpfulness — and the effect
scales in a measurable, non-trivial way between a 0.5B and a 1.5B parameter
model.

Three falsifiable claims:
1. DPO measurably reduces gender-occupation stereotype scores vs. the base model.
2. General helpfulness (measured on a standard instruction-following benchmark)
   does not degrade beyond an acceptable threshold.
3. The size of the effect differs between the 0.5B and 1.5B model in a way we
   can characterize (bigger isn't automatically "more debiased").

## Method overview

1. **Baseline measurement** — evaluate the base model (Qwen2.5-0.5B-Instruct,
   later Qwen2.5-1.5B-Instruct) on:
   - **WinoBias** (Zhao et al., 2018) — coreference resolution, pro-stereotype
     vs. anti-stereotype accuracy.
   - **BOLD** (gender subset, Dhamala et al., 2021) — open-ended generation,
     scored with a regard classifier.
2. **Preference dataset construction** — generate multiple completions per
   occupation-prompt from the base model, score them, and pair
   (chosen = lower-stereotype / higher-regard, rejected = higher-stereotype)
   while controlling for comparable fluency/length so DPO learns "less
   stereotyped," not "better written."
3. **DPO training** — `trl.DPOTrainer`, logged (loss, reward margin, KL to
   reference model).
4. **Post-training evaluation** — same benchmarks, before/after comparison,
   plus a general helpfulness check (e.g. a subset of MT-Bench or AlpacaEval)
   to catch regressions.
5. **Write-up** — results, trade-offs, and an explicit limitations section
   (any "non-stereotyped" label embeds a value judgement — this gets
   discussed, not hidden).

## Status

- [x] Scope and hypothesis defined
- [x] Occupation list and prompt templates drafted (`src/occupations.py`)
- [x] Candidate generation script (`src/generate_dataset.py`)
- [x] Completion labeling / pairing script (`src/label_completions.py`)
- [ ] Baseline evaluation on WinoBias / BOLD — not yet run
- [ ] DPO training script
- [ ] Post-training evaluation
- [ ] Write-up

## Repo structure

```
dpo-bias-project/
├── README.md
├── requirements.txt
├── colab_setup.md
└── src/
    ├── occupations.py        # occupation list + prompt templates
    ├── generate_dataset.py   # generates raw candidate completions
    └── label_completions.py  # scores + pairs completions for DPO
```

## Important note on the occupation data

`src/occupations.py` categorizes occupations as stereotypically
male-associated, female-associated, or balanced based on commonly cited
categorizations in the bias literature (e.g. Zhao et al. 2018, WinoBias).
**Before the final write-up, cross-check current gender-participation
percentages against the U.S. Bureau of Labor Statistics
[Labor Force Statistics, Table 11](https://www.bls.gov/cps/cpsaat11.htm)**
(or your country's equivalent, if you want an Italy/EU angle instead — worth
considering given your background) rather than relying on the qualitative
labels alone. This is flagged in code with a `TODO`.
