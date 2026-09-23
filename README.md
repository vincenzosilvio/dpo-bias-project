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
- [x] Occupation list and prompt templates drafted (`src/occupations.py`) — see
      "Iteration history" below, this went through several revisions
- [x] Candidate generation script (`src/generate_dataset.py`)
- [x] Completion labeling / pairing script (`src/label_completions.py`)
- [ ] **A full dataset run with the current (post-fix) templates and labeling
      logic has NOT been done yet.** The last full run (samples-per-prompt=8)
      used templates from before the "no second character" and "colleague
      stays ungendered" fixes, and labeling from before the genuine-contrast
      filter. **Re-run `generate_dataset.py` + `label_completions.py` before
      trusting any pair counts or moving to training.**
- [ ] Baseline evaluation on WinoBias / BOLD — not yet run
- [ ] DPO training script — not yet written
- [ ] Post-training evaluation
- [ ] Write-up

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
   person" instruction *and* a `REAL_PERSON_BLOCKLIST` safety net in
   `label_completions.py` (instructions alone aren't reliable enough).
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
   limitation, not something prompt engineering fully closes. **Every manual
   spot-check pass must include checking this specifically** — read the
   sentence the pronoun is in, confirm it's about the named occupation-holder
   and not someone else in the scene.

## Next concrete step

Re-run the full pipeline with the current code:
```
python src/generate_dataset.py --samples-per-prompt 8
python src/label_completions.py
python src/review_sample.py --sample-size 20
```
`review_sample.py` prints the review (summary stats, auto-flagged
mixed-pronoun pairs, and a random sample) directly in the Colab output --
read it there. Only then move to writing `train_dpo.py`.

## Repo structure

```
dpo-bias-project/
├── README.md
├── requirements.txt
├── colab_setup.md
└── src/
    ├── occupations.py        # occupation list + prompt templates
    ├── generate_dataset.py   # generates raw candidate completions
    ├── label_completions.py  # scores + pairs completions for DPO
    └── review_sample.py      # prints a flagged/sampled review of the pairs
                               # directly in Colab -- run this instead of
                               # pasting dpo_pairs.jsonl elsewhere for review
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
