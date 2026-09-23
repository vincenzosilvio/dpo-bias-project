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
- [x] Review tooling (`src/review_sample.py`) and unit tests for the pairing
      logic (`tests/test_labeling.py`)
- [x] Held-out split fixed in advance (`split` field in `occupations.py`)
- [ ] **Full dataset run with the current code: NOT done yet.** Everything
      before 2026-09-23 was generated/labelled with logic that had the bugs
      in lesson #7 -- no earlier pair count is valid.
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
   Each pairing bug above has a regression test in `tests/test_labeling.py`.
8. **Generation length and dtype (Kaggle smoke tests, 2026-09-23):** at
   `max_new_tokens=320`, 5/6 stories were truncated. Measured instead of
   guessed: 40 samples with a 1024 cap, 0 truncated, median 379 / p95 679 /
   max 718 tokens -> default cap set to 800, templates unchanged. Same run
   showed `torch.cuda.is_bf16_supported()` returns True on a T4 (it counts
   emulated bf16); dtype is now chosen by compute capability (>= 8.0 ->
   bf16, else float32).

## Evaluation design decisions (fixed before any post-fix data)

- **Held-out occupations** (`split="heldout"`): pilot, electrician,
  hairdresser, librarian never enter training. Balanced occupations are a
  control group, also never trained on. Effects are reported separately for
  train / held-out / control: an effect only on train occupations is
  memorisation, not debiasing.
- **Overshoot is a failure, not a success.** Every pair pushes towards the
  counter-stereotypical gender. Trained long enough, DPO will not stop at
  parity: it will make nurses male and engineers female. The target is a
  *smaller gap* between male-coded and female-coded occupations in P(female
  pronoun), not a reversed one. The generation-bias table printed by
  `review_sample.py` (section 2) is the baseline of this metric.

## Next concrete step

From the repo root, on Colab:
```
python -m pytest tests/ -q
python src/generate_dataset.py --limit-prompts 3 --samples-per-prompt 2   # smoke test
python src/generate_dataset.py --samples-per-prompt 8
python src/label_completions.py
python src/review_sample.py --sample-size 20
```
Read the full `review_sample.py` output, fill `data/manual_review.csv`,
and only then write `train_dpo.py`.

## Repo structure

```
dpo-bias-project/
├── README.md
├── requirements.txt
├── colab_setup.md
├── src/
    ├── occupations.py        # occupation list + prompt templates
    ├── generate_dataset.py   # generates raw candidate completions
    ├── label_completions.py  # filters, classifies, pairs completions for DPO
    └── review_sample.py      # funnel, baseline bias table, flags, sample
tests/
    └── test_labeling.py      # regression tests for past pairing bugs
data/                         # created by the scripts
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
