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
2. **Preference dataset construction** (redesigned in lesson #9) — the base
   model writes short, single-character texts about each occupation; a
   calibration half of the samples measures which gender the model
   defaults to per occupation; for occupations far from parity, each
   majority-gender text (rejected) is paired with its counterfactual
   gender swap (chosen). Chosen and rejected differ *only* in gendered
   words, so DPO learns the gender preference and nothing else.
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
- [x] Occupations, held-out split, templates (`src/occupations.py`)
- [x] Generation script (`src/generate_dataset.py`), short and longform sets
- [x] Long-form natural-pair pipeline: built, run once (2026-09-23), and
      **abandoned** as a source of training data (lesson #9). Long-form
      templates are kept for the transfer evaluation.
- [x] Short-form redesign: text analysis (`src/text_utils.py`),
      counterfactual swap (`src/counterfactual.py`), calibration + pairs
      (`src/build_pairs.py`), review (`src/review_sample.py`), 29 unit tests
- [x] Short-form smoke test #1 -> template fix (lesson #10)
- [ ] Short-form smoke test #2 with the neutral templates
- [ ] **First short-form run** — not done yet
- [ ] Manual audit of the short-form pairs (`data/short/manual_review.csv`)
- [ ] Baseline evaluation (generation bias on short/longform, WinoBias, BOLD)
- [ ] DPO training script
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

## Evaluation design decisions (fixed before any post-fix data)

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
- **Training direction comes from the model, not from stereotype labels**
  (pre-registered rule in `build_pairs.py`: |p_female - 0.5| >= 0.20 on the
  calibration half, n >= 12). Stereotype labels only group the results.

## Next concrete step

On Kaggle (see `kaggle_setup.md`), from the repo root:
```
python -m pytest tests/ -q
python src/generate_dataset.py --template-set short --limit-prompts 12 --samples-per-prompt 4   # smoke test
python src/generate_dataset.py --template-set short
python src/build_pairs.py
python src/review_sample.py --sample-size 25
```
Read all six sections, fill `data/short/manual_review.csv`, and only then
write `train_dpo.py`.

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
│   └── review_sample.py      # funnel, calibration, baseline, audit sample
├── tests/                    # unit + regression tests (pytest)
└── data/<template-set>/      # created by the scripts
```

## Credits

- Names corpus: Mark Kantrowitz, Names Corpus v1.3 (with additions by Bill
  Ross), redistributed in `resources/names/` with its README, as its
  license requires.
- Counterfactual data augmentation: Lu et al. (2020), "Gender Bias in
  Neural Natural Language Processing"; Zmigrod et al. (2019),
  "Counterfactual Data Augmentation for Mitigating Gender Stereotypes in
  Languages with Rich Morphology".

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
