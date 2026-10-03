# Development history and lessons learned

Linked from the [README](README.md). Read this before changing the pipeline.

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

16. **Runs 2 and 3 (2026-09-30).** DPO + NLL (lr 1e-5 and 3e-5) drifted
    toward first-person texts and was stopped at step 20 by the monitor
    in both variants. The hypothesis is a first-token effect on the name
    (see "Run 2 outcome"). Supervised fine-tuning on the chosen texts
    alone (run 3) kept quality and cut GAP from +0.61 to +0.04 at the
    selected step 100. Lesson: on minimal-edit counterfactual pairs,
    start from supervised fine-tuning, and watch generations, not reward
    accuracy.

17. **Manual review of the trained pairs (2026-10-01).** 25 sampled pairs:
    one character only 22/25, pronouns refer to the holder 24/25, swap
    correct 23/25, chosen fluent 24/25 (`results/manual_review/`). Two
    errors:
    - A patient who speaks had her pronoun swapped too. This is the
      residual multi-character risk: an unnamed second person is not
      caught by `analyze()`.
    - "**Dr. John Smith**" became "**Dr. Linda **". spaCy tagged the
      markdown `*` as a proper noun, so the name span became
      [John, Smith, *], and Smith looked like a middle name and was
      dropped. Fixed: tokens with no letter are never name parts (with a
      regression test). Re-running the fixed swap on all 446 trained
      pairs changes exactly one, pair 48. The reported runs used the old
      code.

