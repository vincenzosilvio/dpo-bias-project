# Running on Colab

1. Runtime → Change runtime type → GPU (T4 is fine to start).
2. Clone the repo (push it to GitHub first, private repo is fine for now):

```python
!git clone https://github.com/<your-username>/dpo-bias-project.git
%cd dpo-bias-project
!pip install -q -r requirements.txt
```

3. Smoke test on a handful of prompts before committing to the full run
   (catches bugs/crashes fast instead of after 40 minutes of generation):

```python
%cd src
!python generate_dataset.py --limit-prompts 5 --samples-per-prompt 2
```

Check `data/raw_completions.jsonl` looks sane (real completions, not empty
strings or errors), then run the full generation:

```python
!python generate_dataset.py --samples-per-prompt 4
```

With 25 occupations x 5 templates x 4 samples = 500 generations, at ~120
new tokens each, this should take well under an hour on a T4.

4. Build the preference pairs:

```python
!python label_completions.py
```

Read the printed warning if pair count is low — if so, increase
`--samples-per-prompt` and regenerate, or loosen `--min-pronoun-count`.

5. **Before training anything**: pull `data/dpo_pairs.jsonl` down and
   actually read 20-30 pairs by hand. This is the step most tutorials skip
   and it's the one that will save the project — if the pronoun heuristic
   is picking up garbage (e.g. pronouns referring to someone other than the
   occupation-holder), you want to know now, not after a training run.

Next script (`train_dpo.py`, using `trl.DPOTrainer`) comes once you've
confirmed the dataset looks right — no point tuning hyperparameters on a
noisy dataset.
