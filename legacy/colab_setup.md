# Running on Colab

1. Runtime → Change runtime type → GPU (T4 is fine to start).
2. Clone the repo (push it to GitHub first, private repo is fine for now):

```python
!git clone https://github.com/<your-username>/dpo-bias-project.git
%cd dpo-bias-project
!pip install -q -r requirements.txt
```

3. Run everything **from the repo root** (do not `%cd src`: output paths
   are resolved relative to the repo, and the commands below assume it).

```python
!python -m pytest tests/ -q
!python src/generate_dataset.py --limit-prompts 3 --samples-per-prompt 2
!head -c 1500 data/raw_completions.jsonl
```

Check the smoke-test completions are real stories that end (look at the
`truncated` field), then run the full generation:

```python
!python src/generate_dataset.py --samples-per-prompt 8
```

25 occupations x 5 templates = 125 prompts; the 8 samples per prompt are
generated in a single batched call, so this should take roughly 15-30
minutes on a T4. If the final printout says >15% truncated, raise
`--max-new-tokens` and regenerate.

4. Label and review:

```python
!python src/label_completions.py
!python src/review_sample.py --sample-size 20
```

Read all five sections of the review output, then fill in
`data/manual_review.csv` (download it, or edit it in Colab). Save
`data/` to Drive or commit it: Colab runtimes are wiped on disconnect.

`train_dpo.py` comes only after the dataset has passed this review.
