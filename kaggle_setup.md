# Running on Kaggle

## Notebook settings (right panel -> Session options)

- **Accelerator:** GPU T4 x2 or P100 (the scripts use one GPU).
- **Internet:** On (requires phone verification in your Kaggle account).
- Private repo: add a GitHub token under Add-ons -> Secrets with label
  `GITHUB_TOKEN`, and attach it to the notebook.

## Rule: never edit code on Kaggle

Change code locally, push, then `git pull` in the notebook. GitHub must
always hold the exact code that produced the data.

## Cells

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from kaggle_secrets import UserSecretsClient          # private repo only
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
url = f"https://{token}@github.com/<your-username>/dpo-bias-project.git"
!git clone -q {url} /kaggle/working/dpo-bias-project
%cd /kaggle/working/dpo-bias-project
!git log --oneline -1
```

```python
!pip install -q -r requirements.txt
!pip uninstall -y -q torchao   # Kaggle's torchao 0.10 makes peft 0.21 fail on import; unused here
!python -m pytest tests/ -q
```

Smoke test (interactive session): check `gen_config` shows the GPU and
`torch.float32`, that texts are two sentences, and the truncation rate.

```python
!python src/generate_dataset.py --template-set short --limit-prompts 6 --samples-per-prompt 4
!head -c 1500 data/short/raw_completions.jsonl
```

## Full run: use Save Version -> Save & Run All

Interactive sessions lose `/kaggle/working` when they end (this cost us
the first long-form dataset). For the full run, make the notebook contain
only these cells in order -- setup, install, then:

```python
!python src/generate_dataset.py --template-set short --samples-per-prompt 24
!python src/build_pairs.py
!python src/review_sample.py --sample-size 25
!zip -qr /kaggle/working/data_short.zip data/short
```

then **Save Version -> Save & Run All (Commit)**. It runs in the
background (you can close the browser) and everything in
`/kaggle/working` is kept with that version: open the version, go to
Output, download `data_short.zip`, and read the logs for the printed
review.

## Training run

Also a committed run. Give it the pairs from the data run: in the training
notebook, **Add Input -> Your Work -> the data notebook**; its output then
appears read-only under `/kaggle/input/<data-notebook-name>/`.

```python
# after the setup lines (clone, os.chdir, pip install, pytest)
!unzip -q /kaggle/input/<data-notebook-name>/data_short.zip -d .
!python src/build_pairs.py      # rebuild pairs with the current code (~3 min)
!CUDA_VISIBLE_DEVICES=0 python src/train_dpo.py
!zip -qr /kaggle/working/run_small.zip runs/*/resource_report.json runs/*/split.json data/short/pair_stats.json
```

Rebuilding the pairs in the training run keeps the rule "GitHub holds the
exact code that produced the data": the completions are unchanged, only
the pairing code is newer.

`CUDA_VISIBLE_DEVICES=0` matters: with both T4s visible, Trainer uses
DataParallel and the energy figures cover two GPUs. Download only `run_small.zip`
(`resource_report.json`: wall time, GPU energy, peak memory, full training
log). The checkpoints stay on Kaggle: the evaluation notebook reads them by
adding this notebook as input.

## Evaluation run

A third committed notebook. **Add Input -> Your Work -> the training
notebook**: its output holds the checkpoints (`runs/dpo_qwen0.5b/`) and the
completions (`data/short/raw_completions.jsonl`). Setup lines as above
(clone, `pip install`, `pip uninstall -y -q torchao`, pytest), then:

```python
import glob
run = os.path.dirname(glob.glob("/kaggle/input/**/runs/dpo_qwen0.5b/resource_report.json", recursive=True)[0])
raw = glob.glob("/kaggle/input/**/data/short/raw_completions.jsonl", recursive=True)[0]
print(run, raw)
!python src/evaluate.py --run-dir {run} --raw {raw} --out-dir runs/eval
!zip -qr /kaggle/working/eval.zip runs/eval
```

Expect about 3-4 hours: 7 models in full precision and 2 in 4-bit, 1200
generations each. Everything goes to `runs/eval/`: per model
`completions.jsonl` and `metrics.json`, plus `summary.md` (the results
table) and `summary.json`. `eval.zip` is small (texts and JSON only).
