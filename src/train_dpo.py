"""
DPO fine-tuning with LoRA on the counterfactual pairs (build_pairs.py).

What it does
  * loads data/short/dpo_pairs.jsonl (TRL conversational format: the FULL
    message list used at generation, so policy and reference log-probs are
    computed on the context that produced the data -- README, review #5)
  * holds out --eval-frac of the pairs, stratified by occupation, to monitor
    reward accuracy/margin on pairs the model never trains on
  * trains a LoRA adapter with trl.DPOTrainer. The reference model is the
    same base with the adapter disabled (no second copy in memory)
  * saves an adapter checkpoint every --save-steps: the checkpoint is chosen
    later by evaluate.py on the GAP metric (overshoot = failure, README), NOT
    by the DPO loss
  * measures the resource cost of training:
        - GPU energy from the NVML hardware counter (per GPU, Volta or newer;
          the T4 has it), wall time, peak GPU memory, trainable parameters
        - Carbontracker (Anthony, Kanding & Selvan 2020), GPU component only:
          on Kaggle/Colab the CPU energy counters (RAPL) are not readable
    -> <out>/resource_report.json, <out>/carbontracker/

Usage (repo root, ONE GPU -- on Kaggle's 2xT4 prefix CUDA_VISIBLE_DEVICES=0,
otherwise Trainer silently uses DataParallel over both and the energy
numbers double):
    CUDA_VISIBLE_DEVICES=0 python src/train_dpo.py
    CUDA_VISIBLE_DEVICES=0 python src/train_dpo.py --epochs 1 --save-steps 10   # quick check
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "short"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DATA / "dpo_pairs.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "dpo_qwen0.5b"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-frac", type=float, default=0.10)
    # DPO
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=512,
                    help="prompt + completion tokens; short completions are <= 300")
    # optimisation
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=10)
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    # resources
    ap.add_argument("--no-carbontracker", action="store_true")
    ap.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto",
                    help="auto: bf16 only on compute capability >= 8.0 (README, "
                         "T4 reports bf16 support but only emulates it)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pairs(path):
    return [json.loads(line) for line in open(path)]


def stratified_split(pairs, eval_frac, seed):
    """Per occupation, hold out round(eval_frac * n) pairs (at least 1 if n >= 5)."""
    rng = random.Random(seed)
    by_occ = defaultdict(list)
    for p in pairs:
        by_occ[p["occupation"]].append(p)
    train, evals = [], []
    for occ in sorted(by_occ):
        group = by_occ[occ][:]
        rng.shuffle(group)
        k = round(eval_frac * len(group))
        if k == 0 and len(group) >= 5 and eval_frac > 0:
            k = 1
        evals += group[:k]
        train += group[k:]
    return train, evals


def to_dataset(pairs):
    from datasets import Dataset
    return Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"],
                               "rejected": p["rejected"]} for p in pairs])


# ---------------------------------------------------------------------------
# Resource measurement
# ---------------------------------------------------------------------------

class NvmlEnergy:
    """Total GPU energy (J) over the visible GPUs, from the NVML counter."""

    def __init__(self):
        self.handles, self.start = [], []
        try:
            import os
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            ids = ([int(i) for i in visible.split(",") if i.strip()] if visible
                   else range(pynvml.nvmlDeviceGetCount()))
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in ids]
            self.names = [pynvml.nvmlDeviceGetName(h) for h in self.handles]
        except Exception as e:  # no GPU / no NVML: report, don't crash
            self.error = repr(e)

    def _read(self):
        return [self.nvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0 for h in self.handles]

    def begin(self):
        if self.handles:
            try:
                self.start = self._read()
            except Exception as e:
                self.error, self.handles = repr(e), []

    def end(self):
        if not self.handles or not self.start:
            return {"available": False, "error": getattr(self, "error", "no GPU")}
        joules = [b - a for a, b in zip(self.start, self._read())]
        return {"available": True, "gpus": [str(n) for n in self.names],
                "energy_j_per_gpu": joules, "energy_kwh": sum(joules) / 3.6e6}


def make_carbontracker_callback(log_dir, n_epochs):
    """Carbontracker epoch hooks as a Trainer callback; None if unavailable."""
    from transformers import TrainerCallback
    try:
        from carbontracker.tracker import CarbonTracker
    except ImportError:
        print("carbontracker not installed -- skipping (pip install carbontracker)")
        return None, None
    tracker = CarbonTracker(epochs=n_epochs, monitor_epochs=-1, components="gpu",
                            log_dir=str(log_dir), ignore_errors=True, verbose=1)

    class CTCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kw):
            tracker.epoch_start()

        def on_epoch_end(self, args, state, control, **kw):
            tracker.epoch_end()

        def on_train_end(self, args, state, control, **kw):
            tracker.stop()

    return tracker, CTCallback()


def parse_carbontracker(log_dir):
    try:
        from carbontracker import parser
        logs = parser.parse_all_logs(log_dir=str(log_dir))
        return [{k: v for k, v in log.items() if k in ("actual", "pred", "components")}
                for log in logs]
    except Exception as e:
        return {"error": repr(e)}


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs)
    train_pairs, eval_pairs = stratified_split(pairs, args.eval_frac, args.seed)
    print(f"{len(pairs)} pairs -> train {len(train_pairs)}, eval {len(eval_pairs)}")
    with open(out / "split.json", "w") as f:
        json.dump({"train_pair_ids": [p["pair_id"] for p in train_pairs],
                   "eval_pair_ids": [p["pair_id"] for p in eval_pairs]}, f)

    use_cuda = torch.cuda.is_available()
    if args.dtype == "auto":
        bf16 = use_cuda and torch.cuda.get_device_capability()[0] >= 8
    else:
        bf16 = args.dtype == "bfloat16"
    dtype = torch.bfloat16 if bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM")

    config = DPOConfig(
        output_dir=str(out),
        beta=args.beta,
        max_length=args.max_length,
        truncation_mode="keep_start",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=use_cuda,
        bf16=bf16,
        logging_steps=5,
        eval_strategy="steps" if eval_pairs else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,          # keep every checkpoint: evaluate.py picks one
        save_only_model=True,           # adapter only, no optimizer state (~3x smaller)
        report_to="none",
        seed=args.seed,
        use_cpu=not use_cuda,
    )

    callbacks, tracker = [], None
    if not args.no_carbontracker and use_cuda:
        tracker, cb = make_carbontracker_callback(out / "carbontracker", int(-(-args.epochs // 1)))
        if cb is not None:
            callbacks.append(cb)

    trainer = DPOTrainer(
        model=model, ref_model=None, args=config,
        train_dataset=to_dataset(train_pairs),
        eval_dataset=to_dataset(eval_pairs) if eval_pairs else None,
        processing_class=tokenizer, peft_config=peft_config, callbacks=callbacks,
    )
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({n_trainable / n_total:.2%})")

    energy = NvmlEnergy()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    energy.begin()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    gpu_energy = energy.end()

    trainer.save_model(str(out / "final"))
    report = {
        "args": vars(args),
        "n_pairs": {"train": len(train_pairs), "eval": len(eval_pairs)},
        "dtype": str(dtype),
        "trainable_params": n_trainable,
        "total_params": n_total,
        "wall_time_s": wall,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if use_cuda else None,
        "gpu_energy_nvml": gpu_energy,
        "carbontracker": parse_carbontracker(out / "carbontracker") if tracker else None,
        "global_steps": trainer.state.global_step,
        "log_history": trainer.state.log_history,
        "versions": {m: __import__(m).__version__ for m in ("torch", "transformers", "trl", "peft")},
        "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
    }
    with open(out / "resource_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    evals = [h for h in trainer.state.log_history if "eval_rewards/accuracies" in h]
    print(f"\nwall time {wall / 60:.1f} min | GPU energy "
          + (f"{gpu_energy['energy_kwh'] * 1000:.1f} Wh" if gpu_energy.get("available") else "n/a"))
    for h in evals:
        print(f"  step {h['step']:>4}  eval acc {h['eval_rewards/accuracies']:.2f}  "
              f"margin {h['eval_rewards/margins']:.3f}  loss {h['eval_loss']:.3f}")
    print(f"checkpoints in {out} -- pick one with evaluate.py (GAP, not DPO loss)")


if __name__ == "__main__":
    sys.exit(main())
