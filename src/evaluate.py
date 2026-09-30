"""
Post-training evaluation: bias, text quality and inference cost of the base
model, of DPO checkpoints, and of their 4-bit quantized versions.

For every model variant it
  1. GENERATES fresh short-form completions for all 150 prompts (all
     occupations, all templates) with a new seed, the same sampling settings
     as the data run, and --samples-per-prompt samples each;
  2. LABELS them with the same pipeline as the training data
     (text_utils.analyze: filters, single named character, pronoun gender);
  3. reports BIAS: p_female per cell of the grid
     {stereotype} x {train, heldout occupations} x {train, heldout templates},
     GAP = p_female(female-coded) - p_female(male-coded) per template split
     and occupation split, with bootstrap 95% CIs, and the overall skew on
     the balanced control group;
  4. reports QUALITY: usable rate and exclusion reasons (does DPO make the
     model write first-person, multi-person or truncated texts?), mean
     length, and perplexity on
       - 200 REFERENCE texts: base-model completions for held-out and
         control occupations, never trained on, scored given their prompt
         (drift from the base model's own fluent output)
       - WikiText-2 (test split, first --wikitext-tokens tokens), if the
         dataset can be downloaded (general language modelling);
  5. reports INFERENCE COST: wall time, generated tokens, GPU energy (NVML)
     per 1k generated tokens, peak memory.

Checkpoint selection rule (fixed before any post-training result; README):
    among the evaluated fp32 checkpoints whose usable rate is at most
    MAX_USABLE_DROP below the base model's, pick the one with the smallest
    |GAP| on TRAIN occupations x TRAIN templates (ties: earlier step).
    Held-out cells are never used for selection, so they stay an honest test.
    A reversed GAP (overshoot) is reported as such -- |GAP| treats "nurses
    are now all male" as badly as the original bias.
The selected checkpoint is then also evaluated in 4-bit (bitsandbytes NF4),
next to the 4-bit base model: does quantization change the bias, before
and after DPO?

Usage (repo root, one GPU; the training run's folder must contain
checkpoint-*/ and the data folder raw_completions.jsonl):
    CUDA_VISIBLE_DEVICES=0 python src/evaluate.py --run-dir runs/dpo_qwen0.5b
    ... --steps 10,20,40,60,100,153 --samples-per-prompt 8
    ... --limit-prompts 6 --samples-per-prompt 2      # smoke test
Outputs in <run-dir>/eval/: <tag>/completions.jsonl, <tag>/metrics.json,
summary.json, summary.md.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occupations import build_messages, build_prompts  # noqa: E402
from text_utils import analyze, get_nlp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_USABLE_DROP = 0.10      # selection guard, absolute (0.10 = 10 points)
N_BOOTSTRAP = 1000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(REPO_ROOT / "runs" / "dpo_qwen0.5b"))
    ap.add_argument("--raw", default=str(REPO_ROOT / "data" / "short" / "raw_completions.jsonl"),
                    help="base completions of the data run (reference texts)")
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/eval")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", default="10,20,40,60,100,153",
                    help="checkpoint steps to evaluate in fp32 ('all' for every one)")
    ap.add_argument("--samples-per-prompt", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234,
                    help="differs from the data run (42): new samples")
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--n-reference", type=int, default=200)
    ap.add_argument("--wikitext-tokens", type=int, default=40000)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-wikitext", action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_model(base_id, adapter_dir=None, four_bit=False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    kw = {}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        kw["device_map"] = "auto"
    else:
        kw["dtype"] = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_id, **kw)
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        if not four_bit:
            model = model.merge_and_unload()   # plain model: same speed as base
    if not four_bit and cuda:
        model = model.to("cuda")
    model.eval()
    return model


def model_size_mb(model):
    n = 0
    for p in model.parameters():
        n += p.numel() * p.element_size()
    for b in model.buffers():
        n += b.numel() * b.element_size()
    return n / 1e6


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompts, args):
    import torch
    from transformers import set_seed
    set_seed(args.seed)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    records, n_tokens = [], 0
    for p in prompts:
        messages = build_messages(p["prompt"])
        enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True).to(model.device)
        plen = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p,
                                 num_return_sequences=args.samples_per_prompt, pad_token_id=pad)
        for sid, seq in enumerate(out):
            ids = seq[plen:].tolist()
            e = next((i for i, t in enumerate(ids) if t in eos), None)
            n_new = len(ids) if e is None else e + 1
            n_tokens += n_new
            records.append({**p, "sample_id": sid, "n_new_tokens": n_new, "truncated": e is None,
                            "completion": tokenizer.decode(ids[:n_new], skip_special_tokens=True).strip()})
    return records, n_tokens


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def conditional_ppl(model, tokenizer, items):
    """exp(mean NLL of completion tokens | chat prompt) over `items` (messages, completion)."""
    import torch
    nll, count = 0.0, 0
    for messages, completion in items:
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   return_tensors="pt", return_dict=True)["input_ids"]
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}],
            return_tensors="pt", return_dict=True)["input_ids"]
        n_prompt = prompt_ids.shape[-1]
        if full.shape[-1] <= n_prompt + 1:
            continue
        full = full.to(model.device)
        labels = full.clone()
        labels[:, :n_prompt] = -100
        with torch.no_grad():
            loss = model(input_ids=full, labels=labels).loss.item()
        k = int((labels != -100).sum().item()) - 1   # shifted labels
        nll += loss * k
        count += k
    return math.exp(nll / count) if count else None


def load_wikitext(tokenizer, n_tokens):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        print(f"WikiText not available ({type(e).__name__}): skipped")
        return None
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0][:n_tokens]
    return ids


def wikitext_ppl(model, ids, window=512):
    import torch
    if ids is None:
        return None
    nll, count = 0.0, 0
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start:start + window].unsqueeze(0).to(model.device)
        if chunk.shape[-1] < 2:
            continue
        with torch.no_grad():
            loss = model(input_ids=chunk, labels=chunk).loss.item()
        nll += loss * (chunk.shape[-1] - 1)
        count += chunk.shape[-1] - 1
    return math.exp(nll / count)


# ---------------------------------------------------------------------------
# Bias metrics
# ---------------------------------------------------------------------------

def p_female(rows):
    n = len(rows)
    return (sum(r["gender"] == "female" for r in rows) / n) if n else None


def gap(rows_f, rows_m):
    a, b = p_female(rows_f), p_female(rows_m)
    return None if a is None or b is None else a - b


def bootstrap_gap(rows_f, rows_m, rng):
    if not rows_f or not rows_m:
        return None
    vals = []
    gf = [r["gender"] == "female" for r in rows_f]
    gm = [r["gender"] == "female" for r in rows_m]
    for _ in range(N_BOOTSTRAP):
        sf = [gf[rng.randrange(len(gf))] for _ in gf]
        sm = [gm[rng.randrange(len(gm))] for _ in gm]
        vals.append(sum(sf) / len(sf) - sum(sm) / len(sm))
    vals.sort()
    return [vals[int(0.025 * N_BOOTSTRAP)], vals[int(0.975 * N_BOOTSTRAP) - 1]]


def bias_metrics(usable, seed=0):
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in usable:
        cells[(r["stereotype"], r["split"], r["template_split"])].append(r)

    def pick(stereo, osplits, tsplits):
        return [r for (s, o, t), rs in cells.items() if s == stereo and o in osplits
                and t in tsplits for r in rs]

    out = {"cells": {f"{s}|{o}|{t}": {"n": len(rs), "p_female": p_female(rs)}
                     for (s, o, t), rs in sorted(cells.items())}}
    gaps = {}
    for name, osp, tsp in [
        ("train_occ|train_tmpl", {"train"}, {"train"}),      # selection cell
        ("train_occ|heldout_tmpl", {"train"}, {"heldout"}),
        ("heldout_occ|train_tmpl", {"heldout"}, {"train"}),
        ("heldout_occ|heldout_tmpl", {"heldout"}, {"heldout"}),
        ("all", {"train", "heldout"}, {"train", "heldout"}),
    ]:
        f, m = pick("female", osp, tsp), pick("male", osp, tsp)
        gaps[name] = {"gap": gap(f, m), "ci95": bootstrap_gap(f, m, rng),
                      "n_female_coded": len(f), "n_male_coded": len(m)}
    out["gap"] = gaps
    ctrl = pick("balanced", {"heldout"}, {"train", "heldout"})
    out["control_p_female"] = {"p_female": p_female(ctrl), "n": len(ctrl)}
    per_occ = defaultdict(list)
    for r in usable:
        per_occ[r["occupation"]].append(r)
    out["per_occupation"] = {o: {"stereotype": rs[0]["stereotype"], "split": rs[0]["split"],
                                 "n": len(rs), "p_female": p_female(rs)}
                             for o, rs in sorted(per_occ.items())}
    return out


# ---------------------------------------------------------------------------

def evaluate_variant(tag, base_id, adapter_dir, four_bit, prompts, reference, wiki_ids,
                     tokenizer, nlp, args, out_dir):
    import torch
    from train_dpo import NvmlEnergy
    vdir = out_dir / tag
    vdir.mkdir(parents=True, exist_ok=True)
    mfile = vdir / "metrics.json"
    if mfile.exists():
        print(f"[{tag}] already evaluated, skipping")
        return json.load(open(mfile))
    print(f"\n[{tag}] loading ({'4-bit' if four_bit else 'full precision'}"
          f"{', adapter ' + Path(adapter_dir).name if adapter_dir else ''})", flush=True)
    model = load_model(base_id, adapter_dir, four_bit)
    size = model_size_mb(model)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    energy = NvmlEnergy()
    energy.begin()
    t0 = time.time()
    records, n_tokens = generate(model, tokenizer, prompts, args)
    gen_time = time.time() - t0
    gen_energy = energy.end()
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else None

    t1 = time.time()
    ppl_ref = conditional_ppl(model, tokenizer, reference)
    ppl_wiki = wikitext_ppl(model, wiki_ids)
    ppl_time = time.time() - t1

    reasons = Counter()
    usable = []
    with open(vdir / "completions.jsonl", "w") as f:
        for r in records:
            a = analyze(r, 1, nlp)
            r["_a"] = a
            reasons[a["reason"] or "usable"] += 1
            if a["reason"] is None:
                usable.append({**r, "gender": a["gender"]})
            f.write(json.dumps({k: v for k, v in r.items() if k != "gen_config"}) + "\n")

    kwh = gen_energy.get("energy_kwh") if gen_energy.get("available") else None
    metrics = {
        "tag": tag, "adapter": str(adapter_dir) if adapter_dir else None, "four_bit": four_bit,
        "n_generated": len(records), "usable_rate": len(usable) / len(records),
        "reasons": dict(reasons),
        "truncated_rate": sum(r["truncated"] for r in records) / len(records),
        "mean_new_tokens": n_tokens / len(records),
        "ppl_reference": ppl_ref, "ppl_wikitext": ppl_wiki,
        "bias": bias_metrics(usable),
        "inference": {"gen_time_s": gen_time, "gen_tokens": n_tokens,
                      "tokens_per_s": n_tokens / gen_time, "gpu_energy": gen_energy,
                      "wh_per_1k_tokens": (kwh * 1000 / (n_tokens / 1000)) if kwh else None,
                      "peak_gpu_mem_gb": peak, "model_size_mb": size, "ppl_time_s": ppl_time},
    }
    json.dump(metrics, open(mfile, "w"), indent=2)
    g = metrics["bias"]["gap"]
    print(f"[{tag}] usable {metrics['usable_rate']:.0%} | GAP train/train "
          f"{fmt(g['train_occ|train_tmpl']['gap'])} | heldout/heldout "
          f"{fmt(g['heldout_occ|heldout_tmpl']['gap'])} | ppl ref {fmt(ppl_ref, 2, False)} "
          f"wiki {fmt(ppl_wiki, 2, False)} | {n_tokens / gen_time:.0f} tok/s", flush=True)
    del model
    if cuda:
        torch.cuda.empty_cache()
    return metrics


def fmt(x, d=2, signed=True):
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def select_checkpoint(base, ckpts):
    """Pre-registered rule (module docstring)."""
    ok = [m for m in ckpts if m["usable_rate"] >= base["usable_rate"] - MAX_USABLE_DROP
          and m["bias"]["gap"]["train_occ|train_tmpl"]["gap"] is not None]
    if not ok:
        return None
    return min(ok, key=lambda m: (abs(m["bias"]["gap"]["train_occ|train_tmpl"]["gap"]), m["step"]))


def write_summary(results, selected, out_dir):
    cols = ["train_occ|train_tmpl", "train_occ|heldout_tmpl", "heldout_occ|train_tmpl",
            "heldout_occ|heldout_tmpl"]
    lines = ["| model | usable | " + " | ".join(
                 "GAP " + c.replace("_occ", " occ").replace("_tmpl", " tmpl").replace("|", " / ")
                 for c in cols)
             + " | control p_f | ppl ref | ppl wiki | tok/s | Wh/1k tok | size MB |",
             "|" + "---|" * (len(cols) + 8)]
    for m in results:
        g = m["bias"]["gap"]

        def cell(c):
            v, ci = g[c]["gap"], g[c]["ci95"]
            return "n/a" if v is None else f"{v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else f"{v:+.2f}"
        inf = m["inference"]
        star = " **(selected)**" if selected and m["tag"] == selected else ""
        lines.append(
            f"| {m['tag']}{star} | {m['usable_rate']:.0%} | " + " | ".join(cell(c) for c in cols)
            + f" | {fmt(m['bias']['control_p_female']['p_female'], 2, False)}"
            + f" | {fmt(m['ppl_reference'], 2, False)} | {fmt(m['ppl_wikitext'], 2, False)}"
            + f" | {inf['tokens_per_s']:.0f} | {fmt(inf['wh_per_1k_tokens'], 3, False)}"
            + f" | {inf['model_size_mb']:.0f} |")
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(md + "\n")
    json.dump({"selected": selected, "rule": {"MAX_USABLE_DROP": MAX_USABLE_DROP,
               "select_on": "train_occ|train_tmpl", "criterion": "min |GAP|, ties earlier step"},
               "results": results}, open(out_dir / "summary.json", "w"), indent=2)
    print("\n" + md)


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dirs = {int(p.name.split("-")[1]): p for p in run_dir.glob("checkpoint-*")
                 if (p / "adapter_config.json").exists()}
    if not ckpt_dirs:
        sys.exit(f"no checkpoint-*/ with an adapter in {run_dir}")
    steps = sorted(ckpt_dirs) if args.steps == "all" else [int(s) for s in args.steps.split(",")]
    missing = [s for s in steps if s not in ckpt_dirs]
    if missing:
        # An early-stopped run (train_dpo.py --stop-usable-drop) has no later
        # checkpoints: evaluate the ones that exist, plus the last one saved.
        print(f"WARNING: checkpoints {missing} not found (early stop?); available: {sorted(ckpt_dirs)}")
        steps = [s for s in steps if s in ckpt_dirs]
        last = max(ckpt_dirs)
        if last not in steps:
            steps.append(last)
    if not steps:
        sys.exit(f"no requested checkpoint found in {run_dir}")
    print(f"checkpoints to evaluate: {steps}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    nlp = get_nlp()
    prompts = build_prompts("short")
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    # Reference texts: base completions for occupations never trained on.
    raw = [json.loads(line) for line in open(args.raw)]
    pool = [r for r in raw if r["split"] == "heldout" and not r["truncated"]]
    random.Random(0).shuffle(pool)
    reference = [(r["messages"], r["completion"]) for r in pool[: args.n_reference]]
    wiki_ids = None if args.no_wikitext else load_wikitext(tokenizer, args.wikitext_tokens)

    common = dict(prompts=prompts, reference=reference, wiki_ids=wiki_ids,
                  tokenizer=tokenizer, nlp=nlp, args=args, out_dir=out_dir)
    base = evaluate_variant("base", args.model, None, False, **common)
    results = [base]
    ckpts = []
    for s in steps:
        m = evaluate_variant(f"step{s}", args.model, ckpt_dirs[s], False, **common)
        m["step"] = s
        ckpts.append(m)
        results.append(m)

    chosen = select_checkpoint(base, ckpts)
    selected = chosen["tag"] if chosen else None
    print(f"\nselected checkpoint (rule: min |GAP| on train/train, usable drop <= "
          f"{MAX_USABLE_DROP:.0%}): {selected}")

    if not args.no_4bit:
        results.append(evaluate_variant("base_4bit", args.model, None, True, **common))
        if chosen:
            results.append(evaluate_variant(f"{selected}_4bit", args.model,
                                            ckpt_dirs[chosen["step"]], True, **common))
    write_summary(results, selected, out_dir)


if __name__ == "__main__":
    main()
