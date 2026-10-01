"""
Analysis of a fine-tuning run that PASSED the quality guard (run 3:
supervised fine-tuning on the counterfactual chosen texts). CPU only; reads
evaluate.py's output. Reuses the statistics of analyze_eval.py.

Adds, compared with analyze_eval.py:
  * the trajectory of GAP over training steps on all four grid cells,
    with occupation-cluster bootstrap CIs;
  * PAIRED differences (same occupation draws for both models):
      selected - base            (did fine-tuning reduce the bias?)
      base_4bit - base           (does quantization move the bias?)
      selected_4bit - selected   (does quantization undo the fine-tuning?)
      interaction = (selected_4bit - selected) - (base_4bit - base)
  * per-occupation p_female for base, selected and selected_4bit
    (overshoot check: did any occupation flip past parity?);
  * side-by-side example texts for the same prompt and sample id.

Usage (repo root):
    python src/analyze_finetune.py --eval-dir results/eval3 --out results/analysis3
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from analyze_eval import (C1, C2, C3, GRID, INK, INK2, MIN_N_PER_SIDE, SURFACE, CELLS,
                          fmt_ci, gap_cell, paired_gap_diff, parity_distance,
                          per_occupation, quality, style, usable_by_group)

REPO_ROOT = Path(__file__).resolve().parent.parent
PAIRED_CELLS = [("train_occ|train_tmpl", {"train"}, {"train"}),
                ("train_occ|heldout_tmpl", {"train"}, {"heldout"}),
                ("heldout_occ|all_tmpl", {"heldout"}, {"train", "heldout"}),
                ("heldout_occ|heldout_tmpl", {"heldout"}, {"heldout"})]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(REPO_ROOT / "results" / "eval3"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "analysis3"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selected", default=None,
                    help="checkpoint tag; default: the one summary.json records (run 4 has none)")
    ap.add_argument("--exclude-non-latin", action="store_true",
                    help="SENSITIVITY analysis: also drop usable texts containing CJK/Cyrillic/"
                         "Arabic characters (the fine-tuned model's language drift). Not the "
                         "pre-registered metric.")
    ap.add_argument("--quant-suffix", default="_4bit",
                    help="tag suffix of the quantized fine-tuned model (run 4: _4bit_merged)")
    return ap.parse_args()


def tag_order(tags):
    def key(t):
        m = re.match(r"step(\d+)", t)
        return (t.endswith("_4bit"), -1 if t.startswith("base") else int(m.group(1)) if m else 10**9)
    return sorted(tags, key=key)


def load(eval_dir):
    d = Path(eval_dir)
    data = {}
    for sub in d.iterdir():
        if (sub / "metrics.json").exists():
            recs = [json.loads(line) for line in open(sub / "completions.jsonl")]
            data[sub.name] = {"metrics": json.load(open(sub / "metrics.json")), "records": recs,
                              "usable": [r for r in recs if r["_a"]["reason"] is None]}
    summary = json.load(open(d / "summary.json"))
    return data, summary["selected"]


def paired(data, a, b, rng):
    return {c: paired_gap_diff(data[a]["usable"], data[b]["usable"], o, t, rng)
            for c, o, t in PAIRED_CELLS}


def interaction(data, sel, rng, n_boot=2000):
    """(sel4 - sel) - (base4 - base) on train occupations, all templates, with
    the same occupation draw for all four models in each replicate."""
    from analyze_eval import by_occupation, pooled_pf
    tags = ["base", "base_4bit", sel, f"{sel}_4bit"]
    groups = {t: by_occupation(data[t]["usable"], {"train"}, {"train", "heldout"}) for t in tags}
    occs = {s: sorted(set().union(*(groups[t].get(s, {}).keys() for t in tags)))
            for s in ("female", "male")}

    def gap(g, picks=None):
        out = {}
        for s in ("female", "male"):
            occ = g.get(s, {})
            names = occs[s] if picks is None else [occs[s][i] for i in picks[s]]
            vals = [occ[n] if picks is None else occ[n][rng.integers(0, occ[n].size, occ[n].size)]
                    for n in names if n in occ and occ[n].size]
            v = np.concatenate(vals)
            out[s] = v.mean()
        return out["female"] - out["male"]

    point = (gap(groups[tags[3]]) - gap(groups[tags[2]])) - (gap(groups[tags[1]]) - gap(groups[tags[0]]))
    vals = []
    for _ in range(n_boot):
        picks = {s: rng.integers(0, len(occs[s]), len(occs[s])) for s in occs}
        g = [gap(groups[t], picks) for t in tags]
        vals.append((g[3] - g[2]) - (g[1] - g[0]))
    return {"diff": float(point),
            "ci95_cluster": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def examples(data, sel, n=4):
    """Same prompt, same sample id: base vs selected, held-out occupations first."""
    base = {(r["occupation"], r["template_id"], r["sample_id"]): r for r in data["base"]["records"]}
    out = []
    for r in data[sel]["records"]:
        if r["split"] != "heldout" or r["stereotype"] == "balanced":
            continue
        b = base.get((r["occupation"], r["template_id"], r["sample_id"]))
        if b and b["_a"]["reason"] is None and r["_a"]["reason"] is None \
                and b["_a"]["gender"] != r["_a"]["gender"]:
            out.append((r["occupation"], r["prompt"].split(". Begin")[0], b["completion"], r["completion"],
                        b["_a"]["gender"], r["_a"]["gender"]))
    seen, picked = set(), []
    for e in out:          # one per occupation
        if e[0] not in seen:
            seen.add(e[0])
            picked.append(e)
    return picked[:n]


# ---------------------------------------------------------------------------

def fig_trajectory(res, steps, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=SURFACE, sharey=True)
    for ax, osp, title in [(axes[0], "train", "Trained occupations"),
                           (axes[1], "heldout", "Held-out occupations (never trained on)")]:
        style(ax)
        ax.axhline(0, color=INK2, lw=1)
        for tsp, col, mk, lab, dx in [("train", C1, "o", "trained templates", -1.2),
                                      ("heldout", C2, "s", "held-out templates", 1.2)]:
            cell = f"{osp}_occ|{tsp}_tmpl"
            xs, ys, lo, hi = [], [], [], []
            for t, s in steps:
                g = res["models"][t]["gap"][cell]
                if g["gap"] is None:
                    continue
                xs.append(s + dx)
                ys.append(g["gap"])
                ci = g["ci95_cluster"] or [g["gap"], g["gap"]]
                lo.append(ci[0])
                hi.append(ci[1])
            ax.vlines(xs, lo, hi, color=col, lw=1.5, alpha=0.55)
            ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=lab)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("fine-tuning step (0 = base model)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    axes[0].set_ylabel("GAP = p_female(female-coded) − p_female(male-coded)", color=INK2, fontsize=9)
    axes[0].set_ylim(-0.25, 1.05)
    fig.text(0.01, 0.005, "Bars: 95% occupation-cluster bootstrap CI. Held-out occupations: 2 per "
             "stereotype side, so their CIs rest on few clusters.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig4_finetune_trajectory.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_per_occupation(res, sel, out):
    import matplotlib.pyplot as plt
    a, b = res["models"]["base"]["per_occupation"], res["models"][sel]["per_occupation"]
    fig, ax = plt.subplots(figsize=(6.4, 5.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.axhline(0.5, color=GRID, lw=1)
    for grp, col, mk, lab in [("female", C1, "o", "female-coded"), ("male", C2, "s", "male-coded"),
                              ("balanced", C3, "D", "balanced (control)")]:
        occ = [o for o in a if o in b and a[o]["stereotype"] == grp]
        ax.scatter([a[o]["p_female"] for o in occ], [b[o]["p_female"] for o in occ], s=50,
                   color=col, marker=mk, edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)
        for o in occ:
            if a[o]["split"] == "heldout" and grp != "balanced":
                ax.annotate(o + " (held out)", (a[o]["p_female"], b[o]["p_female"]), xytext=(5, -3),
                            textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("p_female, base model", color=INK2, fontsize=9)
    ax.set_ylabel(f"p_female, fine-tuned ({sel})", color=INK2, fontsize=9)
    ax.set_title("Per occupation: points move toward 0.5\n(dashed = no change)", loc="left",
                 fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "fig5_per_occupation.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_quant_interaction(res, sel, out):
    import matplotlib.pyplot as plt
    tags = [("base", "base\nfp32"), ("base_4bit", "base\n4-bit"), (sel, f"fine-tuned\nfp32"),
            (f"{sel}_4bit", "fine-tuned\n4-bit")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, sharey=True)
    for ax, cell, title in [(axes[0], "train_occ|heldout_tmpl", "Trained occupations, held-out templates"),
                            (axes[1], "heldout_occ|heldout_tmpl", "Held-out occupations and templates")]:
        style(ax)
        for i, (t, lab) in enumerate(tags):
            g = res["models"][t]["gap"][cell]
            col = C1 if "4bit" not in t else C2
            ci = g["ci95_cluster"] or [g["gap"], g["gap"]]
            ax.bar(i, g["gap"], width=0.6, color=col, edgecolor=SURFACE, linewidth=2)
            ax.vlines(i, ci[0], ci[1], color=INK2, lw=1.5)
            ax.annotate(f"{g['gap']:+.2f}", (i, max(g["gap"], 0) + 0.02), ha="center",
                        fontsize=8.5, color=INK)
        ax.set_xticks(range(4), [l for _, l in tags], fontsize=8.5)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].set_ylabel("GAP (95% occupation-cluster CI)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig6_quantization_after_finetune.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------

def fmt_d(v):
    return f"{v['diff']:+.2f} [{v['ci95_cluster'][0]:+.2f}, {v['ci95_cluster'][1]:+.2f}]"


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data, sel = load(args.eval_dir)
    sel = args.selected or sel
    if args.exclude_non_latin:
        from analyze_eval import NON_LATIN
        for d in data.values():
            for r in d["records"]:
                if r["_a"]["reason"] is None and NON_LATIN.search(r["completion"]):
                    r["_a"] = {**r["_a"], "reason": "non_latin"}
            d["usable"] = [r for r in d["records"] if r["_a"]["reason"] is None]
    q = args.quant_suffix
    if q != "_4bit" and f"{sel}{q}" in data:      # analyse the merged model under the usual name
        data[f"{sel}_4bit"] = data.pop(f"{sel}{q}")
    tags = tag_order(data)
    res = {"selected": sel, "models": {}}
    for t in tags:
        d, m = data[t], data[t]["metrics"]
        po = per_occupation(d["usable"])
        res["models"][t] = {
            "quality": quality(d["records"]), "usable_by_stereotype": usable_by_group(d["records"]),
            "ppl_reference": m["ppl_reference"], "ppl_wikitext": m["ppl_wikitext"],
            "inference": {k: m["inference"][k] for k in
                          ("tokens_per_s", "wh_per_1k_tokens", "model_size_mb", "peak_gpu_mem_gb")},
            "gap": {c: gap_cell(d["usable"], {o}, {tt}, rng) for c, o, tt in CELLS},
            "control_p_female": m["bias"]["control_p_female"],
            "parity_distance": dict(zip(("value", "n_occupations"), parity_distance(po))),
            "per_occupation": po}
    res["paired"] = {f"{sel} - base": paired(data, "base", sel, rng)}
    if "base_4bit" in data:
        res["paired"]["base_4bit - base"] = paired(data, "base", "base_4bit", rng)
    if f"{sel}_4bit" in data:
        res["paired"][f"{sel}_4bit - {sel}"] = paired(data, sel, f"{sel}_4bit", rng)
        res["interaction_train_occ_all_tmpl"] = interaction(data, sel, rng)
    json.dump(res, open(out / "analysis.json", "w"), indent=2)

    L = [f"Selected checkpoint (pre-registered rule): **{sel}**", "",
         "| model | usable | " + " | ".join(c for c, _, _ in CELLS) + " | parity dist. | control p_f | ppl ref | ppl wiki |",
         "|---|---|" + "---|" * (len(CELLS) + 4)]
    for t in tags:
        r = res["models"][t]
        L.append(f"| {t} | {r['quality']['usable_rate']:.0%} | "
                 + " | ".join(fmt_ci(r["gap"][c]) for c, _, _ in CELLS)
                 + f" | {r['parity_distance']['value']:.2f} | {r['control_p_female']['p_female']:.2f}"
                 f" | {r['ppl_reference']:.2f} | {r['ppl_wikitext']:.2f} |")
    L += ["", "Paired GAP differences (occupation-cluster bootstrap, 95% CI):", ""]
    for name, cells in res["paired"].items():
        L.append(f"- **{name}**: " + "; ".join(f"{c} {fmt_d(v)}" for c, v in cells.items()))
    if "interaction_train_occ_all_tmpl" in res:
        L.append(f"- **interaction** (4-bit effect after fine-tuning minus 4-bit effect on base), "
                 f"train occ, all templates: {fmt_d(res['interaction_train_occ_all_tmpl'])}")
    L += ["", "Per occupation p_female (n usable):", "",
          f"| occupation | stereotype | split | base | {sel} | {sel}_4bit |", "|---|---|---|---|---|---|"]
    po_b = res["models"]["base"]["per_occupation"]
    for o in sorted(po_b, key=lambda o: (po_b[o]["stereotype"], po_b[o]["split"], o)):
        row = [f"{res['models'][t]['per_occupation'][o]['p_female']:.2f} "
               f"({res['models'][t]['per_occupation'][o]['n']})"
               if o in res["models"].get(t, {}).get("per_occupation", {}) else "–"
               for t in ("base", sel, f"{sel}_4bit")]
        L.append(f"| {o} | {po_b[o]['stereotype']} | {po_b[o]['split']} | " + " | ".join(row) + " |")
    (out / "tables.md").write_text("\n".join(L) + "\n")

    ex = examples(data, sel)
    E = [f"# Same prompt, same sample: base vs fine-tuned ({sel}), held-out occupations", ""]
    for occ, prompt, b, f, gb, gf in ex:
        E += [f"**{occ}**: *{prompt}.*", "", f"- base ({gb}): {b}", f"- fine-tuned ({gf}): {f}", ""]
    (out / "examples.md").write_text("\n".join(E) + "\n")
    print("\n".join(L))

    steps = [(t, 0 if t == "base" else int(t[4:])) for t in tags if t == "base" or
             (t.startswith("step") and not t.endswith("_4bit"))]
    fig_trajectory(res, steps, out)
    fig_per_occupation(res, sel, out)
    if f"{sel}_4bit" in data and "base_4bit" in data:
        fig_quant_interaction(res, sel, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
