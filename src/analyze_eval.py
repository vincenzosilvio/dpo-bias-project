"""
Analysis of the evaluation run (evaluate.py output). CPU only, no model
needed: it works from each model's completions.jsonl (every text keeps its
text_utils.analyze() result) and metrics.json.

What it adds to evaluate.py's summary:
  * QUALITY diagnostics: share of texts with any gendered pronoun, first-
    person texts, non-Latin-script texts -- what the collapse looks like.
  * OCCUPATION-CLUSTER bootstrap CIs for GAP: texts cluster within 25
    occupations, so resampling texts as if independent (evaluate.py) gives
    CIs that are too narrow. Two-stage bootstrap: occupations with
    replacement within each stereotype group, then texts within each
    drawn occupation.
  * PAIRED base-vs-4-bit difference, same occupation draws for both models.
  * Exploratory PARITY distance: mean |p_female - 0.5| over occupations
    (the quantity DPO was actually trained to reduce, README).
  * Usable rate by stereotype group: does a model drop texts unevenly?
  * Figures in <out>/: fig1_collapse.png, fig2_gap.png, fig3_quantization.png

Usage (repo root):
    python src/analyze_eval.py --eval-dir results/eval --out results/analysis
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = ["base", "step10", "step20", "step40", "step60", "step100", "step153", "base_4bit"]
STEPS = {"base": 0, "step10": 10, "step20": 20, "step40": 40, "step60": 60,
         "step100": 100, "step153": 153}
CELLS = [("train_occ|train_tmpl", "train", "train"),
         ("train_occ|heldout_tmpl", "train", "heldout"),
         ("heldout_occ|train_tmpl", "heldout", "train"),
         ("heldout_occ|heldout_tmpl", "heldout", "heldout")]
N_BOOT = 2000
MIN_N_PER_SIDE = 30       # below this a GAP is reported as "not interpretable"

MALE = re.compile(r"\b(he|him|his|himself)\b", re.I)
FEMALE = re.compile(r"\b(she|her|hers|herself)\b", re.I)
FIRST_PERSON = re.compile(r"\bI\b|\b(?i:my|me|myself)\b")
NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿ぀-ヿ一-鿿가-힯]")

# Chart palette (validated: 3 slots, all pairs, light surface)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(REPO_ROOT / "results" / "eval"))
    ap.add_argument("--report", default=str(REPO_ROOT / "results" / "resource_report.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "analysis"))
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load(eval_dir):
    data = {}
    for tag in MODELS:
        d = Path(eval_dir) / tag
        if not (d / "metrics.json").exists():
            continue
        recs = [json.loads(line) for line in open(d / "completions.jsonl")]
        data[tag] = {"metrics": json.load(open(d / "metrics.json")), "records": recs,
                     "usable": [r for r in recs if r["_a"]["reason"] is None]}
    return data


# ---------------------------------------------------------------------------
# Quality diagnostics
# ---------------------------------------------------------------------------

def quality(recs):
    n = len(recs)
    txt = [r["completion"] for r in recs]
    return {
        "n": n,
        "usable_rate": sum(r["_a"]["reason"] is None for r in recs) / n,
        "gendered_pronoun_rate": sum(bool(MALE.search(t) or FEMALE.search(t)) for t in txt) / n,
        "first_person_rate": sum(bool(FIRST_PERSON.search(t)) for t in txt) / n,
        "non_latin_rate": sum(bool(NON_LATIN.search(t)) for t in txt) / n,
        "mean_new_tokens": sum(r["n_new_tokens"] for r in recs) / n,
    }


def usable_by_group(recs):
    tot, ok = Counter(), Counter()
    for r in recs:
        tot[r["stereotype"]] += 1
        ok[r["stereotype"]] += r["_a"]["reason"] is None
    return {g: ok[g] / tot[g] for g in sorted(tot)}


# ---------------------------------------------------------------------------
# Bias with occupation-cluster bootstrap
# ---------------------------------------------------------------------------

def by_occupation(usable, osplit, tsplit):
    """{stereotype: {occupation: np.array of is_female}} for one grid cell."""
    out = defaultdict(lambda: defaultdict(list))
    for r in usable:
        if r["split"] in osplit and r["template_split"] in tsplit:
            out[r["stereotype"]][r["occupation"]].append(r["_a"]["gender"] == "female")
    return {s: {o: np.array(v, dtype=float) for o, v in occ.items()} for s, occ in out.items()}


def pooled_pf(occ):
    allv = np.concatenate(list(occ.values())) if occ else np.array([])
    return (allv.mean() if allv.size else np.nan), int(allv.size)


def boot_group(occ, rng):
    """One two-stage bootstrap draw of pooled p_female for a group:
    occupations with replacement, then texts within each drawn occupation."""
    names = sorted(occ)
    tot, fem = 0, 0.0
    for i in rng.integers(0, len(names), len(names)):
        v = occ[names[i]]
        s = v[rng.integers(0, v.size, v.size)]
        tot += s.size
        fem += s.sum()
    return fem / tot if tot else np.nan


def gap_cell(usable, osplit, tsplit, rng):
    g = by_occupation(usable, osplit, tsplit)
    f, m = g.get("female", {}), g.get("male", {})
    pf, nf = pooled_pf(f)
    pm, nm = pooled_pf(m)
    res = {"gap": pf - pm if nf and nm else None, "n_female_coded": nf, "n_male_coded": nm,
           "occ_female_coded": len(f), "occ_male_coded": len(m), "ci95_cluster": None,
           "interpretable": bool(nf >= MIN_N_PER_SIDE and nm >= MIN_N_PER_SIDE)}
    if nf and nm and len(f) >= 2 and len(m) >= 2:
        vals = np.array([boot_group(f, rng) - boot_group(m, rng) for _ in range(N_BOOT)])
        vals = vals[~np.isnan(vals)]
        res["ci95_cluster"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return res


def paired_gap_diff(u_a, u_b, osplit, tsplit, rng):
    """GAP(b) - GAP(a) with the same occupation draws for both models."""
    ga, gb = by_occupation(u_a, osplit, tsplit), by_occupation(u_b, osplit, tsplit)
    occs = {s: sorted(set(ga.get(s, {})) | set(gb.get(s, {}))) for s in ("female", "male")}

    def point(g):
        return pooled_pf(g.get("female", {}))[0] - pooled_pf(g.get("male", {}))[0]

    def draw(g, picks):
        out = {}
        for s in ("female", "male"):
            occ = g.get(s, {})
            tot, fem = 0, 0.0
            for i in picks[s]:
                v = occ.get(occs[s][i])
                if v is None or v.size == 0:
                    continue
                smp = v[rng.integers(0, v.size, v.size)]
                tot += smp.size
                fem += smp.sum()
            out[s] = fem / tot if tot else np.nan
        return out["female"] - out["male"]

    vals = []
    for _ in range(N_BOOT):
        picks = {s: rng.integers(0, len(occs[s]), len(occs[s])) for s in occs}
        vals.append(draw(gb, picks) - draw(ga, picks))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return {"diff": float(point(gb) - point(ga)),
            "ci95_cluster": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def per_occupation(usable):
    d = defaultdict(list)
    meta = {}
    for r in usable:
        d[r["occupation"]].append(r["_a"]["gender"] == "female")
        meta[r["occupation"]] = (r["stereotype"], r["split"])
    return {o: {"stereotype": meta[o][0], "split": meta[o][1], "n": len(v),
                "p_female": sum(v) / len(v)} for o, v in sorted(d.items())}


def parity_distance(per_occ, min_n=5):
    v = [abs(x["p_female"] - 0.5) for x in per_occ.values()
         if x["n"] >= min_n and x["stereotype"] != "balanced"]
    return (float(np.mean(v)) if v else None), len(v)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_collapse(res, report, out):
    import matplotlib.pyplot as plt
    tags = [t for t in STEPS if t in res["models"]]
    x = [STEPS[t] for t in tags]
    q = [res["models"][t]["quality"] for t in tags]
    evals = [(h["step"], h["eval_rewards/accuracies"]) for h in report["log_history"]
             if "eval_rewards/accuracies" in h]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.35, 1]})
    style(a)
    series = [
        ("held-out pair reward accuracy (training log)", [s for s, _ in evals],
         [v for _, v in evals], C1, "s", "--"),
        ("texts with a gendered pronoun", x, [v["gendered_pronoun_rate"] for v in q], C2, "o", "-"),
        ("usable texts (single named character)", x, [v["usable_rate"] for v in q], C3, "D", "-"),
    ]
    for label, xs, ys, c, mk, ls in series:
        a.plot(xs, ys, color=c, lw=2, ls=ls, marker=mk, ms=6,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    a.set_ylim(-0.03, 1.05)
    a.set_xlabel("DPO training step (0 = base model)", color=INK2, fontsize=9)
    a.set_ylabel("share", color=INK2, fontsize=9)
    a.set_title("a) Offline DPO metrics looked fine; generations collapsed",
                loc="left", fontsize=10.5, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="center right", labelcolor=INK)
    style(b)
    b.plot(x, [res["models"][t]["ppl_reference"] for t in tags], color=C1, lw=2, marker="o",
           ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label="reference texts (base model's own)")
    b.plot(x, [res["models"][t]["ppl_wikitext"] for t in tags], color=C2, lw=2, marker="s",
           ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label="WikiText-2")
    b.set_ylim(0, None)
    b.set_xlabel("DPO training step (0 = base model)", color=INK2, fontsize=9)
    b.set_ylabel("perplexity", color=INK2, fontsize=9)
    b.set_title("b) Perplexity barely moved at step 10", loc="left", fontsize=10.5, color=INK)
    b.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_collapse.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_gap(res, out):
    import matplotlib.pyplot as plt
    tags = [t for t in ["base", "step10", "step20", "step40", "step60", "base_4bit"]
            if t in res["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=True)
    for ax, cell, title in [(axes[0], "train_occ|train_tmpl", "Train occupations, train templates (selection cell)"),
                            (axes[1], "train_occ|heldout_tmpl", "Train occupations, held-out templates")]:
        style(ax)
        ax.axhline(0, color=INK2, lw=1)
        for i, t in enumerate(tags):
            g = res["models"][t]["gap"][cell]
            if g["gap"] is None:
                continue
            ok = g["interpretable"]
            col = C1 if t != "base_4bit" else C2
            lo, hi = g["ci95_cluster"] or [g["gap"], g["gap"]]
            ax.plot([i, i], [lo, hi], color=col if ok else GRID, lw=2, solid_capstyle="round")
            ax.plot(i, g["gap"], marker="o", ms=8, color=col if ok else SURFACE,
                    markeredgecolor=col, markeredgewidth=2)
            n = min(g["n_female_coded"], g["n_male_coded"])
            ax.annotate(f"n={n}", (i, -0.95), ha="center", fontsize=8,
                        color=INK2 if ok else "#b3261e")
        ax.set_xticks(range(len(tags)), [t.replace("_", " ") for t in tags], fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].set_ylabel("GAP = p_female(female-coded) − p_female(male-coded)", color=INK2, fontsize=9)
    fig.text(0.01, 0.005, "Filled: ≥30 usable texts per side, 95% occupation-cluster bootstrap CI. "
             "Hollow: too few usable texts to interpret. n = usable texts on the smaller side.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig2_gap.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_quant(res, out):
    import matplotlib.pyplot as plt
    a, b = res["models"]["base"]["per_occupation"], res["models"]["base_4bit"]["per_occupation"]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    for grp, col, mk, lab in [("female", C1, "o", "female-coded"),
                              ("male", C2, "s", "male-coded"),
                              ("balanced", C3, "D", "balanced (control)")]:
        occ = [o for o in a if o in b and a[o]["stereotype"] == grp]
        ax.scatter([a[o]["p_female"] for o in occ], [b[o]["p_female"] for o in occ], s=46,
                   color=col, marker=mk, edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)
        for o in occ:
            if abs(b[o]["p_female"] - a[o]["p_female"]) >= 0.25:
                ax.annotate(o, (a[o]["p_female"], b[o]["p_female"]), xytext=(5, -3),
                            textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("p_female, base model (fp32)", color=INK2, fontsize=9)
    ax.set_ylabel("p_female, base model (NF4 4-bit)", color=INK2, fontsize=9)
    ax.set_title("Per-occupation gender share, fp32 vs 4-bit\n(labelled: moved ≥ 0.25)",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "fig3_quantization.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------

def fmt_ci(g):
    if g["gap"] is None:
        return "n/a"
    ci = g["ci95_cluster"]
    s = f"{g['gap']:+.2f}" + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")
    return s if g["interpretable"] else f"({s}, n<{MIN_N_PER_SIDE})"


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = load(args.eval_dir)
    report = json.load(open(args.report))

    res = {"n_boot": N_BOOT, "min_n_per_side": MIN_N_PER_SIDE, "models": {}}
    for tag, d in data.items():
        m = d["metrics"]
        po = per_occupation(d["usable"])
        pd, npd = parity_distance(po)
        res["models"][tag] = {
            "quality": quality(d["records"]),
            "usable_by_stereotype": usable_by_group(d["records"]),
            "reasons": m["reasons"],
            "ppl_reference": m["ppl_reference"], "ppl_wikitext": m["ppl_wikitext"],
            "inference": {k: m["inference"][k] for k in
                          ("tokens_per_s", "wh_per_1k_tokens", "model_size_mb", "peak_gpu_mem_gb")},
            "gap": {c: gap_cell(d["usable"], {o}, {t}, rng) for c, o, t in CELLS},
            "control_p_female": m["bias"]["control_p_female"],
            "parity_distance": {"value": pd, "n_occupations": npd},
            "per_occupation": po,
        }
    if "base" in data and "base_4bit" in data:
        res["base_4bit_minus_base"] = {
            c: paired_gap_diff(data["base"]["usable"], data["base_4bit"]["usable"], {o}, {t}, rng)
            for c, o, t in CELLS[:2]}
        res["base_4bit_minus_base"]["train_occ|all_tmpl"] = paired_gap_diff(
            data["base"]["usable"], data["base_4bit"]["usable"], {"train"}, {"train", "heldout"}, rng)

    json.dump(res, open(out / "analysis.json", "w"), indent=2)

    # ---- tables (markdown, pasted into the README) ----
    lines = ["| model | usable | any gendered pronoun | first person | non-Latin script "
             "| mean tokens | ppl ref | ppl wiki | GAP train/train | GAP train occ/held-out tmpl |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, r in res["models"].items():
        q = r["quality"]
        lines.append(
            f"| {tag} | {q['usable_rate']:.0%} | {q['gendered_pronoun_rate']:.0%} "
            f"| {q['first_person_rate']:.0%} | {q['non_latin_rate']:.0%} | {q['mean_new_tokens']:.0f} "
            f"| {r['ppl_reference']:.2f} | {r['ppl_wikitext']:.2f} "
            f"| {fmt_ci(r['gap']['train_occ|train_tmpl'])} | {fmt_ci(r['gap']['train_occ|heldout_tmpl'])} |")
    lines += ["", "| model | " + " | ".join(c for c, _, _ in CELLS) + " | parity dist. | control p_f |",
              "|---|" + "---|" * (len(CELLS) + 2)]
    for tag in ("base", "base_4bit"):
        if tag in res["models"]:
            r = res["models"][tag]
            lines.append(f"| {tag} | " + " | ".join(fmt_ci(r["gap"][c]) for c, _, _ in CELLS)
                         + f" | {r['parity_distance']['value']:.2f} | "
                         f"{r['control_p_female']['p_female']:.2f} |")
    if "base_4bit_minus_base" in res:
        lines += ["", "GAP(4-bit) - GAP(fp32), paired occupation-cluster bootstrap:"]
        for c, v in res["base_4bit_minus_base"].items():
            lines.append(f"- {c}: {v['diff']:+.2f} [{v['ci95_cluster'][0]:+.2f}, {v['ci95_cluster'][1]:+.2f}]")
        lines += ["", "Usable rate by stereotype group:"]
        for tag in ("base", "base_4bit"):
            lines.append(f"- {tag}: " + ", ".join(f"{g} {v:.0%}" for g, v in
                                                  res["models"][tag]["usable_by_stereotype"].items()))
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig_collapse(res, report, out)
    fig_gap(res, out)
    if "base_4bit" in res["models"]:
        fig_quant(res, out)
    print(f"\nwrote {out}/analysis.json, tables.md, fig*.png")


if __name__ == "__main__":
    main()
