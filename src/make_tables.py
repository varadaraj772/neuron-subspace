"""Generate LaTeX tables from results/*/summary.json and results_hybrid/*."""
import glob
import json
import os

SHORT = {
    "Qwen3.5-0.8B": "Qwen3.5-0.8B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "R1-Distill-1.5B",
    "LFM3-1.7B": "LFM3-1.7B",
    "Qwen3.5-4B": "Qwen3.5-4B",
}
ORDER = ["Qwen3.5-0.8B", "DeepSeek-R1-Distill-Qwen-1.5B", "LFM3-1.7B", "Qwen3.5-4B"]


def load(pattern):
    out = {}
    for p in glob.glob(pattern):
        d = json.load(open(p))
        out[d["tag"]] = d
    return out


def pct(d):
    lo, hi = d["refusal_rate_ci95"]
    return f"{100*d['refusal_rate']:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


def pt(d):
    return f"{100*d['refusal_rate']:.1f}"


def write(name, body):
    os.makedirs("paper/tables", exist_ok=True)
    with open(f"paper/tables/{name}.tex", "w") as f:
        f.write(body)
    print("wrote", name)


def main():
    R = load("results/*/summary.json")
    H = load("results_hybrid/*/summary.json")
    tags = [t for t in ORDER if t in R]

    rows = []
    for t in tags:
        d = R[t]
        rows.append(f"{SHORT[t]} & {d['family'].replace('_', r'\_')} & {d['n_layers']} "
                    f"& {d['d_model']} & {d['d_mlp']} & complete \\\\")
    if "Qwen3.5-4B" not in R:
        rows.append(r"Qwen3.5-4B & qwen3\_5 & 32 & 2560 & 9216 & pending \\")
    write("setup", "\n".join([
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Family & Layers & $d$ & $d_{\mathrm{MLP}}$ & Status \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]))

    rows = []
    for t in tags:
        d = R[t]
        q = d["test"]
        rd = q["random_direction"]["harmful_score_mean"] - q["baseline_harmful"]["refusal_score_mean"]
        rows.append(
            f"{SHORT[t]} & {d['selection']['layer']} & {pct(q['baseline_harmful'])} "
            f"& {pct(q['ablated_harmful'])} & {pct(q['single_layer_ablation_harmful'])} "
            f"& ${rd:+.2f}$ & {q['ablated_kl_harmless']:.3f} \\\\")
    write("main", "\n".join([
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Layer & Baseline (\%) & Ablated (\%) & Single-layer (\%) & $\Delta$ random & KL \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]))

    rows = []
    for t in tags:
        d = H[t]
        q = d["test"]
        e = d["external"]
        rows.append(
            f"{SHORT[t]} & {d['selection']['layer']} & {pt(q['baseline_harmful'])} "
            f"& {pt(q['ablated_harmful'])} & {q['ablated_kl_harmless']:.3f} "
            f"& {pt(e['baseline_harmful'])} $\\to$ {pt(e['ablated_harmful'])} \\\\")
    write("calibration", "\n".join([
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Layer & Baseline (\%) & Ablated (\%) & KL & HarmBench (\%) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]))

    rows = []
    for t in tags:
        e = R[t]["external"]
        rows.append(
            f"{SHORT[t]} & {pt(e['baseline_harmful'])} & {pt(e['ablated_harmful'])} "
            f"& {pt(e['baseline_benign'])} & {pt(e['ablated_benign'])} "
            f"& {e['kl_benign']:.3f} \\\\")
    write("external", "\n".join([
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Brutal baseline (\%) & Brutal ablated (\%) & Benign baseline (\%) "
        r"& Benign ablated (\%) & KL \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]))

    rows = []
    for t in tags:
        s = R[t]["style_control"]
        rows.append(
            f"{SHORT[t]} & ${s['cos_with_refusal_direction']:+.3f}$ "
            f"& {pt(s['baseline_harmful'])} & {pt(s['ablation_harmful'])} "
            f"& {s['kl_harmless']:.3f} \\\\")
    write("style", "\n".join([
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & $\cos$ & Baseline (\%) & Style-ablated (\%) & KL \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]))

    rows = []
    for t in tags:
        d = R[t]
        sw = d["neurons"]["sweep"]
        # closest to 5% of the MLP
        row = min(sw, key=lambda r: abs(r["frac"] - 0.05))
        ml = d["neurons"]["multi_layer"][0]
        rows.append(
            f"{SHORT[t]} & {row['k']} & {100*row['frac']:.2f} & "
            f"{100*row['top_refusal_rate']:.1f} & {row['top_refusal_score']:+.2f} & "
            f"{row['random_refusal_score']:+.2f} & {100*ml['refusal_rate']:.1f} & "
            f"{ml.get('kl_harmless', float('nan')):.2f} \\\\")
    write("neurons", "\n".join([
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Model & $k$ & \% MLP & Top-$k$ rate (\%) & Top-$k$ score & Random score "
        r"& Multi-80 rate (\%) & Multi-80 KL \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]))

    rows = []
    for t in tags:
        add = {r["multiplier"]: r for r in R[t]["test"]["addition"]}
        rows.append(
            f"{SHORT[t]} & {pt(add[1.0])} & {add[1.0]['kl_vs_baseline']:.2f} "
            f"& {pt(add[2.0])} & {add[2.0]['kl_vs_baseline']:.2f} \\\\")
    write("addition", "\n".join([
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & 1.0$\times$ refusal (\%) & 1.0$\times$ KL & 2.0$\times$ refusal (\%) & 2.0$\times$ KL \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]))


if __name__ == "__main__":
    main()
