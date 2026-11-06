"""Generate the paper figures from results/."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = {
    "Qwen3.5-0.8B": "Qwen3.5-0.8B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "R1-Distill-1.5B",
    "LFM3-1.7B": "LFM3-1.7B",
    "Qwen3.5-4B": "Qwen3.5-4B",
}
ORDER = ["Qwen3.5-0.8B", "DeepSeek-R1-Distill-Qwen-1.5B", "LFM3-1.7B", "Qwen3.5-4B"]

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C = {"baseline": "#6b7280", "ablated": "#b91c1c", "single": "#d97706",
     "random": "#2563eb", "accent": "#111827"}


def load_json(pattern):
    out = {}
    for p in glob.glob(pattern):
        d = json.load(open(p))
        out[d["tag"]] = d
    return out


def main():
    os.makedirs("paper/figures", exist_ok=True)
    R = load_json("results/*/summary.json")
    H = load_json("results_hybrid/*/summary.json")
    tags = [t for t in ORDER if t in R]

    n = len(tags)
    nrows = 2 if n > 3 else 1
    ncols = 2 if n > 3 else n
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.1 * nrows),
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    for ax, t in zip(axes, tags):
        rows = list(csv.DictReader(open(f"results_hybrid/{t}/layer_sweep.csv")))
        xs = [int(r["layer"]) for r in rows]
        ys = [float(r["ablation_refusal_score"]) for r in rows]
        ax.plot(xs, ys, color=C["accent"], lw=1.4, marker="o", ms=2.5)
        L = H[t]["selection"]["layer"]
        ax.axvline(L, color=C["ablated"], ls="--", lw=1)
        ax.set_title(SHORT[t])
        ax.set_xlabel("layer")
        if ax is axes[0]:
            ax.set_ylabel("ablation refusal score")
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[len(tags):]:
        ax.axis("off")
    fig.savefig("paper/figures/layer_sweep.pdf")
    plt.close(fig)

    n = len(tags)
    nrows = 2 if n > 3 else 1
    ncols = 2 if n > 3 else n
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.1 * nrows),
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    for ax, t in zip(axes, tags):
        rows = R[t]["test"]["addition"]
        ms = [r["multiplier"] for r in rows]
        rate = [100 * r["refusal_rate"] for r in rows]
        kl = [r["kl_vs_baseline"] for r in rows]
        ax.plot(ms, rate, color=C["ablated"], marker="o", ms=3, lw=1.5, label="refusal")
        ax.set_xlabel("steering strength ($\\times\\|\\Delta\\mu\\|$)")
        if ax is axes[0]:
            ax.set_ylabel("harmless refusal (%)")
        ax.set_ylim(-2, 102)
        ax2 = ax.twinx()
        ax2.plot(ms, kl, color=C["random"], marker="s", ms=2.5, lw=1, ls="--", label="KL")
        ax2.set_ylabel("KL", color=C["random"])
        ax2.tick_params(axis="y", colors=C["random"])
        ax2.spines["right"].set_visible(True)
        ax.set_title(SHORT[t])
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[len(tags):]:
        ax.axis("off")
    fig.savefig("paper/figures/dose.pdf")
    plt.close(fig)

    n = len(tags)
    nrows = 2 if n > 3 else 1
    ncols = 2 if n > 3 else n
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.1 * nrows),
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    for ax, t in zip(axes, tags):
        rows = list(csv.DictReader(open(f"results/{t}/neurons.csv")))
        ks = [int(r["k"]) for r in rows]
        top = [100 * float(r["top_refusal_rate"]) for r in rows]
        base = 100 * R[t]["test"]["baseline_harmful"]["refusal_rate"]
        ax.plot(ks, top, color=C["ablated"], marker="o", ms=3, lw=1.5, label="top-$k$")
        ax.axhline(base, color=C["baseline"], ls=":", lw=1, label="baseline")
        ax.set_xscale("log")
        ax.set_xlabel("neurons clamped")
        if ax is axes[0]:
            ax.set_ylabel("refusal (%)")
        ax.set_ylim(-2, 102)
        ax.set_title(SHORT[t])
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[len(tags):]:
        ax.axis("off")
    axes[0].legend(frameon=False, loc="lower left")
    fig.savefig("paper/figures/neurons.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 2.4), constrained_layout=True)
    import numpy as np
    x = np.arange(len(tags))
    w = 0.26
    base = [100 * R[t]["test"]["baseline_harmful"]["refusal_rate"] for t in tags]
    abl = [100 * R[t]["test"]["ablated_harmful"]["refusal_rate"] for t in tags]
    single = [100 * R[t]["test"]["single_layer_ablation_harmful"]["refusal_rate"] for t in tags]
    ax.bar(x - w, base, w, color=C["baseline"], label="baseline")
    ax.bar(x, abl, w, color=C["ablated"], label="all-layer")
    ax.bar(x + w, single, w, color=C["single"], label="single-layer")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in tags], rotation=15, ha="right")
    ax.set_ylabel("harmful refusal (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.savefig("paper/figures/main_bars.pdf")
    plt.close(fig)

    print("figures written to paper/figures/")


if __name__ == "__main__":
    main()
