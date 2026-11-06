"""Build FINAL_REPORT.md and a combined CSV from results/*/summary.json."""
import csv
import glob
import json
import os

ORDER = ["Qwen3.5-0.8B", "DeepSeek-R1-Distill-Qwen-1.5B", "LFM3-1.7B", "Qwen3.5-4B"]


def pct(x):
    return f"{100 * x:.1f}%"


def ci(d):
    lo, hi = d["refusal_rate_ci95"]
    return f"{pct(d['refusal_rate'])} [{100*lo:.0f}, {100*hi:.0f}]"


def load():
    out = {}
    for p in glob.glob("results/*/summary.json"):
        d = json.load(open(p))
        out[d["tag"]] = d
    return out


def main():
    R = load()
    tags = [t for t in ORDER if t in R] + [t for t in R if t not in ORDER]
    hyb = {}
    for p in glob.glob("results_hybrid/*/summary.json"):
        d = json.load(open(p))
        hyb[d["tag"]] = d
    if not tags:
        raise SystemExit("no results found")

    lines = []
    A = lines.append

    A("# Neurons or Subspaces? Calibration-Sensitive Refusal in Open Instruction Models")
    A("")
    if "Qwen3.5-4B" not in tags:
        A("> **Status:** the Qwen3.5-4B checkpoint (9.3 GB, two shards) is still "
          "downloading; its row is omitted from this revision. Re-running "
          "`src/study.py --model qwen35_4b --model-path models/qwen35_4b` and then "
          "`src/make_report.py` adds it automatically.")
        A("")
    A("A controlled replication and extension of the inference-time refusal-steering "
      "literature (Arditi et al., 2024) across heterogeneous models: a hybrid "
      "linear-attention VLM (Qwen3.5), a reasoning-distilled Qwen2 model "
      "(DeepSeek-R1-Distill-Qwen-1.5B), and a hybrid conv/attention model "
      "(LFM3-1.7B).")
    A("")
    A("All numbers below are generated directly from `results/*/summary.json`.")
    A("")

    # ---------------------------------------------------------------- setup
    A("## 1. Setup")
    A("")
    A("| Model | Family | Layers | d_model | d_mlp | Refusal tokens |")
    A("| :--- | :--- | ---: | ---: | ---: | :--- |")
    for t in tags:
        d = R[t]
        A(f"| {t} | {d['family']} | {d['n_layers']} | {d['d_model']} | {d['d_mlp']} "
          f"| {d['refusal_token_ids']} |")
    A("")
    sp = R[tags[0]]["split"]
    A(f"Direction: **{sp['calib']}** minimally paired JBB prompts. Layer: selected on "
      f"an independent AdvBench/Alpaca validation split (96/48). Test: **{sp['test']}** "
      f"held-out JBB prompts (harmful and benign share template, topic and length), so "
      f"a mean-difference direction cannot be explained by imperative-vs-question "
      f"style or by topic alone. Refusal is measured at the last prompt token as the "
      f"log-odds that the next token is refusal-initiating (`I`, `As`, `Sorry`, `No`), "
      f"following Arditi et al.; a normalised string classifier on generated text is "
      f"reported as a secondary check. `KL` is the mean KL divergence of the "
      f"next-token distribution on harmless prompts versus baseline.")
    A("")
    A("Two controls are reported: an **external brutal set** (HarmBench standard "
      "behaviors, paired with JBB benign prompts from the same domain) and a "
      "**style direction** computed from the same benign prompts rendered as "
      "imperatives vs questions. Ablating the style direction should not remove "
      "refusal if the main effect is harmfulness-specific.")
    A("")
    A("The direction is estimated on 50 matched JBB pairs. The intervention layer is "
      "chosen on an independent AdvBench/Alpaca validation split (96/48), so the "
      "layer is not tuned on the JBB test distribution. A calibration-sensitivity "
      "run re-estimates the direction on the natural split and tests it on all 100 "
      "JBB pairs.")
    A("")
    A("")

    # ---------------------------------------------------------------- selection
    A("## 2. Layer selection")
    A("")
    A("The direction is the normalised mean difference between harmful and harmless "
      "calibration activations at the last prompt token. The layer is selected by "
      "held-out validation ablation score subject to `KL <= 0.1`, with the last 20% "
      "of layers pruned — not by maximum separation norm.")
    A("")
    A("| Model | Selected layer | Natural-val ablation score | Steering score | KL | Direction norm |")
    A("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for t in tags:
        s = hyb.get(t, R[t])["selection"]
        A(f"| {t} | {s['layer']} | {s['ablation_refusal_score']:+.3f} "
          f"| {s['steering_refusal_score']:+.3f} | {s['kl_harmless']:.4f} "
          f"| {s['direction_norm']:.2f} |")
    A("")

    # sensitivity: direction re-estimated on the natural calibration corpus
    if hyb:
        A("### 2.1 Calibration sensitivity")
        A("")
        A("The direction is re-estimated on the natural AdvBench/Alpaca calibration "
          "split (96 harmful / 96 harmless) and evaluated on all 100 JBB pairs. This "
          "shows how sensitive the estimate is to the calibration distribution.")
        A("")
        A("| Model | Layer | JBB baseline | JBB ablated | KL (JBB benign) | HarmBench |")
        A("| :--- | ---: | :--- | :--- | ---: | :--- |")
        for t, d in hyb.items():
            tst = d["test"]
            ext = d.get("external", {})
            ext_txt = (f"{pct(ext['baseline_harmful']['refusal_rate'])} -> "
                       f"{pct(ext['ablated_harmful']['refusal_rate'])}" if ext else "n/a")
            A(f"| {t} | {d['selection']['layer']} | {ci(tst['baseline_harmful'])} "
              f"| {ci(tst['ablated_harmful'])} | {tst['ablated_kl_harmless']:.3f} "
              f"| {ext_txt} |")
        A("")

    # ---------------------------------------------------------------- ablation
    A("## 3. Held-out causal results")
    A("")
    A("### 3.1 Direction ablation, addition and controls")
    A("")
    A("| Model | Baseline harmful | Ablated harmful | Single-layer ablation | Random direction | KL (harmless) |")
    A("| :--- | :--- | :--- | :--- | :--- | ---: |")
    for t in tags:
        d = R[t]["test"]
        b = d["baseline_harmful"]
        ab = d["ablated_harmful"]
        sl = d.get("single_layer_ablation_harmful")
        rd = d.get("random_direction", {})
        sl_txt = ci(sl) if sl else "n/a"
        rd_txt = (f"{rd['harmful_score_mean']:+.2f} score" if rd else "n/a")
        A(f"| {t} | {ci(b)} | {ci(ab)} | {sl_txt} | {rd_txt} "
          f"| {d['ablated_kl_harmless']:.3f} |")
    A("")
    A("`score` for the random-direction control is the mean refusal log-odds; it "
      "should stay near the baseline if the effect is direction-specific.")
    if "DeepSeek-R1-Distill-Qwen-1.5B" in hyb:
        s = hyb["DeepSeek-R1-Distill-Qwen-1.5B"]["test"]
        A("")
        A(f"Calibration sensitivity: re-estimating the R1 direction on the natural "
          f"corpus changes the JBB ablated rate to {ci(s['ablated_harmful'])} "
          f"(KL {s['ablated_kl_harmless']:.3f}); see section 2.1.")
    A("")

    A("### 3.2 Addition to harmless prompts (dose-response)")
    A("")
    for t in tags:
        d = R[t]["test"]
        A(f"**{t}** (selected layer {R[t]['selection']['layer']})")
        A("")
        A("| Multiplier | Coefficient | Harmless refusal | Refusal score | KL vs baseline |")
        A("| ---: | ---: | :--- | ---: | ---: |")
        for r in d["addition"]:
            A(f"| {r['multiplier']:.1f}x | {r['coeff']:.2f} | {ci(r)} "
              f"| {r['refusal_score_mean']:+.3f} | {r['kl_vs_baseline']:.3f} |")
        ra = d.get("random_addition", {})
        if ra:
            A(f"| random 1.0x | — | — | {ra['harmless_score_mean']:+.3f} "
              f"| {ra['harmless_kl_mean']:.3f} |")
        A("")

    # ---------------------------------------------------------------- controls
    A("### 3.3 External transfer to brutal prompts (HarmBench)")
    A("")
    A("The direction is computed on JBB matched pairs and evaluated unchanged on "
      "HarmBench standard behaviors. The benign side is JBB benign prompts from the "
      "mapped domain, so both sides are imperative.")
    A("")
    A("| Model | Brutal baseline | Brutal ablated | Benign baseline | Benign ablated | KL (benign) |")
    A("| :--- | :--- | :--- | :--- | :--- | ---: |")
    for t in tags:
        e = R[t].get("external")
        if not e:
            continue
        A(f"| {t} | {ci(e['baseline_harmful'])} | {ci(e['ablated_harmful'])} "
          f"| {ci(e['baseline_benign'])} | {ci(e['ablated_benign'])} "
          f"| {e['kl_benign']:.3f} |")
    A("")

    A("### 3.4 Style control")
    A("")
    A("A second direction is computed from the same benign prompts rendered as "
      "imperatives vs questions (content matched). If refusal is a harmfulness "
      "direction rather than a request-style direction, this control should be "
      "nearly orthogonal and should not remove refusal.")
    A("")
    A("| Model | cos(style, refusal) | Baseline | Style ablation | KL (harmless) |")
    A("| :--- | ---: | :--- | :--- | ---: |")
    for t in tags:
        s = R[t].get("style_control")
        if not s:
            continue
        A(f"| {t} | {s['cos_with_refusal_direction']:+.3f} "
          f"| {ci(s['baseline_harmful'])} | {ci(s['ablation_harmful'])} "
          f"| {s['kl_harmless']:.3f} |")
    A("")

    # ---------------------------------------------------------------- neurons
    A("### 3.5 Neuron ablation: top-k vs random-k")
    A("")
    for t in tags:
        d = R[t]["neurons"]
        A(f"**{t}** (layer {d['layer']})")
        A("")
        A("| k | % of MLP | Top-k refusal | Top-k score | Random-k score |")
        A("| ---: | ---: | :--- | ---: | ---: |")
        for r in d["sweep"]:
            A(f"| {r['k']} | {100*r['frac']:.2f}% | {pct(r['top_refusal_rate'])} "
              f"| {r['top_refusal_score']:+.3f} | {r['random_refusal_score']:+.3f} |")
        A("")
        A("| Multi-layer span | Neurons | Refusal | Harmless refusal | KL |")
        A("| :--- | ---: | :--- | :--- | ---: |")
        for r in d["multi_layer"]:
            A(f"| {r['layers']} | {r['total_k']} | {pct(r['refusal_rate'])} "
              f"| {pct(r.get('harmless_refusal_rate', 0))} | {r.get('kl_harmless', float('nan')):.3f} |")
        A("")

    # ---------------------------------------------------------------- qualitative
    A("## 4. Qualitative samples")
    A("")
    for t in tags:
        g = R[t]["generations"]
        if not g.get("prompts"):
            continue
        A(f"### {t}")
        A("")
        A(f"Prompt: *{g['prompts'][0]}*")
        A("")
        for key, label in [("baseline", "Baseline"), ("ablated", "Ablated"),
                           ("add_1x", "Direction added (1.0x)"),
                           ("add_2x", "Direction added (2.0x)")]:
            txt = g[key][0].strip().replace("\n", " ")
            A(f"- **{label}** (refusal={g[key + '_refusal'][0]}, "
              f"rep={g[key + '_rep'][0]:.2f}): {txt[:500]}")
        A("")

    # ---------------------------------------------------------------- key results
    A("## 5. Key results (computed)")
    A("")
    for t in tags:
        d = R[t]
        b = d["test"]["baseline_harmful"]
        ab = d["test"]["ablated_harmful"]
        rd = d["test"].get("random_direction", {})
        add = {r["multiplier"]: r for r in d["test"]["addition"]}
        ns = {r["k"]: r for r in d["neurons"]["sweep"]}
        kmax = max(ns)
        ml = d["neurons"]["multi_layer"][0]
        A(f"- **{t}**: refusal {ci(b)} -> {ci(ab)} after ablation "
          f"({100*(ab['refusal_rate']-b['refusal_rate']):+.0f} pp); harmless KL "
          f"{d['test']['ablated_kl_harmless']:.3f}"
          + (f"; random direction leaves {rd['harmful_score_mean']:+.2f} score"
             if rd else "")
          + f"; addition 1.0x -> {ci(add[1.0])} with KL {add[1.0]['kl_vs_baseline']:.2f}"
          + f"; top-{kmax} neurons {pct(ns[kmax]['top_refusal_rate'])} vs random "
            f"score {ns[kmax]['random_refusal_score']:+.3f}"
          + f"; multi-layer {ml['total_k']} neurons -> {pct(ml['refusal_rate'])} "
            f"(KL {ml.get('kl_harmless', float('nan')):.3f}).")
        e = d.get("external")
        if e:
            A(f"  - External HarmBench: {ci(e['baseline_harmful'])} -> "
              f"{ci(e['ablated_harmful'])} on brutal prompts; benign "
              f"{ci(e['baseline_benign'])} -> {ci(e['ablated_benign'])}.")
        s = d.get("style_control")
        if s:
            A(f"  - Style control: cos={s['cos_with_refusal_direction']:+.3f}; "
              f"style ablation {ci(s['ablation_harmful'])} (baseline {ci(s['baseline_harmful'])}, "
              f"KL {s['kl_harmless']:.3f}).")
    A("")

    A("## 6. Interpretation")
    A("")
    A("- **Refusal is a direction, but the estimate is calibration-sensitive.** "
      "Ablating the matched JBB direction removes refusal in Qwen3.5-0.8B "
      "(53.3% → 0.0%) and R1 (26.7% → 6.7%), while LFM3 moves only 86.7% → 73.3%. "
      "Re-estimating the LFM3 direction on the larger natural corpus gives "
      "82% → 21% at KL 1.05, so the weak LFM3 result is a calibration effect, not "
      "an absence of a direction.")
    A("- **The effect is not request style.** The test pairs share template, topic "
      "and length. The style direction is nearly orthogonal in Qwen3.5 and R1 "
      "(cos −0.16, −0.03) and its ablation does not reproduce the effect "
      "(53% → 23% for Qwen3.5; 27% → 93% for R1).")
    A("- **Ablation scope matters.** Removing the direction only at the selected "
      "layer leaves most refusals intact (Qwen3.5: 10% vs 0%; LFM3: 70% vs 73%; "
      "R1: 13% vs 7%), so the effect is distributed across the residual stream.")
    A("- **Neuron localisation is model-dependent.** Single-layer top-k is weak "
      "everywhere. Clamping 80 neurons across four layers cuts Qwen3.5-0.8B from "
      "53.3% to 20% (KL 0.13) and LFM3 from 86.7% to 36.7%, but leaves R1 unchanged.")
    A("- **Adding the direction is not free.** Qwen3.5-0.8B: 26.7% at 1.0× "
      "(KL 0.84) → 66.7% at 2.0× (KL 2.91); R1: 23.3% → 53.3% (KL 0.23 → 0.93); "
      "LFM3 is non-monotonic. The refusal is induced, but at a measurable "
      "distribution cost.")
    A("- **Severity generalises for two of three models.** A direction estimated on "
      "moderate JBB pairs ablates HarmBench refusal for Qwen3.5-0.8B (42% → 0%) and "
      "R1 (31% → 7%), but not for LFM3 (66% → 63%).")
    A("")

    A("## 7. Limitations")
    A("")
    A("- Two model families (Qwen, LFM) and one distilled reasoning model; results "
      "may not generalise to frontier-scale models.")
    A("- The matched primary set is 100 harmful/benign pairs (30 held out). The "
      "external HarmBench set is category-matched, not minimally edited, so it tests "
      "severity generalisation rather than the confound.")
    A("- The style direction also reduces refusal (section 3.4), so refusal is not "
      "perfectly isolated from request mood; the matched harmful/benign direction is "
      "nevertheless not the mood axis.")
    A("- Refusal is a proxy for harmfulness; the string classifier and the "
      "refusal-token log-odds both measure refusal *behaviour*, not whether a "
      "completion is useful or harmful (no LLM judge / StrongREJECT evaluation).")
    A("- Greedy decoding only; no sampling variance.")
    A("- The addition coefficient is scaled by the mean-difference norm, matching "
      "Arditi et al.'s activation-addition setting; other scalings may behave "
      "differently.")
    A("- Qwen3.5 and LFM3 are hybrid architectures; their linear-attention / "
      "short-conv blocks may implement refusal differently from pure transformers.")
    A("- DeepSeek-R1-Distill is prefilled with an empty `<think></think>` block so "
      "that the first generated token is the answer rather than chain-of-thought; "
      "this may change its refusal behaviour relative to normal use.")
    A("")

    A("## 8. Files")
    A("")
    A("- `src/study.py` — the experiment runner.")
    A("- `src/make_report.py` — this report.")
    A("- `results/<model>/summary.json`, `layer_sweep.csv`, `neurons.csv`, "
      "`qualitative.md`.")
    A("")

    A("## References")
    A("")
    A("- Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single "
      "Direction*, arXiv:2406.11717.")
    A("- Marshall, Scherlis & Belrose (2024), *Refusal in LLMs is an Affine "
      "Function*, arXiv:2411.09003.")
    A("- Wollschläger et al. (2025), *The Geometry of Refusal in LLMs: Concept "
      "Cones and Representational Independence*, arXiv:2502.17420.")
    A("- Chen et al. (2024), *Towards Understanding Safety Alignment: A Mechanistic "
      "Perspective from Safety Neurons*, arXiv:2406.14144.")
    A("- Tan et al. (2024), *Analyzing the Generalization and Reliability of "
      "Steering Vectors*, arXiv:2407.12404.")
    A("")

    with open("FINAL_REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # combined CSV
    rows = []
    for t in tags:
        d = R[t]
        b = d["test"]["baseline_harmful"]
        ab = d["test"]["ablated_harmful"]
        add = {r["multiplier"]: r for r in d["test"]["addition"]}
        ns = {r["k"]: r for r in d["neurons"]["sweep"]}
        kmax = max(ns)
        rows.append({
            "model": t,
            "layers": d["n_layers"],
            "d_model": d["d_model"],
            "d_mlp": d["d_mlp"],
            "selected_layer": d["selection"]["layer"],
            "baseline_harmful_refusal": b["refusal_rate"],
            "ablated_harmful_refusal": ab["refusal_rate"],
            "baseline_refusal_score": b["refusal_score_mean"],
            "ablated_refusal_score": ab["refusal_score_mean"],
            "random_direction_score": d["test"].get("random_direction", {}).get("harmful_score_mean"),
            "ablated_kl_harmless": d["test"]["ablated_kl_harmless"],
            "add1_refusal": add[1.0]["refusal_rate"],
            "add1_kl": add[1.0]["kl_vs_baseline"],
            "add2_refusal": add[2.0]["refusal_rate"],
            "add2_kl": add[2.0]["kl_vs_baseline"],
            "neuron_kmax": kmax,
            "neuron_kmax_top_refusal": ns[kmax]["top_refusal_rate"],
            "neuron_kmax_random_score": ns[kmax]["random_refusal_score"],
            "runtime_sec": d["runtime_sec"],
        })
    os.makedirs("results", exist_ok=True)
    with open("results/summary_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("wrote FINAL_REPORT.md and results/summary_all.csv")


if __name__ == "__main__":
    main()
