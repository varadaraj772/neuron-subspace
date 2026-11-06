# Neuron–Subspace

Sparse MLP neurons vs. linear residual subspaces as the locus of refusal in open
instruction models, evaluated on matched harmful/benign pairs and brutal HarmBench
prompts.

Inference-time representation engineering and mechanistic interpretability on
heterogeneous instruction models:

| Key | Model | Architecture |
| :--- | :--- | :--- |
| `qwen35_08b` | `Qwen/Qwen3.5-0.8B` | hybrid linear/full attention VLM (text stack used) |
| `r1distill_15b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Qwen2 transformer, reasoning-distilled |
| `lfm3_17b` | `LiquidAI/LFM3-1.7B` | hybrid short-conv / attention |
| `qwen35_4b` | `Qwen/Qwen3.5-4B` | hybrid linear/full attention VLM (text stack used) |

All weights are frozen. Every intervention is an inference-time forward hook.

## What is measured

The matched test set is **JailbreakBench JBB-Behaviors**: 100 harmful prompts with
100 benign counterparts that share the template, topic and length (minimal edit).
This controls the imperative-vs-question style confound that affects
AdvBench-vs-Alpaca designs. HarmBench standard behaviors are used as an external,
more severe held-out set, and a content-matched imperative-vs-question direction is
used as a style control. The direction is estimated on 50 JBB pairs; the layer is
chosen on an independent AdvBench/Alpaca validation split (96/48); the main test is
30 held-out JBB pairs.

For each model:

1. **Refusal direction** — normalised mean difference between harmful and harmless
   residual-stream activations at the last prompt token.
2. **Layer selection** — held-out validation ablation score, subject to
   `KL <= 0.1` on harmless prompts, with the last 20% of layers pruned
   (Arditi et al.'s protocol, not "maximum separation norm").
3. **Held-out ablation** — project the direction out of the embedding and every
   block output; measure the refusal log-odds at the next token.
4. **Addition dose-response** — add `0 … 2×` the mean-difference vector at the
   selected layer and measure refusal on harmless prompts, plus KL divergence.
5. **Neuron ablation** — rank MLP intermediate neurons by their projection onto the
   refusal direction; compare top-k against random-k, including a 5% budget and a
   four-layer clamp.
6. **Controls** — a random unit direction of the same norm (ablation and addition);
   single-layer vs all-layer ablation.
7. **Coherence** — repetition and distinct-bigram scores on generated text.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`Qwen3.5` checkpoints are multimodal; the loader uses
`Qwen3_5ForConditionalGeneration` and takes the text stack
(`model.model.language_model`). `LFM3` uses the model's remote code
(`trust_remote_code=True`). `DeepSeek-R1-Distill` is prefilled with an empty
`<think></think>` block so that the first generated token is the answer, not
reasoning.

## Run

```bash
# all four models + report
./run_all.sh

# one model
python src/study.py --model qwen35_08b

# smoke test
python src/study.py --model qwen35_08b --quick
```

Useful flags:

- `--model-path models/r1` — load a local checkpoint directory instead of the Hub id.
- `--force-layer 9` — skip selection (sensitivity run).
- `--kl-max 0.1` — KL cutoff for layer selection.
- `--out results_layer9` — separate output directory.

## Datasets

| Set | Source | Use |
| :--- | :--- | :--- |
| Layer selection | `mlabonne/harmful_behaviors` vs `harmless_alpaca` | 96 / 48 prompts per class |
| Direction calibration | `JailbreakBench/JBB-Behaviors` | 50 matched pairs |
| Matched test | `JailbreakBench/JBB-Behaviors` | 30 held-out pairs |
| Calibration sensitivity | natural direction, all JBB pairs | 100 pairs (`results_hybrid/`) |
| Brutal external | HarmBench standard behaviors | 119 held-out brutal prompts + domain-matched benign |
| Style control | JBB benign, imperative vs question | tests whether refusal is request mood |

## Outputs

```
results/<model>/
  summary.json        full metrics
  layer_sweep.csv     validation sweep
  neurons.csv         top-k vs random-k
  qualitative.md      generation samples
results/summary_all.csv
FINAL_REPORT.md
```

## Method notes

- Refusal is the log-odds that the next token belongs to a refusal-initiating set
  (`I`, `As`, `Sorry`, `No`), plus a normalised string classifier on generations.
- A binary refusal rate is `P(refusal token) > 0.5`; Wilson 95% intervals are
  reported.
- The addition coefficient is `multiplier × ||mean-difference||`, matching
  Arditi et al.'s activation-addition scale.
- Calibration, validation and test prompts are disjoint; there is no silent
  fallback dataset.

## References

- Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*, arXiv:2406.11717.
- Marshall, Scherlis & Belrose (2024), *Refusal in LLMs is an Affine Function*, arXiv:2411.09003.
- Wollschläger et al. (2025), *The Geometry of Refusal in LLMs: Concept Cones and Representational Independence*, arXiv:2502.17420.
- Chen et al. (2024), *Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons*, arXiv:2406.14144.
- Tan et al. (2024), *Analyzing the Generalization and Reliability of Steering Vectors*, arXiv:2407.12404.
