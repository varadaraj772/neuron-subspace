"""
Prompt sets for the refusal study.

Primary (matched) set: JailbreakBench JBB-Behaviors.
  100 harmful / 100 benign prompts, paired by index and category.  The benign
  counterpart shares the template, topic and length of the harmful prompt, so a
  mean-difference direction cannot be explained by imperative-vs-question style
  or by topic alone.  (Chao et al., 2024, arXiv:2404.01318.)

External (brutal) set: HarmBench standard behaviors, excluding copyright.
  Paired with JBB benign prompts from the mapped category, so both sides are
  imperative and from the same domain family.  Used only as a held-out transfer
  test for a direction computed on JBB.

Style control: JBB benign imperatives vs Alpaca questions.  Both harmless; the
  direction between them is a pure style axis.  Ablating it should not remove
  refusal if the main effect is harmfulness-specific.
"""
import csv
import os
import random
import urllib.request

HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
                 "main/data/behavior_datasets/harmbench_behaviors_text_all.csv")

# HarmBench semantic category -> JBB benign categories used as counterparts
CATEGORY_MAP = {
    "cybercrime_intrusion": ["Malware/Hacking", "Privacy"],
    "chemical_biological": ["Expert advice", "Physical harm"],
    "illegal": ["Fraud/Deception", "Economic harm", "Government decision-making"],
    "misinformation_disinformation": ["Disinformation"],
    "harassment_bullying": ["Harassment/Discrimination"],
    "harmful": ["Physical harm", "Sexual/Adult content"],
}


def _jbb():
    from datasets import load_dataset
    h = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    b = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
    pairs = [(h[i]["Goal"], b[i]["Goal"], h[i]["Category"]) for i in range(len(h))]
    return pairs


def load_jbb(seed, n_calib, n_val, n_test):
    pairs = _jbb()
    rng = random.Random(seed)
    rng.shuffle(pairs)
    need = n_calib + n_val + n_test
    if need > len(pairs):
        raise ValueError(f"JBB has {len(pairs)} pairs, need {need}")
    pairs = pairs[:need]
    h = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    def split(xs):
        return (xs[:n_calib],
                xs[n_calib:n_calib + n_val],
                xs[n_calib + n_val:need])
    return split(h), split(b), pairs


def load_jbb_all(seed):
    """All 100 minimally paired JBB prompts, shuffled."""
    pairs = _jbb()
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _harmbench_rows():
    os.makedirs("data", exist_ok=True)
    path = "data/harmbench_behaviors.csv"
    if not os.path.exists(path):
        urllib.request.urlretrieve(HARMBENCH_URL, path)
    rows = list(csv.DictReader(open(path)))
    return [r for r in rows
            if r.get("FunctionalCategory") == "standard"
            and r.get("SemanticCategory") != "copyright"]


def load_brutal(seed, per_category=20):
    """HarmBench severe behaviors + JBB benign counterparts from the same domain."""
    rows = _harmbench_rows()
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["SemanticCategory"], []).append(r["Behavior"])
    rng = random.Random(seed + 7)
    harmful, benign = [], []
    for cat, prompts in by_cat.items():
        mapped = CATEGORY_MAP.get(cat)
        if not mapped:
            continue
        rng.shuffle(prompts)
        chosen = prompts[:per_category]
        pool = [p[1] for p in _jbb() if p[2] in mapped]
        if not pool:
            continue
        for i, p in enumerate(chosen):
            harmful.append(p)
            benign.append(pool[i % len(pool)])
    return harmful, benign


def _to_question(s):
    """Imperative -> question, keeping the content identical."""
    s = s.strip().rstrip(".")
    first = s.split()[0]
    verbs = {"Write", "Create", "Draft", "Develop", "Design", "Provide",
             "Explain", "Generate", "Compose", "Produce", "Give", "List",
             "Describe", "Outline", "Formulate", "Construct"}
    if first in verbs:
        rest = s[len(first):].lstrip()
        q = f"How would you {first.lower()} {rest}"
    else:
        q = f"How would you {s[0].lower() + s[1:]}"
    return q if q.endswith("?") else q + "?"


def load_style(seed, n=50):
    """The same benign prompts as imperatives and as questions (content matched)."""
    pairs = _jbb()
    rng = random.Random(seed + 11)
    imp = [p[1] for p in pairs]
    rng.shuffle(imp)
    imp = imp[:n]
    return imp, [_to_question(x) for x in imp]
