"""Zero-shot evaluation tasks, scored by log-likelihood (no fine-tuning, no prompting tricks).

1. ``story_cloze`` -- HellaSwag-style 4-way multiple choice built from *held-out*
   validation stories: the context is a story minus its last sentence; the
   choices are the true last sentence and the last sentences of three other
   random validation stories. A model that tracks characters, objects and plot
   should prefer the true ending. Reported as ``acc`` (sum of log-probs) and
   ``acc_norm`` (log-prob per character of the ending, as in HellaSwag).
   Chance = 25 %.

2. ``grammar`` -- minimal pairs (BLiMP-style; Warstadt et al. 2020) generated
   from templates: the model must assign higher probability to the grammatical
   sentence. Phenomena: subject-verb agreement (singular / plural), a/an,
   pronoun-gender consistency with a named character, past-tense consistency,
   and reflexive pronoun agreement. Chance = 50 %.
"""
import random
import re

import torch

NAMES_BOY = ["Tom", "Ben", "Max", "Sam", "Tim", "Jack", "Leo", "Bob"]
NAMES_GIRL = ["Lily", "Sue", "Mia", "Anna", "Lucy", "Emma", "Amy", "Zoe"]
NOUNS_SG = ["dog", "cat", "bird", "boy", "girl", "car", "tree", "ball", "frog", "bear"]
NOUNS_PL = ["dogs", "cats", "birds", "boys", "girls", "cars", "trees", "balls", "frogs", "bears"]
ADJ = ["happy", "sad", "big", "small", "red", "tired", "kind", "funny"]
VOWEL_NOUNS = ["apple", "egg", "owl", "ant", "elephant", "orange", "ice cream", "umbrella"]
CONS_NOUNS = ["dog", "ball", "cake", "toy", "hat", "box", "kite", "book"]
PLACES = ["park", "beach", "store", "forest", "garden", "lake"]


def grammar_pairs(seed: int = 0):
    """List of (phenomenon, good, bad)."""
    r = random.Random(seed)
    out = []
    for n_sg, n_pl in zip(NOUNS_SG, NOUNS_PL):
        for a in r.sample(ADJ, 3):
            out.append(("agreement_sg", f"The {n_sg} was very {a}.", f"The {n_sg} were very {a}."))
            out.append(("agreement_pl", f"The {n_pl} were very {a}.", f"The {n_pl} was very {a}."))
    for n in VOWEL_NOUNS:
        for name in r.sample(NAMES_BOY + NAMES_GIRL, 3):
            out.append(("a_an", f"{name} saw an {n}.", f"{name} saw a {n}."))
    for n in CONS_NOUNS:
        for name in r.sample(NAMES_BOY + NAMES_GIRL, 3):
            out.append(("a_an", f"{name} saw a {n}.", f"{name} saw an {n}."))
    for boy, girl in zip(NAMES_BOY, NAMES_GIRL):
        for p in r.sample(PLACES, 3):
            out.append(("pronoun_gender", f"{boy} went to the {p}. He was happy.",
                        f"{boy} went to the {p}. She was happy."))
            out.append(("pronoun_gender", f"{girl} went to the {p}. She was happy.",
                        f"{girl} went to the {p}. He was happy."))
    for name in NAMES_BOY + NAMES_GIRL:
        for p in r.sample(PLACES, 2):
            out.append(("past_tense", f"Yesterday, {name} went to the {p} and played.",
                        f"Yesterday, {name} go to the {p} and played."))
    for boy, girl in zip(NAMES_BOY, NAMES_GIRL):
        out.append(("reflexive", f"{boy} looked at himself in the mirror.",
                    f"{boy} looked at herself in the mirror."))
        out.append(("reflexive", f"{girl} looked at herself in the mirror.",
                    f"{girl} looked at himself in the mirror."))
    return out


_SENT = re.compile(r"(?<=[.!?])\s+")


def cloze_items(stories, n_items: int = 500, seed: int = 0):
    """[(context, [4 endings], gold_index)] from a list of story strings."""
    r = random.Random(seed)
    usable = []
    for s in stories:
        sents = [x.strip() for x in _SENT.split(s.strip()) if x.strip()]
        if 5 <= len(sents) <= 30 and 3 <= len(sents[-1].split()) <= 30:
            usable.append((" ".join(sents[:-1]), sents[-1]))
    r.shuffle(usable)
    items = []
    for i, (ctx, gold) in enumerate(usable[:n_items]):
        pool = []
        for _, e in r.sample(usable, min(len(usable), 12)):
            if e != gold and e not in pool:
                pool.append(e)
        pool = pool[:3]
        if len(pool) < 3:
            continue
        choices = pool + [gold]
        r.shuffle(choices)
        items.append((ctx, choices, choices.index(gold)))
    return items


@torch.no_grad()
def continuation_logprob(model, tok, context: str, endings, device, max_len: int):
    """Sum of log p(ending tokens | context) for each ending (batched, right-padded)."""
    ctx_ids = [tok.eos_id] + tok.encode(context)
    seqs, spans = [], []
    for e in endings:
        e_ids = tok.encode(" " + e if context else e)
        ids = (ctx_ids + e_ids)[-max_len:]
        seqs.append(ids)
        spans.append(len(e_ids))
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), tok.eos_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    x = x.to(device)
    logits = model(x[:, :-1]).float()
    lp = torch.log_softmax(logits, -1).gather(2, x[:, 1:, None]).squeeze(2)
    out = []
    for i, s in enumerate(seqs):
        n = len(s)
        out.append(float(lp[i, n - 1 - spans[i]:n - 1].sum()))
    return out


def run_cloze(model, tok, stories, device, n_items=500, max_len=512):
    items = cloze_items(stories, n_items)
    acc = acc_norm = 0
    for ctx, choices, gold in items:
        lps = continuation_logprob(model, tok, ctx, choices, device, max_len)
        norm = [lp / max(1, len(c)) for lp, c in zip(lps, choices)]
        acc += int(max(range(4), key=lambda i: lps[i]) == gold)
        acc_norm += int(max(range(4), key=lambda i: norm[i]) == gold)
    n = max(1, len(items))
    return {"n": len(items), "acc": acc / n, "acc_norm": acc_norm / n, "chance": 0.25}


def run_grammar(model, tok, device, max_len=512):
    pairs = grammar_pairs()
    per: dict = {}
    for ph, good, bad in pairs:
        g, b = continuation_logprob(model, tok, "", [good, bad], device, max_len)
        c, t = per.get(ph, (0, 0))
        per[ph] = (c + int(g > b), t + 1)
    total = sum(c for c, _ in per.values()) / len(pairs)
    return {"n": len(pairs), "acc": total, "chance": 0.5,
            "by_phenomenon": {k: round(c / t, 4) for k, (c, t) in per.items()}}
