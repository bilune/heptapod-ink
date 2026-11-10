#!/usr/bin/env python3
"""Does the ring carry meaning? Two frozen-model tests (debate 4, round 4). Local, small.

Stimuli (written fresh for this test): frames with one slot filled by a TARGET word and three
alternatives: a SYNONYM, a CONTRAST (antonym for adjectives/verbs, same-category item for nouns)
and an UNRELATED word of the same part of speech.

Test 1 (causal, decoding): encode the four versions of each frame. Under teacher forcing with the
frame's own words, measure the log-probability the reader gives to each candidate word at the
slot, given each ring. The context prior is the same for all rings, so the ring's contribution is
    boost(w | ring X) = log p(w | ring X) - log p(w | ring of the UNRELATED version)
Semantic information in the ring predicts boost(synonym | ring TARGET) > boost(contrast | ring TARGET).
Run symbolically and through the rendered image + eyes.

Test 2 (geometry): Hamming distance between rings (36 slots) of target vs synonym / contrast /
unrelated versions of the same frame; and for paraphrase triplets, sentence vs paraphrase
(different words, same meaning) vs role reversal (same words, different meaning) vs a random
other sentence.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402

# (target, synonym, contrast, unrelated)
WORDS = {
    "adj": [("big", "large", "small", "wooden"), ("happy", "glad", "sad", "metal"), ("fast", "quick", "slow", "purple"),
            ("cold", "chilly", "hot", "legal"), ("quiet", "silent", "loud", "square"), ("rich", "wealthy", "poor", "frozen"),
            ("easy", "simple", "hard", "plastic"), ("angry", "mad", "calm", "round"), ("tired", "sleepy", "awake", "yellow"),
            ("smart", "clever", "stupid", "empty")],
    "noun": [("boat", "ship", "car", "idea"), ("house", "home", "tent", "answer"), ("road", "street", "river", "song"),
             ("doctor", "physician", "teacher", "window"), ("shop", "store", "bank", "cloud"), ("child", "kid", "adult", "spoon"),
             ("gift", "present", "bill", "planet"), ("bread", "loaf", "soup", "engine"), ("city", "town", "farm", "promise"),
             ("film", "movie", "book", "tooth")],
    "verb": [("began", "started", "ended", "painted"), ("shouted", "yelled", "whispered", "cooked"),
             ("bought", "purchased", "sold", "swam"), ("arrived", "came", "left", "melted")],
}
FRAMES = {
    "adj": ["The {} dog waited outside the door.", "It was a {} afternoon in the village.", "My neighbor seemed very {} today.",
            "They lived in a {} house near the lake."],
    "noun": ["We saw the {} from the window.", "She talked about the {} all evening.", "Nobody expected the {} to be there.",
             "He drew a picture of the {} yesterday."],
    "verb": ["The concert {} at nine in the evening.", "My uncle {} when he heard the news.", "The guests {} before the rain."],
}
PARA = [  # (sentence, paraphrase with different words, role reversal / meaning change with same words)
    ("The dog bit the man.", "The man was bitten by the dog.", "The man bit the dog."),
    ("Anna gave Peter a book.", "Peter received a book from Anna.", "Peter gave Anna a book."),
    ("The cat chased the mouse.", "The mouse was chased by the cat.", "The mouse chased the cat."),
    ("I am taller than my brother.", "My brother is shorter than me.", "My brother is taller than me."),
    ("The train left before we arrived.", "We arrived after the train had left.", "We left before the train arrived."),
    ("She sold her car to Tom.", "Tom bought her car from her.", "Tom sold his car to her."),
    ("The teacher praised the student.", "The student was praised by the teacher.", "The student praised the teacher."),
    ("It started raining after lunch.", "The rain began once we had eaten lunch.", "Lunch started after the rain."),
    ("Mark is older than Lisa.", "Lisa is younger than Mark.", "Lisa is older than Mark."),
    ("The bird flew over the house.", "The house was flown over by the bird.", "The house flew over the bird."),
    ("We won the game easily.", "The game was an easy win for us.", "The game easily won us."),
    ("My sister called me last night.", "I got a call from my sister last night.", "I called my sister last night."),
]


def wordpieces(tok, w):
    return tok(" " + w, add_special_tokens=False).input_ids


@torch.no_grad()
def word_logprob(m, probs, sentence_with_word, word):
    """log p(word pieces | ring, gold prefix) at the word's position, via teacher forcing."""
    tok = m.tok
    pre = sentence_with_word[:sentence_with_word.index(word)].rstrip()
    before = tok(pre, add_special_tokens=False).input_ids if pre else []
    upto = tok((pre + " " + word) if pre else word, add_special_tokens=False).input_ids
    full = tok(sentence_with_word).input_ids
    if full[:len(upto)] != upto or upto[:len(before)] != before:
        return None                                                      # tokenisation not separable: skip
    w_ids = upto[len(before):]
    labels = torch.tensor([full], device=m.dev)
    mem = m.memory(probs)
    lg = F.log_softmax(m.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem), labels=labels).logits.float(), -1)[0]
    return float(sum(lg[len(before) + k, t] for k, t in enumerate(w_ids)))


def main():
    torch.manual_seed(0)
    m, _ = H.load(Path("v4/runs/s2/model.pt"), H.device())
    m.eval()
    n = m.script.n_marks
    items = []
    for pos, quads in WORDS.items():
        for frame in FRAMES[pos]:
            for q in quads:
                items.append((pos, frame, q))
    roles = ["target", "synonym", "contrast", "unrelated"]
    t1 = {"symbolic": {r: [] for r in roles}, "visual": {r: [] for r in roles}}
    dist = {"synonym": [], "contrast": [], "unrelated": []}
    for pos, frame, q in items:
        sents = [frame.format(w) for w in q]
        code = m.code(sents)                                               # (4, 36) marks
        for k, name in ((1, "synonym"), (2, "contrast"), (3, "unrelated")):
            dist[name].append(int((code[0] != code[k]).sum()))
        sym = F.one_hot(code, n).float()
        with torch.no_grad():
            vis = m.eye_probs(m.draw(code))
        for path, P in (("symbolic", sym), ("visual", vis)):
            # log p(candidate w | ring of version X), candidates placed in the target frame
            lp = [[word_logprob(m, P[x:x + 1], frame.format(q[w]), q[w]) for w in range(4)] for x in range(4)]
            if any(v is None for row in lp for v in row):
                continue
            lp = np.array(lp)
            boost_from_target_ring = lp[0] - lp[3]                       # ring TARGET vs ring UNRELATED
            for w, r in enumerate(roles):
                t1[path][r].append(float(boost_from_target_ring[w]))
    res = {"n_items": len(items)}
    for path in t1:
        res[f"test1_{path}"] = {r: {"mean_boost": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3)} for r, v in t1[path].items()}
        syn, con, unr = (np.array(t1[path][r]) for r in ("synonym", "contrast", "unrelated"))
        d = syn - con
        boot = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
        res[f"test1_{path}"]["synonym_minus_contrast"] = {"mean": round(float(d.mean()), 3),
                                                          "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                                                          "share_items_positive": round(float((d > 0).mean()), 3)}
    res["test2_word_hamming"] = {k: {"mean": round(float(np.mean(v)), 2), "sd": round(float(np.std(v)), 2)} for k, v in dist.items()}
    rng = random.Random(0)
    allsents = [p for t in PARA for p in t]
    pd = {"paraphrase": [], "role_reversal": [], "random_other": []}
    for s, para, rev in PARA:
        c = m.code([s, para, rev, rng.choice([x for x in allsents if x not in (s, para, rev)])])
        pd["paraphrase"].append(int((c[0] != c[1]).sum()))
        pd["role_reversal"].append(int((c[0] != c[2]).sum()))
        pd["random_other"].append(int((c[0] != c[3]).sum()))
    res["test2_sentence_hamming"] = {k: {"mean": round(float(np.mean(v)), 2), "values": v} for k, v in pd.items()}
    reads = m.generate(F.one_hot(m.code([p for t in PARA for p in t]), n).float())
    res["paraphrase_reads_exact"] = round(float(np.mean([H.norm(a) == H.norm(b) for a, b in zip(reads, [p for t in PARA for p in t])])), 3)
    Path("v4/runs/s2/semantic_probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
