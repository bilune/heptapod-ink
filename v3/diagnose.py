"""Where does a trained model fail on unseen sentences? (local, read-only)"""
import sys, json, random, collections
from pathlib import Path
import torch
sys.path.insert(0, "v3")
import heptapod as V

ckpt_path = Path(sys.argv[1])
V.OUT = ckpt_path.parent / "diag"
V.OUT.mkdir(exist_ok=True)
(V.OUT / "model.pt").unlink(missing_ok=True)
(V.OUT / "model.pt").symlink_to(ckpt_path.resolve())
dev = V.device()
model = V.load(dev)
train = V.load_split("train")
test = V.load_split("test")
freq = collections.Counter(w for r in train for w in V.normalize(r["text"]).replace(".", " .").split())

@torch.no_grad()
def codes_and_preds(rows):
    out = []
    for i in range(0, len(rows), 128):
        chunk = rows[i:i + 128]
        batch = model.tokenize([r["text"] for r in chunk])
        rc, mc = model.writer(*model.read_text(batch))
        img = model.script(rc, mc)
        pred = model.decode(model.reader.generate(img, model.bos, model.eos))
        code = torch.stack([rc.argmax(-1), mc.argmax(-1)], -1).cpu()
        for r, p, c in zip(chunk, pred, code):
            out.append((V.normalize(r["text"]), p, tuple(c.flatten().tolist())))
    return out

res = codes_and_preds(test)
exact = sum(g == p for g, p, _ in res) / len(res)
# 1. code collisions among test sentences, and with train
by_code = collections.defaultdict(set)
for g, _, c in res:
    by_code[c].add(g)
coll = sum(len(v) for v in by_code.values() if len(v) > 1)
train_res = codes_and_preds(random.Random(0).sample(train, 3000))
train_codes = {c: g for g, _, c in train_res}
same_as_train = sum(1 for g, _, c in res if c in train_codes and train_codes[c] != g)
# 2. word errors by training frequency
err_by_bucket = collections.defaultdict(lambda: [0, 0])
swaps = collections.Counter()
n_wrong_words = collections.Counter()
for g, p, _ in res:
    gw, pw = g.replace(".", " .").split(), p.replace(".", " .").split()
    wrong = 0
    for i, w in enumerate(gw):
        ok = i < len(pw) and pw[i] == w
        f = freq[w]
        b = "1-9" if f < 10 else "10-99" if f < 100 else "100-999" if f < 1000 else "1000+"
        err_by_bucket[b][0] += not ok
        err_by_bucket[b][1] += 1
        if not ok:
            wrong += 1
            if i < len(pw):
                swaps[(w, pw[i])] += 1
    if len(gw) != len(pw):
        wrong += 1
    n_wrong_words[min(wrong, 3)] += 1
report = {
    "test_exact": round(exact, 4),
    "test_code_collisions_sentences": coll,
    "test_codes_identical_to_a_different_train_sentence": same_as_train,
    "word_error_rate_by_train_frequency": {k: round(v[0] / v[1], 3) for k, v in sorted(err_by_bucket.items())},
    "sentences_by_number_of_wrong_words": {str(k): v for k, v in sorted(n_wrong_words.items())},
    "top_swaps(gold->pred)": [f"{a}->{b} x{n}" for (a, b), n in swaps.most_common(20)],
    "train_sample_exact": round(sum(g == p for g, p, _ in train_res) / len(train_res), 4),
}
print(json.dumps(report, indent=1))
fails = [(g, p) for g, p, _ in res if g != p]
for g, p in random.Random(1).sample(fails, 12):
    print("GOLD:", g, "\nPRED:", p, "\n")
