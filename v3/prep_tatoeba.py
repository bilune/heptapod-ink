"""Build the open-English dataset from Tatoeba (CC BY 2.0).

Sentences up to --max-words words, deduplicated case-insensitively, keeping
only those the MiniLM WordPiece vocabulary can write without [UNK] and within
the token limit. Random splits: 3,000 dev, 3,000 test, the rest train.
Also writes fixed "random word combination" sets for dev and test (words
sampled from the train vocabulary), to measure reading of arbitrary word
sequences separately from natural sentences.
"""
import collections, json, random, sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "research" / "tatoeba" / "eng_sentences.tsv"
OUT = ROOT / "v3" / "data_tatoeba"
MAX_WORDS, MAX_TOKENS = 12, 38
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

seen, keep = set(), []
for line in SRC.read_text(encoding="utf-8").splitlines():
    parts = line.split("\t")
    if len(parts) < 3:
        continue
    s = " ".join(parts[2].split())
    if not s or len(s.split()) > MAX_WORDS:
        continue
    k = s.lower()
    if k in seen:
        continue
    seen.add(k)
    keep.append(s)
print("candidates", len(keep), flush=True)
ok = []
for i in range(0, len(keep), 4096):
    batch = keep[i:i + 4096]
    enc = tok(batch, add_special_tokens=True)["input_ids"]
    for s, ids in zip(batch, enc):
        if len(ids) <= MAX_TOKENS and tok.unk_token_id not in ids:
            ok.append(s)
print("writable", len(ok), flush=True)
rng = random.Random(0)
rng.shuffle(ok)
dev, test, train = ok[:3000], ok[3000:6000], ok[6000:]
freq = collections.Counter(w.lower().strip(".,!?\"';:") for s in train for w in s.split())
words = [w for w, c in freq.items() if w and c >= 3 and w.isalpha()]
weights = [freq[w] ** 0.5 for w in words]

def random_set(n, seed):
    r = random.Random(seed)
    return [" ".join(r.choices(words, weights=weights, k=r.randint(3, 10))) for _ in range(n)]

def dump(name, rows, cat):
    (OUT / f"{name}.json").write_text(json.dumps([{"sentence": s, "category": cat} for s in rows]))

dump("train", train, "tatoeba")
dump("dev", dev, "tatoeba")
dump("test", test, "tatoeba")
dump("dev_random", random_set(1000, 1), "random")
dump("test_random", random_set(1000, 2), "random")
json.dump({"words": words, "weights": weights}, open(OUT / "random_vocab.json", "w"))
print({"train": len(train), "dev": len(dev), "test": len(test), "random_vocab": len(words)})
print("examples:", train[:5], random_set(3, 9))
