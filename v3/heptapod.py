#!/usr/bin/env python3
"""Heptapod Ink v3: a model that writes and reads Heptapod-style logograms.

text -> frozen MiniLM (contextual token vectors) -> Writer -> one ring logogram
in the film's ink -> Reader (CNN + transformer decoder) -> text.

Trained end to end on one objective: the text that comes out must match the
text that went in. Nothing about grammar or meaning structure is hand-coded;
the only fixed part is the script itself (a ring of 24 slots, each holding a
real ring segment and a real ink mark cut from the 38 Arrival logograms).

Usage (from the project root, with the project venv):
  .venv/bin/python v3/heptapod.py train --steps 30000
  .venv/bin/python v3/heptapod.py say "a girl liked the donut"
  .venv/bin/python v3/heptapod.py read v3/out/say_a_girl_liked_the_donut.png
  .venv/bin/python v3/heptapod.py fresh-reader     # can a new reader learn it?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "v2"))
import heptapod_ink as INK  # noqa: E402  ring+mark renderer and ink library

OUT = HERE / "out"
DATA = HERE / "data"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
RES = 192
SLOTS = 24
MAX_TOKENS = 40
MARK_SCALE = 0.4
_HI = {}


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_all(seed: int):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)


# ---------------------------------------------------------------- data


def load_split(name: str) -> list[dict]:
    rows = json.loads((DATA / f"{name}.json").read_text())
    return [{"text": r["sentence"], "category": r["category"]} for r in rows]


def normalize(text: str) -> str:
    return " ".join(text.lower().replace(" .", ".").split())


# ---------------------------------------------------------------- the script


def upsample_library():
    """The v2 low-res library is 0.5 columns/degree; at 192 px we need 1/degree."""
    hi_r = INK.with_blank(INK.ring_rows(torch.tensor(INK.LIB["rings_hi"]).float() / 255)[INK.RING_PICK])
    hi_m = INK.with_blank(torch.tensor(INK.LIB["marks_hi"]).float() / 255)
    rad = 68
    rings = F.interpolate(hi_r[:, None], size=(rad, int(INK.RING_DEG)), mode="area")[:, 0]
    # MARK_SCALE < 1 condenses each real mark along the ring so 24 slots don't
    # bury each other; the ink is the film's, only narrower.
    marks = F.interpolate(hi_m[:, None], size=(rad, int(INK.MARK_DEG * MARK_SCALE)), mode="area")[:, 0]
    return rings, marks


# ---------------------------------------------------------------- writer


def st_gumbel(logits, tau, hard):
    u = torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)
    y = F.softmax((logits - torch.log(-torch.log(u))) / tau, -1)
    if not hard:
        return y
    one = F.one_hot(y.argmax(-1), y.shape[-1]).to(y.dtype)
    return one + y - y.detach()


class Writer(nn.Module):
    """Every slot looks at the whole sentence at once (no reading order)."""

    def __init__(self, n_rings, n_marks, d=256):
        super().__init__()
        self.inp = nn.Linear(384, d)
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, 2)
        self.queries = nn.Parameter(torch.randn(SLOTS, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.slots = nn.TransformerDecoder(dec, 2)
        self.ring_head = nn.Linear(d, n_rings)
        self.mark_head = nn.Linear(d, n_marks)
        self.n_rings = n_rings
        self.tau, self.hard = 1.0, True

    def logits(self, tok_vecs, tok_mask):
        h = self.encoder(self.inp(tok_vecs), src_key_padding_mask=~tok_mask)
        q = self.queries.expand(len(h), -1, -1)
        s = self.slots(q, h, memory_key_padding_mask=~tok_mask)
        return self.ring_head(s), self.mark_head(s)

    def forward(self, tok_vecs, tok_mask):
        lr, lm = self.logits(tok_vecs, tok_mask)
        if self.training:
            return st_gumbel(lr, self.tau, self.hard), st_gumbel(lm, self.tau, self.hard)
        return F.one_hot(lr.argmax(-1), lr.shape[-1]).float(), F.one_hot(lm.argmax(-1), lm.shape[-1]).float()


# ---------------------------------------------------------------- reader


class PolarEyes(nn.Module):
    """Circular eyes: unwrap the ring into a strip (angle across, radius down),
    then read along the angle with wrap-around padding. Like Wolfram's unwrap
    of the film logograms; no knowledge of slots or meaning."""

    def __init__(self, d=256, angles=384, radii=48, tokens=48):
        super().__init__()
        theta = torch.linspace(-math.pi, math.pi, angles + 1)[:-1]
        r = torch.linspace(0.20, 0.86, radii)  # ring at 0.50, ink spans 0.225-0.825
        tt, rr = torch.meshgrid(theta, r, indexing="xy")
        self.register_buffer("grid", torch.stack([rr * torch.cos(tt), rr * torch.sin(tt)], -1)[None])

        def block(i, o, pool):
            return nn.Sequential(CircConv(i, o), nn.BatchNorm2d(o), nn.GELU(),
                                 CircConv(o, o), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d(pool))

        self.cnn = nn.Sequential(block(1, 32, 2), block(32, 64, 2), block(64, 128, 2), block(128, d, (6, 1)))
        self.tokens = tokens  # 384/8 = 48 angular positions

    def forward(self, img):
        strip = F.grid_sample(img[:, None], self.grid.expand(len(img), -1, -1, -1), align_corners=False)
        f = self.cnn(strip)                    # (B, d, 1, 48)
        return f.mean(2).transpose(1, 2)        # (B, 48, d), one token per angle


class CircConv(nn.Module):
    """3x3 conv that wraps around in angle (the strip is a closed ring)."""

    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 3)

    def forward(self, x):
        x = torch.cat([x[..., -1:], x, x[..., :1]], -1)
        x = F.pad(x, (0, 0, 1, 1))
        return self.conv(x)


class Reader(nn.Module):
    """Sees only the image, writes the text token by token."""

    def __init__(self, vocab, d=256):
        super().__init__()
        self.cnn = PolarEyes(d)
        self.pos = nn.Parameter(torch.randn(48, d) * 0.02)       # angular position
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.img_enc = nn.TransformerEncoder(enc, 2)
        self.tok = nn.Embedding(vocab, d)
        self.tpos = nn.Parameter(torch.randn(MAX_TOKENS, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, 3)
        self.out = nn.Linear(d, vocab)
        self.bow = nn.Linear(d, vocab)  # which words are in the sentence, from the image alone

    def bag_of_words(self, img):
        return self.bow(self.memory(img).mean(1))

    def memory(self, img):
        return self.img_enc(self.cnn(img) + self.pos)

    def forward(self, img, tgt_in):
        mem = self.memory(img)
        t = tgt_in.shape[1]
        x = self.tok(tgt_in) + self.tpos[:t]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=img.device), 1)
        return self.out(self.dec(x, mem, tgt_mask=causal))

    @torch.no_grad()
    def generate(self, img, bos, eos, max_len=MAX_TOKENS):
        mem = self.memory(img)
        seq = torch.full((len(img), 1), bos, dtype=torch.long, device=img.device)
        done = torch.zeros(len(img), dtype=torch.bool, device=img.device)
        for _ in range(max_len - 1):
            t = seq.shape[1]
            x = self.tok(seq) + self.tpos[:t]
            causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=img.device), 1)
            nxt = self.out(self.dec(x, mem, tgt_mask=causal))[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            done |= nxt == eos
            if done.all():
                break
        return seq


# ---------------------------------------------------------------- channel


def channel(img, train: bool):
    """Fixed orientation like the film: only a slight hand tremor and ink noise."""
    if not train:
        return img
    b = img.shape[0]
    ang = (torch.rand(b, device=img.device) * 2 - 1) * math.radians(4)
    scale = 1 + (torch.rand(b, device=img.device) * 2 - 1) * 0.04
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    shift = (torch.rand(b, 2, device=img.device) * 2 - 1) * 0.03
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, (b, 1, RES, RES), align_corners=False)
    out = F.grid_sample(img[:, None], grid, align_corners=False)[:, 0]
    return (out + 0.1 * torch.randn_like(out)).clamp(0, 1)


# ---------------------------------------------------------------- system


class Heptapod(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        self.encoder = AutoModel.from_pretrained(ENCODER).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        rings, marks = upsample_library()
        INK.SLOTS, INK.PITCH_DEG = SLOTS, 360 / SLOTS
        self.script = INK.RingMarkRenderer(rings, marks, 1.0, RES)
        self.writer = Writer(len(rings), len(marks))
        self.reader = Reader(self.tokenizer.vocab_size)
        self.bos, self.eos, self.pad = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id
        self.to(dev)
        self.dev = dev

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # frozen: never in training mode (and MPS has no SDPA dropout)
        return self

    def tokenize(self, texts):
        b = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
        return {k: v.to(self.dev) for k, v in b.items()}

    @torch.no_grad()
    def read_text(self, batch):
        return self.encoder(**batch).last_hidden_state, batch["attention_mask"].bool()

    def write(self, texts):
        batch = self.tokenize(texts)
        vecs, mask = self.read_text(batch)
        rc, mc = self.writer(vecs, mask)
        return self.script(rc, mc), batch["input_ids"], (rc, mc)

    def decode(self, ids):
        return [normalize(self.tokenizer.decode(s, skip_special_tokens=True)) for s in ids.tolist()]


def target_loss(model, img, ids, word_dropout=0.0, reader=None):
    """Token cross-entropy. With word_dropout, some previous words fed to the
    decoder are hidden, so it cannot guess the sentence from its own prefix and
    has to get the information from the logogram."""
    reader = reader or model.reader
    tgt_in, tgt_out = ids[:, :-1].clone(), ids[:, 1:]
    if word_dropout > 0:
        hide = (torch.rand(tgt_in.shape, device=ids.device) < word_dropout) & (tgt_in != model.pad)
        hide[:, 0] = False  # keep [CLS] as start symbol
        tgt_in[hide] = model.tokenizer.mask_token_id
    logits = reader(img, tgt_in)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1), ignore_index=model.pad)


def bow_loss(model, img, ids, reader=None):
    reader = reader or model.reader
    target = torch.zeros(len(ids), model.tokenizer.vocab_size, device=ids.device)
    target.scatter_(1, ids, 1.0)
    for special in (model.pad, model.bos, model.eos):
        target[:, special] = 0
    return F.binary_cross_entropy_with_logits(reader.bag_of_words(img), target)


@torch.no_grad()
def information_check(model, rows):
    """Loss with the right logogram vs a logogram of another sentence.
    If they are equal, the drawing carries no information."""
    model.eval()
    img, ids, _ = model.write([r["text"] for r in rows])
    right = target_loss(model, img, ids).item()
    wrong = target_loss(model, img[torch.randperm(len(img))], ids).item()
    model.train()
    return round(right, 4), round(wrong, 4)


@torch.no_grad()
def evaluate(model, rows, batch=128, limit=None, reader=None):
    reader = reader or model.reader
    modes = [(m, m.training) for m in (model.writer, model.reader, reader)]
    model.eval()
    reader.eval()
    rows = rows[:limit] if limit else rows
    exact, by_cat, words_ok, words_total = 0, {}, 0, 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        img, _, _ = model.write([r["text"] for r in chunk])
        pred = model.decode(reader.generate(img, model.bos, model.eos))
        for r, p in zip(chunk, pred):
            gold = normalize(r["text"])
            ok = p == gold
            exact += ok
            gw, pw = gold.split(), p.split()
            words_ok += sum(a == b for a, b in zip(gw, pw))
            words_total += len(gw)
            c = by_cat.setdefault(r["category"], [0, 0])
            c[0] += ok
            c[1] += 1
    for m, was in modes:
        m.train(was)
    model.encoder.eval()
    by_cat = {k: round(v[0] / v[1], 4) for k, v in sorted(by_cat.items())}
    by_cat["_word_accuracy"] = round(words_ok / max(words_total, 1), 4)
    return round(exact / len(rows), 4), by_cat


def configure(mark_scale: float):
    global MARK_SCALE
    MARK_SCALE = mark_scale
    _HI.clear()


def train(args):
    configure(args.mark_scale)
    seed_all(args.seed)
    dev = device()
    model = Heptapod(dev)
    train_rows, dev_rows = load_split("train"), load_split("dev")
    params = list(model.writer.parameters()) + list(model.reader.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    OUT.mkdir(exist_ok=True)
    log = open(OUT / "train_log.jsonl", "a")
    print(f"device={dev} train={len(train_rows)} dev={len(dev_rows)} slots={SLOTS} res={RES}", flush=True)
    best, t0 = -1.0, time.time()
    frozen = False
    for step in range(1, args.steps + 1):
        frac = step / args.steps
        model.writer.tau = 2.0 * (0.25 ** frac)
        model.writer.hard = frac >= args.soft_until
        if args.freeze_writer and model.writer.hard and not frozen:
            # The soft phase found a code; from here the writer always picks its
            # most likely piece and only the reader keeps learning to read it.
            frozen = True
            for p in model.writer.parameters():
                p.requires_grad_(False)
            params = list(model.reader.parameters())
            opt = torch.optim.AdamW(params, lr=args.lr * 0.5, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps - step + 1)
            print({"step": step, "event": "writer frozen"}, flush=True)
        if frozen:
            model.writer.eval()
        rows = random.sample(train_rows, args.batch)
        if frozen:
            with torch.no_grad():
                img, ids, _ = model.write([r["text"] for r in rows])
        else:
            img, ids, _ = model.write([r["text"] for r in rows])
        seen = channel(img, True)
        loss = target_loss(model, seen, ids, word_dropout=args.word_dropout)
        bow = bow_loss(model, seen, ids)
        ink = img.mean()
        total = loss + args.bow_weight * bow * 100
        if not frozen:
            total = total + args.ink_weight * F.relu(ink - args.ink_max)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(), sched.step()
        if step % 100 == 0 or step == 1:
            row = {"step": step, "loss": round(loss.item(), 4), "bow": round(bow.item() * 100, 3),
                   "ink": round(ink.item(), 4),
                   "hard": model.writer.hard, "tau": round(model.writer.tau, 3),
                   "sec": round(time.time() - t0)}
            if step % 500 == 0:
                row["loss_right_vs_wrong_logogram"] = information_check(model, random.sample(dev_rows, 128))
            if step % args.eval_every == 0:
                row["dev_exact"], cats = evaluate(model, dev_rows, limit=512)
                row["dev_words"] = cats["_word_accuracy"]
                snapshot(model, step)
                if frozen:
                    model.writer.eval()
                if model.writer.hard and row["dev_exact"] > best:
                    best = row["dev_exact"]
                    save(model, args, OUT / "model.pt")
            print(row, flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
    if best < 0:
        save(model, args, OUT / "model.pt")
    report(args)


SNAPSHOT_TEXTS = ["A girl liked the donut .", "The donut was liked by a girl .", "Emma gave the cake to a dog .",
                  "A dog gave the cake to Emma .", "The boy needed to sleep .", "Liam said that a cat ate the cake on the table ."]


@torch.no_grad()
def snapshot(model, step):
    """Save the logograms of six fixed sentences, to watch the script form."""
    was = model.writer.training
    model.writer.eval()
    img, _, _ = model.write(SNAPSHOT_TEXTS)
    model.writer.train(was)
    folder = OUT / "snapshots"
    folder.mkdir(exist_ok=True)
    row = torch.cat(list(img.cpu()), 1).numpy()
    plt.imsave(folder / f"{step:06d}.png", 1 - row, cmap="gray", vmin=0, vmax=1)


def save(model, args, path):
    torch.save({"writer": model.writer.state_dict(), "reader": model.reader.state_dict(),
                "config": {**vars(args), "slots": SLOTS, "res": RES, "encoder": ENCODER,
                           "rings": int(model.writer.n_rings)}}, path)


def load(dev):
    ckpt = torch.load(OUT / "model.pt", map_location=dev)
    configure(ckpt["config"].get("mark_scale", 0.4))
    model = Heptapod(dev)
    model.writer.load_state_dict(ckpt["writer"])
    model.reader.load_state_dict(ckpt["reader"])
    return model.eval()


def report(args=None):
    dev = device()
    model = load(dev)
    out = {}
    for split in ("train", "dev", "test", "gen"):
        rows = load_split(split)
        if split == "train":
            rows = random.Random(0).sample(rows, 2000)
        out[split] = dict(zip(("exact", "by_category"), evaluate(model, rows)))
        out[split]["word_accuracy"] = out[split]["by_category"].pop("_word_accuracy")
        if split != "gen":
            out[split].pop("by_category")
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------- say / read


def hi_renderer(res: int = 1024):
    """Same library and geometry as training (24 slots, condensed marks), at high res."""
    if res not in _HI:
        INK.SLOTS, INK.PITCH_DEG = SLOTS, 360 / SLOTS
        rings = INK.with_blank(INK.ring_rows(torch.tensor(INK.LIB["rings_hi"]).float() / 255)[INK.RING_PICK])
        marks = INK.with_blank(torch.tensor(INK.LIB["marks_hi"]).float() / 255)
        cpd = float(INK.LIB["hi_cpd"])
        marks = F.interpolate(marks[:, None], size=(marks.shape[1], int(INK.MARK_DEG * MARK_SCALE * cpd)),
                              mode="area")[:, 0]
        _HI[res] = INK.RingMarkRenderer(rings, marks, cpd, res)
    return _HI[res]


def paint(rc, mc, res: int = 1024):
    img = hi_renderer(res)(rc, mc)
    img = ((img - 0.35) / 0.3).clamp(0, 1)
    return img * img * (3 - 2 * img)


@torch.no_grad()
def say(args):
    dev = device()
    model = load(dev)
    rc, mc = model.writer(*model.read_text(model.tokenize([args.text])))
    img = paint(rc.cpu(), mc.cpu())[0]
    name = "_".join(normalize(args.text).replace(".", "").split())[:80]
    path = Path(args.out) if args.out else OUT / f"say_{name}.png"
    plt.imsave(path, np.repeat((1 - img.numpy())[..., None], 3, -1))
    print(path)


@torch.no_grad()
def read(args):
    """Read an exported PNG the same way the reader was trained: downsample to 192."""
    dev = device()
    model = load(dev)
    from PIL import Image
    g = np.asarray(Image.open(args.image).convert("L").resize((RES, RES), Image.LANCZOS)) / 255.0
    img = torch.tensor(1 - g, dtype=torch.float32, device=dev)[None]
    print(model.decode(model.reader.generate(img, model.bos, model.eos))[0])


# ---------------------------------------------------------------- reader-only training


def load_from(path: Path, dev):
    ckpt = torch.load(path, map_location=dev)
    configure(ckpt["config"].get("mark_scale", 0.4))
    model = Heptapod(dev)
    model.writer.load_state_dict(ckpt["writer"])
    model.reader.load_state_dict(ckpt["reader"])
    return model, ckpt


def train_reader(args):
    """Freeze the writer of a trained checkpoint and train a reader on its code.

    --fresh starts the reader from random weights (the "new reader learns the
    language" test); otherwise it continues the checkpoint's reader. With
    --aug, half of every batch comes from v3/data/train_aug.json (train-only
    word substitutions), drawn by the same frozen writer.
    Selection is on full dev; test is never touched here.
    """
    seed_all(args.seed)
    dev = device()
    model, ckpt = load_from(Path(args.init), dev)
    for p in model.writer.parameters():
        p.requires_grad_(False)
    if args.fresh:
        model.reader = Reader(model.tokenizer.vocab_size).to(dev)
    reader = model.reader
    train_rows, dev_rows = load_split("train"), load_split("dev")
    aug_rows = []
    if args.aug:
        aug_rows = [{"text": r["sentence"], "category": "aug"} for r in json.loads((DATA / "train_aug.json").read_text())]
    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=0.01)
    warm, total = args.warmup, args.steps

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return args.final_ratio + (1 - args.final_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    OUT.mkdir(exist_ok=True)
    config = {**ckpt["config"], "reader_training": vars(args)}
    log = open(OUT / "train_log.jsonl", "a")
    best, t0 = -1.0, time.time()
    print(f"device={dev} init={args.init} fresh={args.fresh} aug={len(aug_rows)} train={len(train_rows)}", flush=True)
    for step in range(1, args.steps + 1):
        model.writer.eval()
        model.encoder.eval()
        reader.train()
        if aug_rows:
            h = args.batch // 2
            rows = random.sample(train_rows, h) + random.sample(aug_rows, args.batch - h)
        else:
            rows = random.sample(train_rows, args.batch)
        with torch.no_grad():
            img, ids, _ = model.write([r["text"] for r in rows])
        half = args.batch // 2  # half clean images, half through the training channel
        seen = torch.cat([img[:half], channel(img[half:], True)])
        wd = args.word_dropout if step <= args.steps // 4 else 0.0
        loss = target_loss(model, seen, ids, word_dropout=wd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1 or step == args.steps or step % args.eval_every == 0:
            row = {"step": step, "loss": round(loss.item(), 4), "lr": round(sched.get_last_lr()[0], 7),
                   "hard": True, "sec": round(time.time() - t0)}
            if step % args.eval_every == 0 or step == args.steps:
                row["dev_exact"], cats = evaluate(model, dev_rows)
                row["dev_words"] = cats["_word_accuracy"]
                ckpt_out = {"writer": model.writer.state_dict(), "reader": reader.state_dict(),
                            "config": config, "step": step}
                torch.save(ckpt_out, OUT / "latest.pt")
                if row["dev_exact"] > best:
                    best = row["dev_exact"]
                    torch.save(ckpt_out, OUT / "model.pt")
            print(row, flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
    print({"best_full_dev_exact": best}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--soft-until", type=float, default=0.4)
    t.add_argument("--ink-max", type=float, default=0.12)
    t.add_argument("--ink-weight", type=float, default=5.0)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--word-dropout", type=float, default=0.3)
    t.add_argument("--bow-weight", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--mark-scale", type=float, default=0.4)
    t.add_argument("--freeze-writer", action="store_true",
                   help="freeze the writer when the soft phase ends; only the reader keeps training")
    s = sub.add_parser("say")
    s.add_argument("text")
    s.add_argument("--out")
    r = sub.add_parser("read")
    r.add_argument("image")
    f = sub.add_parser("train-reader", help="freeze a checkpoint's writer, train (or retrain) only the reader")
    f.add_argument("--init", required=True, help="path to the checkpoint with the writer to freeze")
    f.add_argument("--fresh", action="store_true", help="start the reader from scratch")
    f.add_argument("--aug", action="store_true", help="half of each batch from train_aug.json")
    f.add_argument("--steps", type=int, default=12000)
    f.add_argument("--batch", type=int, default=64)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--warmup", type=int, default=300)
    f.add_argument("--final-ratio", type=float, default=0.1)
    f.add_argument("--word-dropout", type=float, default=0.1, help="only for the first quarter of steps")
    f.add_argument("--eval-every", type=int, default=4000)
    f.add_argument("--seed", type=int, default=17)
    sub.add_parser("report")
    args = ap.parse_args()
    {"train": train, "say": say, "read": read, "train-reader": train_reader,
     "report": lambda a: report(a)}[args.cmd](args)


if __name__ == "__main__":
    main()
