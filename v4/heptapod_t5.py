#!/usr/bin/env python3
"""Heptapod Ink v4: open English through a ring logogram, learned in stages.

text -> T5-small encoder -> slot writer -> 36 ink marks on one fixed ring
     -> (symbolic) code embedding -> adapter -> T5-small decoder -> text
     -> (visual)   rendered ring -> polar eyes -> slot distributions -> same adapter/decoder

Stage 0  eyes learn to recognise marks on random rings; measure the real
         mark confusion matrix; drop marks the eyes cannot read reliably.
Stage 1  "on paper": writer and decoder invent the code with no image in
         between. Symbols pass through the measured confusion channel, so
         the code has to survive the eyes' real mistakes.
Stage 2  the writer is frozen. Eyes learn to read its rings (slot labels are
         free: we know what we drew), then to produce text through the
         frozen symbolic reader; late on, the decoder adapts gently.

All information is in the marks; the ring line is the same in every slot.

  .venv/bin/python v4/heptapod_t5.py train --smoke          # tiny local run
  .venv/bin/python v4/heptapod_t5.py train                  # full staged run (H100)
  .venv/bin/python v4/heptapod_t5.py eval --split dev       # visual + symbolic metrics
  .venv/bin/python v4/heptapod_t5.py say "I left my keys at home."
  .venv/bin/python v4/heptapod_t5.py read out.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "v2"))
import heptapod_ink as INK  # noqa: E402

OUT = Path(os.environ.get("HEPTAPOD_OUT", HERE / "out"))
DATA = ROOT / "v3" / "data_tatoeba"
T5 = "google-t5/t5-small"
SLOTS, RES, MARK_SCALE, CPD = 36, 288, 0.25, 1.5
ANGLES, RADII, TOKENS = 576, 48, 72
MAX_T = 64
D = 256
SELECT = ("dev", "exact")  # (panel, metric) used to keep the best checkpoint
LEVELS = (3, 3, 3, 3)  # FSQ: 4 numbers per slot, each rounded to -1/0/+1 -> 81 marks


def device():
    if os.environ.get("HEPTAPOD_DEVICE"):
        return torch.device(os.environ["HEPTAPOD_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_all(s):
    random.seed(s), np.random.seed(s), torch.manual_seed(s)


def autocast(dev):
    if dev.type == "cuda" or os.environ.get("HEPTAPOD_BF16"):  # HEPTAPOD_BF16=1 on cpu mimics the H100 path
        return torch.autocast(dev.type, dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


# ---------------------------------------------------------------- data


def load_rows(name):
    return [r["sentence"] for r in json.loads((DATA / f"{name}.json").read_text())]


class Sampler:
    """80% natural Tatoeba train sentences, 20% fresh random word combinations."""

    def __init__(self, random_frac=0.2, seed=0, pool="train"):
        self.train = load_rows(pool)
        vocab = json.loads((DATA / "random_vocab.json").read_text())
        self.words, self.weights = vocab["words"], vocab["weights"]
        self.held = set()
        for split in ("dev", "test", "dev_random", "test_random", "fresh_natural", "fresh_random",
                      "locked_natural", "locked_random", "locked_uniform10"):
            if (DATA / f"{split}.json").exists():
                self.held |= {s.lower().strip() for s in load_rows(split)}
        self.random_frac = random_frac
        self.rng = random.Random(seed)

    def random_sentence(self):
        while True:
            s = " ".join(self.rng.choices(self.words, weights=self.weights, k=self.rng.randint(3, 10)))
            if s not in self.held:
                return s

    def batch(self, n):
        k = sum(self.rng.random() < self.random_frac for _ in range(n))
        return self.rng.sample(self.train, n - k) + [self.random_sentence() for _ in range(k)]


# ---------------------------------------------------------------- the script


def library():
    """Film ink at training resolution: 33 ring segments, 84 marks (0 = blank)."""
    rings = INK.with_blank(INK.ring_rows(torch.tensor(INK.LIB["rings_hi"]).float() / 255)[INK.RING_PICK])
    marks = INK.with_blank(torch.tensor(INK.LIB["marks_hi"]).float() / 255)
    rad = int(round(68 * RES / 192))
    rings = F.interpolate(rings[:, None], size=(rad, int(INK.RING_DEG * CPD)), mode="area")[:, 0]
    marks = F.interpolate(marks[:, None], size=(rad, int(INK.MARK_DEG * MARK_SCALE * CPD)), mode="area")[:, 0]
    return rings, marks


def fixed_ring_index(rings):
    """The plainest real ring segment (least ink): the ring line carries no information."""
    ink = rings[1:].flatten(1).mean(1)
    return int(ink.argmin()) + 1


class Script(nn.Module):
    def __init__(self, res=RES, cpd=CPD, rings=None, marks=None):
        super().__init__()
        INK.SLOTS, INK.PITCH_DEG = SLOTS, 360 / SLOTS
        if rings is None:
            rings, marks = library()
        self.n_marks = len(marks)
        self.ring_id = fixed_ring_index(rings)
        self.render = INK.RingMarkRenderer(rings, marks, cpd, res)
        self.n_rings = len(rings)

    def forward(self, marks_onehot):
        b = marks_onehot.shape[0]
        ring = F.one_hot(torch.full((b, SLOTS), self.ring_id, device=marks_onehot.device), self.n_rings).float()
        with torch.autocast(marks_onehot.device.type, enabled=False):  # ink arithmetic stays fp32
            return self.render(ring, marks_onehot.float())


def channel(img, train):
    """Fixed orientation like the film: slight tremor, scale, shift and ink noise."""
    if not train:
        return img
    b, res = img.shape[0], img.shape[-1]
    ang = (torch.rand(b, device=img.device) * 2 - 1) * math.radians(4)
    scale = 1 + (torch.rand(b, device=img.device) * 2 - 1) * 0.04
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    shift = (torch.rand(b, 2, device=img.device) * 2 - 1) * 0.03
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, (b, 1, res, res), align_corners=False)
    out = F.grid_sample(img[:, None], grid, align_corners=False)[:, 0]
    return (out + 0.1 * torch.randn_like(out)).clamp(0, 1)


# ---------------------------------------------------------------- eyes


class CircConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 3)

    def forward(self, x):
        x = torch.cat([x[..., -1:], x, x[..., :1]], -1)
        return self.conv(F.pad(x, (0, 0, 1, 1)))


class Eyes(nn.Module):
    """Unwrap the ring into a strip (angle across, radius down), read it with
    wrap-around convolutions, then 36 slot queries say which mark is where."""

    def __init__(self, n_marks):
        super().__init__()
        theta = torch.linspace(-math.pi, math.pi, ANGLES + 1)[:-1]
        r = torch.linspace(0.20, 0.86, RADII)
        tt, rr = torch.meshgrid(theta, r, indexing="xy")
        self.register_buffer("grid", torch.stack([rr * torch.cos(tt), rr * torch.sin(tt)], -1)[None])

        def block(i, o, pool):
            return nn.Sequential(CircConv(i, o), nn.BatchNorm2d(o), nn.GELU(),
                                 CircConv(o, o), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d(pool))

        self.cnn = nn.Sequential(block(1, 32, 2), block(32, 64, 2), block(64, 128, 2), block(128, D, (6, 1)))
        self.pos = nn.Parameter(torch.randn(TOKENS, D) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.1, batch_first=True, norm_first=True), 2)
        self.queries = nn.Parameter(torch.randn(SLOTS, D) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(D, 4, 4 * D, 0.1, batch_first=True, norm_first=True), 2)
        self.head = nn.Linear(D, n_marks)

    def forward(self, img):
        strip = F.grid_sample(img[:, None], self.grid.expand(len(img), -1, -1, -1), align_corners=False)
        f = self.cnn(strip).mean(2).transpose(1, 2)          # (B, 72, D)
        h = self.enc(f + self.pos)
        return self.head(self.dec(self.queries.expand(len(img), -1, -1), h))  # (B, 36, n_marks)


# ---------------------------------------------------------------- writer and symbolic reader


class SlotWriter(nn.Module):
    """36 slot queries attend over T5 encoder states; each emits mark logits
    (categorical) or 4 numbers that are rounded to a mark (FSQ)."""

    def __init__(self, n_out):
        super().__init__()
        self.inp = nn.Linear(512, D)
        self.queries = nn.Parameter(torch.randn(SLOTS, D) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(D, 4, 4 * D, 0.1, batch_first=True, norm_first=True), 2)
        self.head = nn.Linear(D, n_out)

    def forward(self, states, mask):
        s = self.dec(self.queries.expand(len(states), -1, -1), self.inp(states), memory_key_padding_mask=~mask)
        return self.head(s)


class Adapter(nn.Module):
    """Slot symbols (one-hot or eyes' probabilities) -> memory for the T5 decoder."""

    def __init__(self, n_marks):
        super().__init__()
        self.n_marks = n_marks
        self.emb = nn.Parameter(torch.randn(n_marks, D) * 0.02)
        self.fsq_in = nn.Linear(len(LEVELS), D)
        self.pos = nn.Parameter(torch.randn(SLOTS, D) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.1, batch_first=True, norm_first=True), 2)
        self.out = nn.Sequential(nn.Linear(D, 512), nn.LayerNorm(512))

    def forward(self, x, table=None):
        """x: (B, 36, n_marks) mark distribution, or (B, 36, 4) FSQ vectors when table is given."""
        if table is None:
            h = x @ self.emb
        else:
            h = self.fsq_in(x @ table if x.shape[-1] == self.n_marks else x)
        return self.out(self.enc(h + self.pos))


class Heptapod(nn.Module):
    def __init__(self, dev, quant="cat"):
        super().__init__()
        self.quant = quant
        self.noise = 1.0
        self.tok = AutoTokenizer.from_pretrained(T5)
        self.script = Script()
        n = self.script.n_marks
        attn = {"attn_implementation": "eager"} if dev.type == "mps" else {}  # MPS SDPA has no dropout
        self.encoder = T5EncoderModel.from_pretrained(T5, **attn)
        self.writer = SlotWriter(D if quant == "vq" else len(LEVELS) if quant in ("fsq", "noise") else n)
        self.adapter = Adapter(n)
        self.t5 = T5ForConditionalGeneration.from_pretrained(T5, **attn)
        for p in self.t5.encoder.parameters():  # unused: memory comes from the adapter (kept for generate())
            p.requires_grad_(False)
        self.eyes = Eyes(n)
        self.register_buffer("allowed", torch.ones(n, dtype=torch.bool))
        self.register_buffer("confusion", torch.eye(n))
        self.register_buffer("fsq_table", torch.zeros(n, len(LEVELS)))            # mark id -> FSQ vector
        self.register_buffer("fsq_to_mark", torch.zeros(math.prod(LEVELS), dtype=torch.long))  # symbol -> mark
        K = math.prod(LEVELS)
        self.register_buffer("books", torch.zeros(SLOTS, K, D))      # VQ: one 81-entry codebook per slot
        self.register_buffer("ema_n", torch.ones(SLOTS, K))
        self.register_buffer("ema_w", torch.zeros(SLOTS, K, D))
        self.register_buffer("last_used", torch.zeros(SLOTS, K))
        self.sentinel = self.tok.convert_tokens_to_ids("<extra_id_0>")
        self.to(dev)
        self.dev = dev

    def mask(self, logits):
        return logits.masked_fill(~self.allowed, -1e4)

    def tokenize(self, texts):
        b = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_T)
        return {k: v.to(self.dev) for k, v in b.items()}

    @property
    def table(self):
        return self.fsq_table if self.quant in ("fsq", "noise") else None

    def set_fsq_marks(self, mark_ids):
        """FSQ index k (digits in base 3) -> one readable mark."""
        assert len(mark_ids) == len(self.fsq_to_mark)
        self.fsq_to_mark.copy_(torch.as_tensor(mark_ids, device=self.dev))
        self.fsq_table.zero_()
        for k, m in enumerate(mark_ids):
            digits = [(k // 3 ** i) % 3 - 1 for i in range(len(LEVELS))]
            self.fsq_table[m] = torch.tensor(digits, dtype=torch.float, device=self.dev)

    def writer_out(self, texts):
        b = self.tokenize(texts)
        states = self.encoder(**b).last_hidden_state
        return self.writer(states, b["attention_mask"].bool()).float()

    def code_logits(self, texts):
        return self.mask(self.writer_out(texts))

    def encode(self, texts, mode="argmax", tau=1.0):
        """-> (z for the adapter with straight-through gradients, mark ids)."""
        raw = self.writer_out(texts)
        if self.quant == "vq":
            if mode == "cont":
                return raw, torch.zeros(raw.shape[:2], dtype=torch.long, device=raw.device)
            idx = self.assign(raw.detach())
            q = self.books[torch.arange(SLOTS, device=raw.device)[None], idx]
            self._vq = (raw, idx, q)  # for the commitment loss and EMA update
            return raw + (q - raw).detach(), self.fsq_to_mark[idx]
        if self.quant in ("fsq", "noise"):
            h = torch.tanh(raw)
            q = torch.round(h)
            if self.quant == "noise" and self.training:
                # train with rounding-sized noise (no rounding); at test time the code is rounded
                z = h + (torch.rand_like(h) - 0.5) * self.noise
            else:
                z = h + (q - h).detach()
            if mode == "continuous":
                z = h
            k = ((q + 1).long() * torch.tensor([3 ** i for i in range(len(LEVELS))], device=q.device)).sum(-1)
            return z, self.fsq_to_mark[k]
        logits = self.mask(raw)
        return quantize(logits, mode, tau), logits.argmax(-1)

    @torch.no_grad()
    def code(self, texts):
        return self.encode(texts, "vq" if self.quant == "vq" else "argmax")[1]   # (B, 36) mark ids

    def assign(self, z):
        return torch.cdist(z.float().transpose(0, 1), self.books).argmin(-1).transpose(0, 1)  # (B, S)

    def memory(self, x):
        """x: (B,S,n_marks) mark distribution (one-hot or eyes) or (B,S,D) writer vectors (VQ)."""
        if self.quant == "vq":
            if x.shape[-1] == self.script.n_marks:
                x = torch.einsum("bsk,skd->bsd", x[..., self.fsq_to_mark].float(), self.books)
            a = self.adapter
            return a.out(a.enc(x + a.pos))
        return self.adapter(x, self.table)

    @torch.no_grad()
    def vq_update(self, decay=0.99, reseed_after=200):
        z, idx, q = self._vq
        z = z.float()
        oh = F.one_hot(idx, self.books.shape[1]).float()
        self.ema_n.mul_(decay).add_((1 - decay) * oh.sum(0))
        self.ema_w.mul_(decay).add_((1 - decay) * torch.einsum("bsk,bsd->skd", oh, z))
        self.books.copy_(self.ema_w / self.ema_n.clamp_min(1e-5)[..., None])
        self.last_used.add_(1)
        self.last_used[oh.sum(0) > 0] = 0
        dead = (self.last_used > reseed_after).nonzero().tolist()
        if dead:
            err = (z - q.float()).pow(2).sum(-1)
            for s_, k in dead:
                v = z[err[:, s_].argmax(), s_]
                self.books[s_, k] = v
                self.ema_w[s_, k] = v
                self.ema_n[s_, k] = 1.0
                self.last_used[s_, k] = 0
        return len(dead)

    @torch.no_grad()
    def vq_init(self, sampler, n=2048, iters=15):
        """k-means per slot on the continuous writer's vectors."""
        was = self.training
        self.eval()
        Z = torch.cat([self.writer_out(sampler.batch(128)) for _ in range(n // 128)]).float()
        K = self.books.shape[1]
        for s_ in range(SLOTS):
            x = Z[:, s_]
            c = x[torch.randperm(len(x), device=x.device)[:K]].clone()
            for _ in range(iters):
                a = torch.cdist(x, c).argmin(-1)
                cnt = torch.bincount(a, minlength=K).float()
                summ = torch.zeros_like(c).index_add_(0, a, x)
                c = torch.where(cnt[:, None] > 0, summ / cnt.clamp_min(1)[:, None], x[torch.randint(0, len(x), (K,), device=x.device)])
            self.books[s_] = c
        self.ema_w.copy_(self.books)
        self.ema_n.fill_(1.0)
        self.last_used.zero_()
        self.train(was)

    def draw(self, code):
        return self.script(F.one_hot(code, self.script.n_marks))

    def eye_probs(self, img):
        return F.softmax(self.mask(self.eyes(img).float()), -1)

    def labels(self, texts):
        ids = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_T).input_ids.to(self.dev)
        return ids.masked_fill(ids == self.tok.pad_token_id, -100)

    def text_loss(self, probs, labels, prefix_dropout=0.0):
        mem = self.memory(probs)
        dec_in = self.t5._shift_right(labels)
        if prefix_dropout > 0:
            hide = (torch.rand(dec_in.shape, device=dec_in.device) < prefix_dropout) & (dec_in != self.tok.pad_token_id)
            hide[:, 0] = False
            dec_in = dec_in.masked_fill(hide, self.sentinel)
        out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem), decoder_input_ids=dec_in, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(self, probs, beams=1):
        mem = self.memory(probs)
        out = self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                               attention_mask=torch.ones(mem.shape[:2], dtype=torch.long, device=mem.device),
                               max_new_tokens=MAX_T, num_beams=beams, num_return_sequences=beams, do_sample=False)
        return self.tok.batch_decode(out, skip_special_tokens=True)


# ---------------------------------------------------------------- metrics


def norm(s):
    return " ".join(s.lower().split())


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def score(gold, pred):
    n = len(gold)
    words_ok = words = errs = 0
    for g, p in zip(gold, pred):
        gw, pw = norm(g).split(), norm(p).split()
        words_ok += sum(a == b for a, b in zip(gw, pw))
        words += len(gw)
        errs += edit(gw, pw)
    return {"exact": round(sum(norm(g) == norm(p) for g, p in zip(gold, pred)) / n, 4),
            "exact_cased": round(sum(g.strip() == p.strip() for g, p in zip(gold, pred)) / n, 4),
            "word_acc": round(words_ok / words, 4), "wer": round(errs / words, 4)}


@torch.no_grad()
def evaluate(model, texts, mode, batch=256, beams=1, shuffle=False):
    """mode: 'symbolic' (clean code, no image) or 'visual' (rendered ring -> eyes)."""
    was = model.training
    model.eval()
    preds, slot_ok, slots = [], 0, 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        with autocast(model.dev):
            code = model.code(chunk)
            if mode == "symbolic":
                probs = F.one_hot(code, model.script.n_marks).float()
            elif mode == "continuous":
                probs = model.encode(chunk, "cont" if model.quant == "vq" else "continuous")[0]
            else:
                img = model.draw(code)
                if shuffle:
                    img = img[torch.randperm(len(img), device=img.device)]
                probs = model.eye_probs(img)
                slot_ok += (probs.argmax(-1) == code).sum().item()
                slots += code.numel()
            preds += model.generate(probs, beams=1)
    model.train(was)
    out = score(texts, preds)
    if slots:
        out["slot_acc"] = round(slot_ok / slots, 4)
    return out, preds


# ---------------------------------------------------------------- training


def quantize(logits, mode, tau):
    p = F.softmax(logits / tau, -1)
    if mode == "soft":
        return p
    if mode == "gumbel":
        u = torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)
        p = F.softmax((logits - torch.log(-torch.log(u))) / tau, -1)
    return F.one_hot(p.argmax(-1), p.shape[-1]).float() + p - p.detach()


def slot_distances(model, raw):
    return (raw.float()[:, :, None, :] - model.books.detach()[None]).pow(2).mean(-1)  # (B, S, K)


def entropy(p, dim=-1):
    return -(p * (p + 1e-9).log()).sum(dim)


@torch.no_grad()
def calibrate_tau(model, sampler, target=math.log(4), n=512):
    """Per-slot temperature so the median soft-assignment entropy is ln(4)."""
    was = model.training
    model.eval()
    d = torch.cat([slot_distances(model, model.writer_out(sampler.batch(128))) for _ in range(n // 128)])
    model.train(was)
    tau = torch.ones(SLOTS, device=d.device)
    for s_ in range(SLOTS):
        lo, hi = 1e-6, 1e3
        for _ in range(50):
            mid = math.sqrt(lo * hi)
            h = entropy(F.softmax(-d[:, s_] / mid, -1)).median()
            lo, hi = (mid, hi) if h < target else (lo, mid)
        tau[s_] = math.sqrt(lo * hi)
    return tau


def usage_entropy_loss(model, raw, tau_s):
    """Per slot: sharp individual assignments, spread-out batch usage (MAGVIT-v2 style)."""
    p = F.softmax(-slot_distances(model, raw) / tau_s[None, :, None], -1)    # (B, S, K)
    return (entropy(p).mean(0) - entropy(p.mean(0))).mean()


def corrupt(model, z, idx, alpha):
    """Send a fraction of symbols through the measured eyes' confusion channel
    (identity straight-through gradient)."""
    if alpha <= 0:
        return z
    new = torch.multinomial(model.confusion[idx.flatten()], 1).view_as(idx)
    new = torch.where(torch.rand_like(idx, dtype=torch.float) < alpha, new, idx)
    if model.quant == "vq":
        mark_to_sym = torch.full((model.script.n_marks,), -1, dtype=torch.long, device=z.device)
        mark_to_sym[model.fsq_to_mark] = torch.arange(len(model.fsq_to_mark), device=z.device)
        sym = mark_to_sym[new]
        target = model.books[torch.arange(SLOTS, device=z.device)[None], sym]
        return target + z - z.detach()
    target = model.fsq_table[new] if model.quant in ("fsq", "noise") else F.one_hot(new, z.shape[-1]).float()
    return target + z - z.detach()


def log(row, f):
    print(row, flush=True)
    f.write(json.dumps(row) + "\n")
    f.flush()


def save(model, path, stage, extra=None):
    keep = {k: v for k, v in model.state_dict().items() if not k.startswith("script.")}
    torch.save({"model": keep, "stage": stage, "quant": model.quant, **(extra or {})}, path)


def stage0(model, args, f):
    """Eyes on random rings; then the real confusion matrix and the readable marks."""
    n = model.script.n_marks
    opt = torch.optim.AdamW(model.eyes.parameters(), lr=1e-3, weight_decay=0.01)
    t0, step = time.time(), 0
    while step < args.s0_steps and time.time() - t0 < args.s0_sec:
        step += 1
        code = torch.randint(0, n, (args.batch, SLOTS), device=model.dev)
        with torch.no_grad():
            img = channel(model.draw(code), True)
        with autocast(model.dev):
            logits = model.eyes(img).float()
        loss = F.cross_entropy(logits.reshape(-1, n), code.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            acc = (logits.argmax(-1) == code).float().mean().item()
            log({"stage": 0, "step": step, "loss": round(loss.item(), 4), "slot_acc": round(acc, 4),
                 "sec": round(time.time() - t0)}, f)
    model.eyes.eval()
    counts = torch.full((n, n), 0.5, device=model.dev)
    with torch.no_grad():
        for _ in range(args.s0_confusion_batches):
            code = torch.randint(0, n, (args.batch, SLOTS), device=model.dev)
            with autocast(model.dev):
                pred = model.eyes(channel(model.draw(code), True)).float().argmax(-1)
            counts += torch.bincount((code * n + pred).flatten(), minlength=n * n).view(n, n).float()
    model.eyes.train()
    conf = counts / counts.sum(1, keepdim=True)
    diag = conf.diag()
    if model.quant in ("fsq", "noise", "vq"):  # the 81 most readable marks (blank always in)
        order = [0] + [int(i) for i in diag[1:].argsort(descending=True) + 1]
        keep = order[: len(model.fsq_to_mark)]
        allowed = torch.zeros(n, dtype=torch.bool, device=model.dev)
        allowed[keep] = True
        model.set_fsq_marks(keep)
    else:
        allowed = diag >= args.min_mark_acc
        allowed[0] = True
    model.allowed.copy_(allowed)
    sub = conf[allowed][:, allowed]
    model.confusion.copy_(torch.zeros_like(conf))
    idx = allowed.nonzero().flatten()
    model.confusion[idx[:, None], idx[None, :]] = sub / sub.sum(1, keepdim=True)
    info = {"stage": 0, "event": "confusion", "mean_mark_acc": round(diag.mean().item(), 4),
            "allowed_marks": int(allowed.sum()), "of": n,
            "mean_acc_allowed": round(diag[allowed].mean().item(), 4),
            "bits_per_slot_nominal": round(math.log2(int(allowed.sum())), 2)}
    log(info, f)
    save(model, OUT / "stage0.pt", 0, {"info": info})


def stage1(model, args, sampler, dev_sets, f):
    """Invent the code on paper: text -> marks -> (confusion channel) -> text."""
    new = list(model.writer.parameters()) + list(model.adapter.parameters())
    pre = list(model.encoder.parameters()) + [p for n_, p in model.t5.named_parameters() if not n_.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": new, "lr": args.lr_new}, {"params": pre, "lr": 0.0}], weight_decay=0.01)
    t0, step, best = time.time(), 0, -1.0
    sampler.random_frac = args.s1_random
    vq_step = 0
    tau_s = calibrate_tau(model, sampler) if (model.quant == "vq" and args.entropy > 0 and args.resume_stage1) else None
    while step < args.s1_steps and time.time() - t0 < args.s1_sec:
        step += 1
        frac = min(1.0, (time.time() - t0) / args.s1_sec)
        tau = 1.0 - 0.5 * frac
        mode = "soft" if step <= args.soft_steps else "gumbel" if step <= args.soft_steps + args.gumbel_steps else "argmax"
        alpha = args.alpha
        if model.quant in ("fsq", "noise"):
            mode = "fsq"
            model.noise = args.noise
        if model.quant == "vq":
            mode = "cont" if step <= args.warm_steps else "vq"
            if step == args.warm_steps + 1 and not args.resume_stage1:
                model.vq_init(sampler)
                if args.entropy > 0:
                    tau_s = calibrate_tau(model, sampler)
                log({"stage": 1, "step": step, "event": "vq books initialised (k-means)"}, f)
            vq_step = max(0, step - args.warm_steps)
            alpha = args.alpha * min(1.0, max(0.0, (vq_step - args.alpha_after) / 500))
            if args.resume_stage1:
                alpha = args.alpha
        opt.param_groups[1]["lr"] = 0.0 if (step <= args.pre_freeze and not args.resume_stage1) else args.lr_pre
        texts = sampler.batch(args.batch)
        with autocast(model.dev):
            z, ids = model.encode(texts, mode, tau)
            if mode not in ("soft", "cont"):
                z = corrupt(model, z, ids, alpha)
            loss = model.text_loss(z, model.labels(texts), args.prefix_dropout)
            total = loss
            if mode == "vq":
                raw, _, q = model._vq
                total = loss + args.commit * F.mse_loss(raw.float(), q.detach())
                if tau_s is not None:
                    lam = args.entropy * min(1.0, vq_step / 200)
                    ent = usage_entropy_loss(model, raw, tau_s)
                    total = total + lam * ent
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(new + pre, 1.0)
        opt.step()
        reseeded = model.vq_update() if mode == "vq" else 0
        if step % 100 == 0 or step == 1 or step % args.eval_every == 0:
            row = {"stage": 1, "step": step, "loss": round(loss.item(), 4), "mode": mode, "alpha": round(alpha, 3),
                   "sec": round(time.time() - t0)}
            if mode == "vq":
                row["reseeded"] = reseeded
            if step % args.eval_every == 0:
                for name, texts_ in dev_sets.items():
                    if mode != "cont":
                        m, _ = evaluate(model, texts_[:args.eval_n], "symbolic")
                        row[f"sym_{name}"] = m
                    if model.quant == "noise" or mode == "cont":
                        m, _ = evaluate(model, texts_[:args.eval_n], "continuous")
                        row[f"cont_{name}"] = m
                if mode != "cont":
                    model.eval()
                    c = model.code(next(iter(dev_sets.values()))[:256])
                    model.train()
                    row["marks_used"] = int((c.flatten().bincount(minlength=model.script.n_marks) > 0).sum())
                    row["min_marks_per_slot"] = min(len(set(c[:, s_].tolist())) for s_ in range(SLOTS))
                sel = row[f"sym_{SELECT[0]}"][SELECT[1]]
                row["select"] = sel
                if mode in ("argmax", "fsq", "vq") and sel > best:
                    best = sel
                    save(model, OUT / "stage1.pt", 1)
            log(row, f)
    if best < 0:
        save(model, OUT / "stage1.pt", 1)
    return best


def stage2(model, args, sampler, dev_sets, f):
    """Frozen writer; eyes learn to read it, first slots, then text through the symbolic reader."""
    for m in (model.encoder, model.writer, model.adapter, model.t5):
        for p in m.parameters():
            p.requires_grad_(False)
    dec = [p for n_, p in model.t5.named_parameters() if n_.startswith("decoder.")]
    opt = torch.optim.AdamW([{"params": list(model.eyes.parameters()), "lr": args.lr_eyes},
                             {"params": dec, "lr": 0.0}], weight_decay=0.01)
    t0, step, best = time.time(), 0, -1.0
    sampler.random_frac = args.s2_random
    n = model.script.n_marks
    while step < args.s2_steps and time.time() - t0 < args.s2_sec:
        step += 1
        adapt = step > args.decoder_after
        if adapt and not dec[0].requires_grad:
            for p in dec:
                p.requires_grad_(True)
            log({"stage": 2, "step": step, "event": "decoder adapting"}, f)
        opt.param_groups[1]["lr"] = args.lr_dec if adapt else 0.0
        model.encoder.eval(), model.writer.eval(), model.adapter.eval()
        model.t5.train(adapt)
        texts = sampler.batch(args.batch)
        with torch.no_grad(), autocast(model.dev):
            code = model.code(texts)
            img = model.draw(code)
        seen = torch.cat([img[: len(img) // 4], channel(img[len(img) // 4:], True)])
        with autocast(model.dev):
            logits = model.mask(model.eyes(seen).float())
            slot = F.cross_entropy(logits.reshape(-1, n), code.reshape(-1))
            loss = slot
            if step > args.text_after:
                labels = model.labels(texts)
                text = model.text_loss(F.softmax(logits, -1), labels)
                loss = loss + args.text_weight * text
                if adapt:  # clean-symbol replay keeps the decoder anchored to the code
                    k = max(1, len(texts) // 5)
                    clean = F.one_hot(code[:k], n).float()
                    loss = loss + model.text_loss(clean, labels[:k])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.eyes.parameters()) + dec, 1.0)
        opt.step()
        if step % 100 == 0 or step == 1 or step % args.eval_every == 0:
            row = {"stage": 2, "step": step, "loss": round(loss.item(), 4), "slot_loss": round(slot.item(), 4),
                   "slot_acc": round((logits.argmax(-1) == code).float().mean().item(), 4),
                   "sec": round(time.time() - t0)}
            if step % args.eval_every == 0:
                for name, texts_ in dev_sets.items():
                    m, _ = evaluate(model, texts_[:args.eval_n], "visual")
                    row[f"vis_{name}"] = m
                sel = row[f"vis_{SELECT[0]}"][SELECT[1]]
                row["select"] = sel
                if sel > best:
                    best = sel
                    save(model, OUT / "model.pt", 2)
            log(row, f)
    save(model, OUT / "latest.pt", 2)
    if best < 0:
        save(model, OUT / "model.pt", 2)
    return best


def load(path, dev):
    ckpt = torch.load(path, map_location=dev)
    model = Heptapod(dev, ckpt.get("quant", "cat"))
    model.load_state_dict(ckpt["model"], strict=False)
    return model, ckpt


def train(args):
    seed_all(args.seed)
    dev = device()
    OUT.mkdir(parents=True, exist_ok=True)
    f = open(OUT / "train_log.jsonl", "a")
    global SELECT
    sampler = Sampler(seed=args.seed, pool=args.pool)
    names = args.dev_sets.split()
    # panels: fixed random samples (seed 0, as in search.py eval) so numbers are comparable across tools
    dev_sets = {n: random.Random(0).sample(load_rows(n), min(args.eval_n, len(load_rows(n)))) for n in names}
    SELECT = (names[0], args.select)
    if args.resume:
        model, ck = load(OUT / args.resume, dev)
        start = ck["stage"] + 1
    else:
        model, start = Heptapod(dev, args.quant), 0
    model.train()
    log({"event": "start", "device": str(dev), "train": len(sampler.train), "stage": start,
         "args": vars(args), "fixed_ring": model.script.ring_id}, f)
    if start <= 0:
        stage0(model, args, f)
    if args.resume_stage1:
        args.warm_steps = 0  # books already exist: VQ from the first update
        best1 = stage1(model, args, sampler, dev_sets, f)
        log({"stage": 1, "event": "done (continued)", "best_symbolic_dev_exact": best1}, f)
        model, _ = load(OUT / "stage1.pt", dev)
        model.train()
    elif start <= 1:
        best1 = stage1(model, args, sampler, dev_sets, f)
        log({"stage": 1, "event": "done", "best_symbolic_dev_exact": best1}, f)
        model, _ = load(OUT / "stage1.pt", dev)
        model.train()
    if args.resume_stage1:
        start = 1
    if start <= 2 and not args.stop_after_stage1:
        best2 = stage2(model, args, sampler, dev_sets, f)
        log({"stage": 2, "event": "done", "best_visual_dev_exact_subset": best2}, f)
    if args.final_eval and not args.stop_after_stage1:
        model, _ = load(OUT / "model.pt", dev)
        report(model, "dev", f)


def report(model, split, f=None, beams=1):
    out = {}
    for name in (split, f"{split}_random"):
        texts = load_rows(name)
        vis, preds = evaluate(model, texts, "visual")
        sym, _ = evaluate(model, texts, "symbolic")
        ctrl, _ = evaluate(model, texts[:512], "visual", shuffle=True)
        out[name] = {"visual": vis, "symbolic": sym, "shuffled_image_control": ctrl,
                     "examples": [[g, p] for g, p in list(zip(texts, preds))[:12]]}
    path = OUT / f"metrics_{split}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if f:
        log({"event": "report", "split": split, **{k: v["visual"] for k, v in out.items()}}, f)
    return out


def cmd_eval(args):
    model, _ = load(OUT / args.ckpt, device())
    report(model, args.split)


# ---------------------------------------------------------------- say / read


def hi_script(res=1024):
    rings = INK.with_blank(INK.ring_rows(torch.tensor(INK.LIB["rings_hi"]).float() / 255)[INK.RING_PICK])
    marks = INK.with_blank(torch.tensor(INK.LIB["marks_hi"]).float() / 255)
    cpd = float(INK.LIB["hi_cpd"])
    marks = F.interpolate(marks[:, None], size=(marks.shape[1], int(INK.MARK_DEG * MARK_SCALE * cpd)), mode="area")[:, 0]
    s = Script(res, cpd, rings, marks)
    s.ring_id = Script().ring_id  # same ring segment as in training
    return s


def export_image(model, code, hi=None, size=4 * RES):
    """Ring as a printable image in [0,1] ink. Default: the exact training-resolution
    drawing, upsampled; `hi` uses the separate high-resolution renderer (prettier,
    but its pixels differ slightly from what the eyes were trained on)."""
    if hi is not None:
        img = hi(F.one_hot(code.to(hi.render.rings.device), model.script.n_marks))
        img = ((img - 0.35) / 0.3).clamp(0, 1)
        return (img * img * (3 - 2 * img))[0].cpu().numpy()
    img = model.draw(code)[None]
    return F.interpolate(img, size=(size, size), mode="nearest")[0, 0].cpu().numpy()  # 4x pixel blocks: exact on read-back


def save_png(path, img):
    """Ink in [0,1] -> 16-bit grayscale PNG (white paper, dark ink)."""
    from PIL import Image
    Image.fromarray(np.round((1 - img) * 65535).astype(np.uint16)).save(path)


def load_png(path):
    """Any PNG -> ink tensor at training resolution (area average)."""
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        g = np.asarray(im).astype(np.float64) / 65535.0
    else:
        g = np.asarray(im.convert("L")).astype(np.float64) / 255.0
    t = torch.tensor(1 - g, dtype=torch.float32)[None, None]
    if t.shape[-1] % RES == 0 and t.shape[-2] % RES == 0:
        t = F.avg_pool2d(t, t.shape[-1] // RES)
    else:
        t = F.interpolate(t, size=(RES, RES), mode="area")
    return t[0, 0]


@torch.no_grad()
def say(args):
    import matplotlib.pyplot as plt
    model, _ = load(OUT / args.ckpt, device())
    model.eval()
    img = export_image(model, model.code([args.text]))
    name = "_".join("".join(ch for ch in w if ch.isalnum()) for w in norm(args.text).split())[:80]
    path = Path(args.out) if args.out else OUT / f"say_{name}.png"
    save_png(path, img)
    print(path)


@torch.no_grad()
def read(args):
    from PIL import Image
    model, _ = load(OUT / args.ckpt, device())
    model.eval()
    img = load_png(args.image).to(model.dev)[None]
    print(model.generate(model.eye_probs(img))[0])


def main():
    import faulthandler
    faulthandler.dump_traceback_later(300, repeat=True)  # if a run stalls, its logs show where
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--quant", default="vq", choices=["vq", "fsq", "cat", "noise"], help="how a slot becomes a mark")
    t.add_argument("--noise", type=float, default=1.0, help="noise width for --quant noise (0 = continuous ceiling)")
    t.add_argument("--warm-steps", type=int, default=4000, help="VQ: continuous steps before the codebooks")
    t.add_argument("--alpha-after", type=int, default=3000, help="VQ: start ramping the confusion channel after this many VQ steps")
    t.add_argument("--commit", type=float, default=0.25)
    t.add_argument("--s1-random", type=float, default=0.5)
    t.add_argument("--s2-random", type=float, default=0.3)
    t.add_argument("--entropy", type=float, default=0.0, help="VQ per-slot batch-entropy weight (0 = off)")
    t.add_argument("--resume-stage1", action="store_true", help="continue stage 1 (VQ) from --resume, keeping its books")
    t.add_argument("--stop-after-stage1", action="store_true")
    t.add_argument("--pool", default="train", help="training sentences: train (Tatoeba) or mixed_train")
    t.add_argument("--dev-sets", default="dev dev_random", help="dev panels; the first one selects checkpoints")
    t.add_argument("--select", default="exact", choices=["exact", "exact_cased", "word_acc"])
    t.add_argument("--resume", help="checkpoint file in OUT to continue from (stage0.pt or stage1.pt)")
    t.add_argument("--s0-steps", type=int, default=4000)
    t.add_argument("--s0-sec", type=float, default=600)
    t.add_argument("--s0-confusion-batches", type=int, default=40)
    t.add_argument("--min-mark-acc", type=float, default=0.5)
    t.add_argument("--s1-steps", type=int, default=12000)
    t.add_argument("--s1-sec", type=float, default=4200)
    t.add_argument("--soft-steps", type=int, default=500)
    t.add_argument("--gumbel-steps", type=int, default=1500)
    t.add_argument("--pre-freeze", type=int, default=500)
    t.add_argument("--lr-new", type=float, default=3e-4)
    t.add_argument("--lr-pre", type=float, default=2e-5)
    t.add_argument("--alpha", type=float, default=0.5, help="fraction of symbols sent through the confusion channel")
    t.add_argument("--prefix-dropout", type=float, default=0.15)
    t.add_argument("--s2-steps", type=int, default=15000)
    t.add_argument("--s2-sec", type=float, default=5100)
    t.add_argument("--lr-eyes", type=float, default=5e-4)
    t.add_argument("--lr-dec", type=float, default=1e-5)
    t.add_argument("--text-after", type=int, default=1000)
    t.add_argument("--decoder-after", type=int, default=2000)
    t.add_argument("--text-weight", type=float, default=1.0)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--eval-n", type=int, default=512)
    t.add_argument("--final-eval", action="store_true")
    t.add_argument("--smoke", action="store_true", help="tiny local run that exercises every stage")
    e = sub.add_parser("eval")
    e.add_argument("--split", default="dev", choices=["dev", "test"])
    e.add_argument("--ckpt", default="model.pt")
    s = sub.add_parser("say")
    s.add_argument("text")
    s.add_argument("--out")
    s.add_argument("--ckpt", default="model.pt")
    r = sub.add_parser("read")
    r.add_argument("image")
    r.add_argument("--ckpt", default="model.pt")
    args = ap.parse_args()
    if args.cmd == "train" and args.smoke:
        args.batch, args.s0_steps, args.s0_confusion_batches = 8, 5, 2
        args.s1_steps, args.soft_steps, args.gumbel_steps, args.pre_freeze = 12, 3, 3, 4
        args.warm_steps, args.alpha_after = 4, 2
        args.s2_steps, args.text_after, args.decoder_after = 8, 2, 4
        args.eval_every, args.eval_n, args.min_mark_acc = 4, 8, 0.0
    {"train": train, "eval": cmd_eval, "say": say, "read": read}[args.cmd](args)


if __name__ == "__main__":
    main()
