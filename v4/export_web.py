#!/usr/bin/env python3
"""Export the Heptapod v4 system to ONNX for in-browser inference (onnxruntime-web).

Five graphs (batch 1):
  writer.onnx   input_ids, attention_mask (1,T)      -> marks (1,36) int64      [T5 encoder + slot writer + codebooks]
  render.onnx   marks (1,36) int64                   -> ink (1,288,288) float   [real Arrival ink, fixed ring]
  eyes.onnx     ink (1,288,288)                       -> probs (1,36,84)        [mark recognition]
  memory.onnx   probs (1,36,84)                       -> memory (1,36,512)      [adapter]
  decoder.onnx  decoder_input_ids (1,L), memory       -> next_logits (1,32128)  [T5 decoder, no cache]
writer and decoder are also saved int8-quantised (*_q8.onnx). Then a parity check against PyTorch.

  .venv/bin/python v4/export_web.py --out web/models
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402


class Writer(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        states = self.m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        raw = self.m.writer(states, attention_mask.bool())
        d = ((raw[:, :, None, :] - self.m.books[None]) ** 2).sum(-1)
        return self.m.fsq_to_mark[d.argmin(-1)]


class Render(nn.Module):
    """Same ink composition as RingMarkRenderer, written with gathers + cumulative products (export-friendly)."""

    def __init__(self, m):
        super().__init__()
        r = m.script.render
        self.r = r
        ring = r.rings.view(r.rings.shape[0], r.rad, r.rw)[m.script.ring_id] * r.ring_window      # fixed ring segment
        self.register_buffer("ring", ring.clamp(0, 1))
        self.register_buffer("marks_lib", r.marks.view(-1, r.rad, r.mw).clamp(0, 1))

    def forward(self, marks):
        r = self.r
        keep = torch.ones(1, r.rad, r.cols)
        for k in range(H.SLOTS):
            keep[..., r.ring_idx[k]] = keep[..., r.ring_idx[k]] * (1 - self.ring)
            mk = self.marks_lib[marks[0, k]]
            keep[..., r.mark_idx[k]] = keep[..., r.mark_idx[k]] * (1 - mk)
        return r.to_image(1 - keep)


class Eyes(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ink):
        return self.m.eye_probs(ink)


class Memory(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, probs):
        return self.m.memory(probs)


class Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, decoder_input_ids, memory):
        out = self.m.t5(encoder_outputs=BaseModelOutput(last_hidden_state=memory), decoder_input_ids=decoder_input_ids, use_cache=False)
        return out.logits[:, -1]


def export(mod, args, path, names_in, names_out, dyn):
    torch.onnx.export(mod, args, str(path), input_names=names_in, output_names=names_out, dynamic_axes=dyn,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print("exported", path.name, f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4/runs/s2/model.pt")
    ap.add_argument("--out", default="web/models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)   # nn.Transformer fast path is not exportable
    m, _ = H.load(Path(args.ckpt), torch.device("cpu"))
    m.eval()
    for mod in (m.encoder, m.t5):
        mod.config._attn_implementation = "eager"
    tok = m.tok("Hello, earthling.", return_tensors="pt", padding="max_length", max_length=H.MAX_T)  # fixed T=64
    ids, am = tok.input_ids, tok.attention_mask
    export(Writer(m).eval(), (ids, am), out / "writer.onnx", ["input_ids", "attention_mask"], ["marks"], {})
    m.eval()  # export restores the wrapper's (training) mode on children: force eval again
    marks = Writer(m).eval()(ids, am)
    rend = Render(m).eval()
    assert torch.allclose(rend(marks), m.script(F.one_hot(marks, m.script.n_marks).float()), atol=1e-5), "render mismatch"
    # the legacy exporter produces an invalid Add for the in-place ink composition; the dynamo exporter is fine
    torch.onnx.export(rend, (marks,), str(out / "render.onnx"), input_names=["marks"], output_names=["ink"],
                      opset_version=18, dynamo=True)
    print("exported render.onnx", f"{(out / 'render.onnx').stat().st_size / 1e6:.1f} MB", flush=True)
    ink = rend(marks)
    m.eval()
    export(Eyes(m).eval(), (ink,), out / "eyes.onnx", ["ink"], ["probs"], {})
    m.eval()
    probs = Eyes(m).eval()(ink)
    export(Memory(m).eval(), (probs,), out / "memory.onnx", ["probs"], ["memory"], {})
    m.eval()
    mem = Memory(m).eval()(probs)
    dec_ids = torch.tensor([[m.t5.config.decoder_start_token_id]])
    export(Decoder(m).eval(), (dec_ids, mem), out / "decoder.onnx", ["decoder_input_ids", "memory"], ["next_logits"],
           {"decoder_input_ids": {1: "L"}})
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for name in ("writer", "decoder"):
        quantize_dynamic(str(out / f"{name}.onnx"), str(out / f"{name}_q8.onnx"), weight_type=QuantType.QInt8)
        print("quantised", name, f"{(out / f'{name}_q8.onnx').stat().st_size / 1e6:.1f} MB", flush=True)
    cfg = {"slots": H.SLOTS, "res": H.RES, "max_tokens": H.MAX_T, "n_marks": int(m.script.n_marks),
           "decoder_start_token_id": int(m.t5.config.decoder_start_token_id), "eos_token_id": int(m.tok.eos_token_id),
           "pad_token_id": int(m.tok.pad_token_id), "tokenizer": "Xenova/t5-small"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    m.eval()
    parity(m, out, cfg)


def parity(m, out, cfg):
    import onnxruntime as ort
    S = {n: ort.InferenceSession(str(out / f"{n}.onnx")) for n in ("writer_q8", "render", "eyes", "memory", "decoder_q8", "writer", "decoder")}
    texts = ["Hello, earthling.", "The ink spread slowly across the glass.", "Tom didn't go to school yesterday?",
             "My grandmother still writes letters by hand.", "The teachers are outside.", "Can you send me the photos from Saturday?"]
    rows = []
    for t in texts:
        enc = m.tok(t, return_tensors="np", padding="max_length", max_length=cfg["max_tokens"])
        for tag, wname, dname in (("fp32", "writer", "decoder"), ("int8", "writer_q8", "decoder_q8")):
            marks = S[wname].run(None, {"input_ids": enc.input_ids.astype(np.int64), "attention_mask": enc.attention_mask.astype(np.int64)})[0]
            ink = S["render"].run(None, {"marks": marks})[0]
            probs = S["eyes"].run(None, {"ink": ink})[0]
            mem = S["memory"].run(None, {"probs": probs})[0]
            seq = [cfg["decoder_start_token_id"]]
            for _ in range(cfg["max_tokens"]):
                logits = S[dname].run(None, {"decoder_input_ids": np.array([seq], dtype=np.int64), "memory": mem})[0]
                nxt = int(logits[0].argmax())
                if nxt == cfg["eos_token_id"]:
                    break
                seq.append(nxt)
            onnx_read = m.tok.decode(seq[1:], skip_special_tokens=True)
            with torch.no_grad():
                tm = m.code([t])
                torch_read = m.generate(m.eye_probs(m.draw(tm)))[0]
            rows.append({"text": t, "precision": tag, "marks_equal_torch": bool((tm.numpy() == marks).all()),
                         "onnx_reads": onnx_read, "torch_reads": torch_read})
            print(rows[-1], flush=True)
    (out / "parity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
