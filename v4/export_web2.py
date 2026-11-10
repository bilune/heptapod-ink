#!/usr/bin/env python3
"""Leaner browser export: one shared decoder for reading AND rereading, tied vocabulary stored once, fp16 option.

sym_memory.onnx  sym (N,36) int64 -> memory (N,36,512)          [codebooks + adapter, for rereading candidates]
memory.onnx      probs (1,36,84) -> memory (1,36,512)            [already exported; eyes path]
decoder.onnx     decoder_input_ids (N,L), labels (N,L), memory (N,36,512)
                   -> label_logprob (N,L)   log p(label_t | prefix, memory)   (rereading: mean over t = -NLL)
                   -> next_logits (N,V)     logits at the last position       (greedy reading)
                 The LM head reuses the shared embedding matrix (T5 ties them), so the 32k x 512 table is stored once.
decoder_fp16.onnx same graph in half precision for WebGPU (inputs/outputs stay fp32/int64).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402


class SymMemory(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.books, self.a, self.K = m.books, m.adapter, m.books.shape[1]

    def forward(self, sym):
        h = torch.einsum("bsk,skd->bsd", F.one_hot(sym, self.K).to(self.books.dtype), self.books)
        return self.a.out(self.a.enc(h + self.a.pos))


class Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.dec, self.shared = m.t5.decoder, m.t5.shared
        self.scale = m.t5.config.d_model ** -0.5 if m.t5.config.tie_word_embeddings else 1.0
        assert m.t5.config.tie_word_embeddings

    def forward(self, decoder_input_ids, labels, memory):
        h = self.dec(input_ids=decoder_input_ids, encoder_hidden_states=memory, use_cache=False).last_hidden_state
        logits = torch.matmul(h * self.scale, self.shared.weight.t())
        lp = torch.log_softmax(logits.float(), -1)
        top = (logits.argmax(-1) == labels).to(torch.float32)          # 1 where the label is the greedy choice
        return lp.gather(-1, labels[..., None])[..., 0], logits[:, -1].float(), top


@torch.no_grad()
def main():
    out = Path("web/models")
    torch.backends.mha.set_fastpath_enabled(False)
    m, _ = H.load(Path("v4/runs/s2/model.pt"), torch.device("cpu"))
    m.eval()
    m.t5.config._attn_implementation = "eager"
    sym = torch.randint(0, 81, (3, 36))
    labels = m.tok("Hello, earthling.", return_tensors="pt").input_ids.repeat(3, 1)
    start = torch.full_like(labels[:, :1], m.t5.config.decoder_start_token_id)
    dec_in = torch.cat([start, labels[:, :-1]], 1)
    sm = SymMemory(m).eval()
    torch.onnx.export(sm, (sym,), str(out / "sym_memory.onnx"), input_names=["sym"], output_names=["memory"],
                      dynamic_axes={"sym": {0: "N"}, "memory": {0: "N"}}, opset_version=17, dynamo=False)
    mem = sm(sym)
    dec = Decoder(m).eval()
    dyn = {"decoder_input_ids": {0: "N", 1: "L"}, "labels": {0: "N", 1: "L"}, "memory": {0: "N"},
           "label_logprob": {0: "N", 1: "L"}, "next_logits": {0: "N"}, "label_is_top": {0: "N", 1: "L"}}
    torch.onnx.export(dec, (dec_in, labels, mem), str(out / "decoder.onnx"), input_names=["decoder_input_ids", "labels", "memory"],
                      output_names=["label_logprob", "next_logits", "label_is_top"], dynamic_axes=dyn, opset_version=17, dynamo=False,
                      do_constant_folding=False)   # folding would store a transposed copy of the tied vocabulary
    # fp16 copy: export a half-precision clone whose float inputs/outputs are cast at the boundary
    class Half(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def forward(self, decoder_input_ids, labels, memory):
            a, b, c = self.d(decoder_input_ids, labels, memory.half())
            return a.float(), b.float(), c
    import copy
    dh = copy.deepcopy(dec).half()
    torch.onnx.export(Half(dh).eval(), (dec_in, labels, mem), str(out / "decoder_fp16.onnx"),
                      input_names=["decoder_input_ids", "labels", "memory"], output_names=["label_logprob", "next_logits", "label_is_top"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False, do_constant_folding=False)
    print({f.name: round(f.stat().st_size / 1e6, 1) for f in sorted(out.glob("*.onnx"))}, flush=True)
    # parity: rereading scores and greedy reading, fp32 / fp16 ONNX vs PyTorch
    import onnxruntime as ort
    from search import Searcher
    S = {k: ort.InferenceSession(str(out / f"{k}.onnx")) for k in ("sym_memory", "decoder", "decoder_fp16")}
    texts = ["The ink spread slowly across the glass.", "Tom didn't go to school yesterday?", "Hi."]
    srch = Searcher(m)
    for t in texts:
        lab = m.tok(t, return_tensors="pt").input_ids
        sy = torch.randint(0, 81, (5, 36))
        torch_nll = srch.token_nll(srch.sym_mem(F.one_hot(sy, 81).float()), lab.repeat(5, 1)).numpy()
        memx = S["sym_memory"].run(None, {"sym": sy.numpy()})[0]
        di = torch.cat([torch.full_like(lab[:, :1], m.t5.config.decoder_start_token_id), lab[:, :-1]], 1).repeat(5, 1).numpy()
        for k in ("decoder", "decoder_fp16"):
            lp = S[k].run(None, {"decoder_input_ids": di, "labels": lab.repeat(5, 1).numpy(), "memory": memx})[0]
            print(t[:30], k, "max |nll - torch| =", float(np.abs(-lp.mean(1) - torch_nll).max()), flush=True)


if __name__ == "__main__":
    main()
