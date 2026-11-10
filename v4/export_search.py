#!/usr/bin/env python3
"""Export the pieces for sender-side search in the browser (WebGPU).

writer_alt.onnx  input_ids, attention_mask (1,64) -> alt (1,36,16) int64: the 16 nearest codebook symbols per
                 slot, nearest first (alt[...,0] is the plain writer's symbol). int8 per-channel copy for WASM.
scorer.onnx      sym (N,36) int64, labels (1,T) int64 -> nll (N,) float: mean token NLL of the source text under
                 the symbolic reader for each candidate code (teacher forcing). fp32, meant for WebGPU.
Also writes symbol->mark table to config. Then checks: dynamic N/T, and parity with PyTorch scores.
"""
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


class WriterAlt(nn.Module):
    def __init__(self, m, k=16):
        super().__init__()
        self.m, self.k = m, k

    def forward(self, input_ids, attention_mask):
        states = self.m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        raw = self.m.writer(states, attention_mask.bool())
        d = ((raw[:, :, None, :] - self.m.books[None]) ** 2).sum(-1)
        return torch.topk(d, self.k, dim=-1, largest=False).indices


class Scorer(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.K = m.books.shape[1]

    def forward(self, sym, labels):
        n = sym.shape[0]
        x = F.one_hot(sym, self.K).float()
        h = torch.einsum("bsk,skd->bsd", x, self.m.books)
        a = self.m.adapter
        mem = a.out(a.enc(h + a.pos))
        start = torch.full_like(labels[:, :1], self.m.t5.config.decoder_start_token_id)
        dec_in = torch.cat([start, labels[:, :-1]], 1).expand(n, -1)
        logits = self.m.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem), decoder_input_ids=dec_in, use_cache=False).logits
        lp = torch.log_softmax(logits.float(), -1)
        tgt = labels.expand(n, -1)
        return -lp.gather(-1, tgt[..., None])[..., 0].mean(1)


@torch.no_grad()
def main():
    out = Path("web/models")
    torch.backends.mha.set_fastpath_enabled(False)
    m, _ = H.load(Path("v4/runs/s2/model.pt"), torch.device("cpu"))
    m.eval()
    for mod in (m.encoder, m.t5):
        mod.config._attn_implementation = "eager"
    tok = m.tok("Hello, earthling.", return_tensors="pt", padding="max_length", max_length=H.MAX_T)
    w = WriterAlt(m).eval()
    torch.onnx.export(w, (tok.input_ids, tok.attention_mask), str(out / "writer_alt.onnx"), input_names=["input_ids", "attention_mask"],
                      output_names=["alt"], opset_version=17, dynamo=False)
    m.eval()
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(out / "writer_alt.onnx"), str(out / "writer_alt_q8pc.onnx"), weight_type=QuantType.QInt8,
                     per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])
    (out / "writer_alt.onnx").unlink()
    alt = w(tok.input_ids, tok.attention_mask)
    labels = m.tok("Hello, earthling.", return_tensors="pt").input_ids
    sc = Scorer(m).eval()
    sym = alt[:, :, 0].repeat(3, 1)
    torch.onnx.export(sc, (sym, labels), str(out / "scorer.onnx"), input_names=["sym", "labels"], output_names=["nll"],
                      dynamic_axes={"sym": {0: "N"}, "labels": {1: "T"}, "nll": {0: "N"}}, opset_version=17, dynamo=False)
    m.eval()
    print("sizes MB", {f.name: round(f.stat().st_size / 1e6, 1) for f in out.glob("*.onnx")}, flush=True)
    cfg = json.loads((out / "config.json").read_text())
    cfg["sym_to_mark"] = [int(x) for x in m.fsq_to_mark.tolist()]
    cfg["alt_k"] = 16
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    # checks
    import onnxruntime as ort
    s_alt = ort.InferenceSession(str(out / "writer_alt_q8pc.onnx"))
    s_sc = ort.InferenceSession(str(out / "scorer.onnx"))
    for text in ["The ink spread slowly across the glass.", "Tom didn't go to school yesterday?"]:
        enc = m.tok(text, return_tensors="np", padding="max_length", max_length=H.MAX_T)
        a = s_alt.run(None, {"input_ids": enc.input_ids.astype(np.int64), "attention_mask": enc.attention_mask.astype(np.int64)})[0]
        lab = m.tok(text, return_tensors="np").input_ids.astype(np.int64)
        cand = np.repeat(a[:, :, 0], 7, 0)
        for i in range(1, 7):
            cand[i, i * 3] = a[0, i * 3, i]
        onnx_nll = s_sc.run(None, {"sym": cand, "labels": lab})[0]
        torch_nll = sc(torch.tensor(cand), torch.tensor(lab)).numpy()
        print(text, "N=7 T=%d" % lab.shape[1], "max |onnx-torch| =", float(np.abs(onnx_nll - torch_nll).max()), flush=True)


if __name__ == "__main__":
    main()
