---
license: cc-by-nc-4.0
language:
- en
base_model: google-t5/t5-small
tags:
- arrival
- emergent-communication
- visual-language
- onnx
- transformers.js
---

# Heptapod Ink

Weights for [Heptapod Ink](https://github.com/bilune/heptapod-ink), a hobby experiment inspired by the film *Arrival*. Two networks learned a ring-shaped writing system made of real ink marks cut from the film's logograms. A writer turns an English sentence into one logogram, and a reader recovers the sentence from the image alone.

Try it in the browser: [Heptapod Ink Space](https://huggingface.co/spaces/bilune/heptapod-ink).

## Files

| File | What it is |
| --- | --- |
| `model.pt` | Full PyTorch checkpoint (v4, stage `s2`): T5-small encoder, slot writer, codebooks, eyes, adapter, T5-small decoder. Load with `v4/heptapod_t5.py` from the code repo. |
| `web/writer_alt_mix.onnx` | Writer: tokens to the 16 nearest symbols per slot (int8 weights, fp16 vocabulary). |
| `web/render.onnx` | Canonical renderer: 36 mark ids to a 288 px ink image. |
| `web/eyes_q8pc.onnx` | Eyes: ink image to mark probabilities per slot. |
| `web/memory.onnx` | Adapter for reading images. |
| `web/sym_memory.onnx` | Adapter for scoring candidate symbols while rereading. |
| `web/decoder_fp16.onnx` | T5 decoder with tied vocabulary, used both for reading and for rereading. |
| `web/config.json`, `web/tokenizer/` | Ring config, symbol to mark table, T5 tokenizer. |

## How well it reads

Locked exam on 2,000 natural sentences from Universal Dependencies English treebanks, scored once:

| Mode | Exact sentence | Words |
| --- | --- | --- |
| Plain writing | 33.5% | |
| Writer rereads before sending | 81.0% | 92.9% |
| Rereading, noisy image | 75.5% | |
| Control: shuffled image | 1.2% | |

Short sentences are easy (3 to 5 words: 94.4% exact) and long ones much harder (9 to 12 words: 67.9%). Random word salad mostly fails, and so do characters T5 does not know (for example "ñ").

## Limits

English only, at most about 12 words per logogram. The code is learned for model-to-model use; humans are not expected to read it.

## License and credits

CC BY-NC 4.0, because the ink comes from the Arrival logograms by Martine Bertrand, released by Wolfram Research under CC BY-NC 4.0. T5-small by Google (Apache 2.0). Training data: Tatoeba, COGS, FineWeb, Wikipedia, CC-News. See `NOTICE.md` in the code repo.
