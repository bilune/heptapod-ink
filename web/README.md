# Heptapod in the browser

Everything runs client-side with onnxruntime-web (WASM) and the transformers.js T5 tokenizer.

    python web/serve.py        # http://127.0.0.1:8795, from any folder; PORT=9000 to change the port

The server only adds COOP/COEP headers, which multi-threaded WASM needs. The models are not in git:
`ONLY=web HF_REPO=<user>/heptapod-ink bash scripts/download_models.sh` puts them in `web/models/`.
The same page is published as a static Hugging Face Space (see `hf/`).

Models (`web/models/`, ~168 MB total; exported by `v4/export_web.py`, `v4/export_search.py`, `v4/export_web2.py` from `v4/runs/s2/model.pt`):
- `writer_alt_mix.onnx`  input_ids, attention_mask (1,64 padded) -> 16 nearest codebook symbols per slot (1,36,16); [...,0] = plain writer.
  int8 per-channel MatMuls + fp16 vocabulary table (58 MB, no quality loss vs fp32 on 80 dev sentences).
- `render.onnx`          marks (1,36) -> ink (1,288,288), real Arrival ink, fixed ring.
- `eyes_q8pc.onnx`       ink -> mark probabilities (1,36,84).
- `memory.onnx`          mark probabilities -> decoder memory (1,36,512)            [reading from an image]
- `sym_memory.onnx`      candidate symbols (N,36) -> memory (N,36,512)              [rereading]
- `decoder_fp16.onnx`    ONE T5 decoder for both: (decoder_input_ids, labels, memory) -> label log-probs, next-token logits,
                         and whether each label is the greedy choice (free "would it read back exactly?" check).
                         Tied vocabulary stored once (no constant folding), fp16, 83 MB, WebGPU with WASM fallback.
- `tokenizer/`           Xenova/t5-small (transformers.js-compatible; matches the Python token ids).

Loading is staged: the writer + renderer (~64 MB) load first, so a first ring can be drawn right away; the reader
(eyes, memories, decoder, ~103 MB) loads in the background and is awaited only to reread or to read an image.

Rereading in the browser: coordinate search over the 8 nearest symbols per slot, up to 5 improving moves on distinct slots
accepted per round (cumulative combinations re-scored), stop as soon as the scorer says every token is the greedy choice and
the image read through the eyes confirms it; result kept only if not worse than the first try. URL params: k, multi, early, rounds, chunk, gpu.

Benchmark (headless Chromium + WebGPU, M4 Pro, 24 dev sentences: 16 EWT web + 8 Tatoeba; exact = identical string):
| configuration | exact | rereading time median / mean / max |
| --- | --- | --- |
| no rereading | 6/24 | - |
| first version: 16 alternatives, 1 move/round, 20 rounds | 16/24 | 21.5 / 18.7 / 37.1 s |
| + stop when it reads back exactly | 16/24 | 5.6 / 16.2 / 70.0 s |
| + 8 alternatives | 17/24 | 2.7 / 8.6 / 37.4 s |
| + up to 3 moves per round | 18/24 | 2.1 / 4.6 / 22.6 s |
| + free exactness check from the scorer | 18/24 | 2.3 / 4.0 / 14.2 s |
| + up to 5 moves per round (default) | 18/24 | 2.3 / 3.5 / 11.6 s |
Writing (first ring) ~20-30 ms; reading an image ~50-100 ms. Small sample: quality differences of 1-2 sentences are noise.

Ink: derived from the Arrival (2016) logograms by Martine Bertrand, released by Wolfram Research under CC BY-NC 4.0; marks cut and modified. The models and the rings they draw are CC BY-NC 4.0 as well, non-commercial only (see [NOTICE.md](../NOTICE.md)).
