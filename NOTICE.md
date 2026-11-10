# Third-party material

Heptapod Ink is a non-commercial hobby project. It builds on the following work.

| What | Source | License |
| --- | --- | --- |
| Logogram ink (cut into the marks in `reference/marks.npz`, `reference/fragments.npz`) | Arrival (2016) logograms by Martine Bertrand, released by Wolfram Research in [Arrival-Movie-Live-Coding](https://github.com/WolframResearch/Arrival-Movie-Live-Coding) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| Encoder/decoder initialisation | [T5-small](https://huggingface.co/google-t5/t5-small) (Google) | Apache 2.0 |
| Browser tokenizer | [Xenova/t5-small](https://huggingface.co/Xenova/t5-small) | Apache 2.0 |
| Training sentences | [Tatoeba](https://tatoeba.org) | CC BY 2.0 FR |
| Training sentences | [COGS](https://github.com/najoungkim/COGS) / [ReCOGS](https://github.com/frankaging/ReCOGS) | MIT |
| Training sentences | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ODC-By 1.0 |
| Training sentences | Wikipedia | CC BY-SA 4.0 |
| Training sentences | CC-News | Common Crawl terms of use |
| Evaluation sentences | [Universal Dependencies](https://universaldependencies.org) English EWT, GUM, PUD, LinES, ParTUT | CC BY-SA 4.0 (per treebank) |

Because the ink is CC BY-NC 4.0, the trained weights, ONNX exports and any
logogram images they produce are shared under CC BY-NC 4.0 as well: free to use,
modify and share for non-commercial purposes, with attribution to Martine
Bertrand / Wolfram Research for the ink and to this project for the model.

The datasets are not redistributed here; `scripts/fetch_data.sh` downloads them.
