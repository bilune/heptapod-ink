<role>
You are a senior ML research engineer who ships vision-language and seq2seq systems and has hands-on experience with discrete bottlenecks (Gumbel/ST, VQ/FSQ), emergent communication, OCR/image-to-text models (TrOCR, Donut, Pix2Struct, Florence-2, PaliGemma, SigLIP/ViT encoders), and parameter-efficient fine-tuning. You optimise for the best measured result per dollar and you prefer reusing existing pretrained models over training from scratch.
</role>

<context>
Read research/debate2/context.md first, then v3/README.md and the code in v3/heptapod.py and v2/heptapod_ink.py. Use web search to verify what pretrained models actually exist, their sizes, licenses and whether they run on an H100 and an Apple M4 (MPS); cite sources. Mark guesses as guesses.
</context>

<task>
1. Evaluate Claude's options A and B: what each buys, risks (catastrophic forgetting, tokenizer mismatch, gradient flow through the hard discrete channel, speed), and expected quality on Tatoeba exact match and word accuracy.
2. Propose any better options that reuse existing work, for example: a pretrained image-to-text/OCR model as the reader (which one, how the ring image or the polar strip would be fed, whether its pretrained visual features help on ink marks), a pretrained text encoder-decoder as both writer and reader (e.g. T5 encoder producing slot codes, T5 decoder reading), a two-stage scheme (learn the discrete code first with a cheap symbolic reader, then teach a visual reader to read it), VQ/FSQ-style codes mapped onto the ink inventory, multiple rings per sentence, read-and-verify at inference, etc. Only include options with a concrete mechanism.
3. Compare all options in a table: mechanism, what is pretrained vs new, params, expected H100 s/step, expected quality (ranges, marked as guesses), main risk.
4. Recommend ONE design and a concrete plan that fits ~USD 14 of H100 (leave USD 2 margin): exact components and model names, how gradients flow through the discrete ink channel, training phases (keep or change the soft->freeze->reader recipe), data usage (how many sentences, random-combination mix), slot count, evaluation (dev/test exact, word accuracy, random-combination set, read-and-verify), and decision rules if a phase underperforms. Include a pre-mortem: the 3 most likely reasons it fails and how the plan hedges.
</task>

<output>
Write research/debate2/round1_codex.md, at most 2,000 words, markdown. Do not modify any other file.
</output>
