# Heptapod Ink v0: thoughts in two dimensions

A small, reproducible and **trainable** experiment to test an idea inspired by *Arrival*: can a model
represent and manipulate relations directly in a two-dimensional field, without words as an
intermediate representation?

**Status:** proof of concept of *relational reasoning in 2D*. It is not a translation of canonical
Heptapod B, nor a solution to the general problem of visual language. This was the starting
prototype of the project, written with ChatGPT's help; later versions took a different direction
(see [docs/history.md](../docs/history.md)).

## What it learns

A relation between entities is an `N x N` binary matrix. Cell `(i, j)` says that `i` is related to
`j`. These are 2D fields, not sentences or token sequences.

The model gets two fields, for example:

- `A[i,j]`: entity `i` **trusts** `j`.
- `B[j,k]`: entity `j` **helps** `k`.

It has to produce a third field `C[i,k]`: entity `i` trusts someone who helps `k`.

The composition has this exact semantics:

\[
C[i,k] = \bigvee_j (A[i,j] \land B[j,k])
\]

A small neural network learns what each pair of values `(A[i,j], B[j,k])` means. The system combines
the evidence from all intermediaries `j` at once with a `noisy-OR` aggregation, which does not depend
on a reading order.

## Try it

```bash
python -m pip install numpy matplotlib
python v0/heptapod_field.py --steps 900 --output v0/out
```

It produces:

- `out/results.json`: metrics and the values learned by the gate.
- `out/model.npz`: trained weights (for inference without retraining).
- `out/demo.png`: input fields, expected field and generated field, with a circular rendering.
- `out/training.png`: learning curve.

Pure NumPy for training and inference, no PyTorch or GPU. Any laptop will do.

To reuse the saved weights (from inside `v0/`):

```python
import numpy as np
from pathlib import Path
from heptapod_field import SpatialComposer

model = SpatialComposer.load(Path("out/model.npz"))
relation_a = np.eye(7, dtype=np.float32)
relation_b = np.roll(relation_a, 1, axis=1)
composed = model.infer(relation_a, relation_b)  # another 2D field
```

## What it checks

1. **New compositions:** it trains on random worlds and is evaluated on unseen ones, so it cannot
   memorise whole matrices.
2. **Structural generalisation:** trained with `N=4` entities, evaluated with `N=7` and `N=12`
   without retraining.
3. **Multi-step composition:** the output of one 2D operation is the input of the next.
4. **Negative control:** shuffling the intermediate identities should make quality drop, because the
   relational meaning no longer lines up.

## What it does NOT show

- The input is not natural language or an arbitrary visual symbol: it is a synthetic relational graph.
- The 2D representation and the aggregation were designed by hand; **the model learns the local
  operator (in practice a 4-cell AND gate), it does not invent a language**. The 100% scores follow
  from the architecture.
- The circular picture is a deterministic rendering of the fields, not an official logogram and not
  an image the model reads.
- It does not show that the model thinks without human concepts, or that the representation is
  better than a sequential one.

## Next steps it proposed

- **V1. Visual perception.** An image -> 2D field encoder with circular symbols, varying stroke,
  rotation, noise and occlusion; exact graph reconstruction as the metric.
- **V2. Compositional language.** Several predicates, negation, quantifiers, tense and causality, with
  held-out combinations.
- **V3. Emergent representation.** Replace the hand-made visual code with a trainable encoder/decoder
  and a 2D bottleneck, and make two models communicate only through it.
- **V4. Heptapod-like.** Render the representations with a circular grammar and test whether a second
  model decodes from the image alone.

The project ended up going down the "emergent representation" road (v1 onwards).
