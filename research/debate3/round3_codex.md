The evidence points to objective/parameter coupling, not an indexing bug. I retract my earlier recommendation of entropy-calibrated distance CE plus STE text loss as the first distillation experiment.

`books[arange(S)[None], target]` correctly produces `(B,S,D)`: element `(b,s)` is `books[s,target[b,s]]`. Search returns book indices; `encode()` converts those to mark IDs only afterward. Sum versus mean squared distance preserves argmin. Current `train()` uses MSE; `tau` is unused, and logged agreement is measured before the optimizer update. Verify cached targets came from exactly these books, writer checkpoint and mark mapping.

**Why agreement falls.** Nearest-centroid cells are convex. Moving an independent raw vector straight toward its target centroid cannot lose an already-correct target assignment. Being far from every centroid therefore does not, alone, explain the failure. But optimizing shared writer parameters does not move each output independently toward its centroid: locally, output updates involve the cross-example/cross-slot Jacobian product `J Jᵀ`. Shared decoder layers, slot self-attention and the shared head couple these updates. Large radial reconstruction errors can dominate tiny decision margins, sacrificing many correct slots while reducing average MSE. This implicates optimization, not proof of insufficient architecture capacity.

A short CPU check confirmed indexing and centroid recovery. A two-output shared-parameter counterexample reduced pure MSE **0.895→0.655** while agreement fell **100%→50%**. Independent centroid interpolation preserved initially correct labels. These are tensor checks, not reproduction of the model failure.

Tiny-temperature CE magnifies gradients; separable distance logits can drive norms indefinitely. STE text gradients optimize a surrogate through discontinuous assignments and can oppose imitation. Dividing distances by their current standard deviation changes the objective if differentiated; even detached, it can wash out nearest-neighbor distinctions. Neither failure establishes that discrete imitation is impossible.

**Use a decision-region objective.** Freeze encoder, books and reader; initially remove text loss, fresh commitment and weight decay. For target `t`, use

`Lslot = max(0, m + max[j≠t] ((||z-c_t||²-||z-c_j||²)/(2||c_t-c_j||)))`.

This penalizes violated target half-spaces in raw-space distance units; compute the numerator in affine form to avoid subtracting large distances. Choose a small feasible positive margin from training geometry. Add a weak `||z-z_initial||²` trust penalty. Average changed and unchanged slots separately with equal group weight; report both. Satisfied slots stop demanding centroid relocation, while violated preservation constraints remain active.

**Two-minute gate:** CPU, FP32, fixed seed, cached encoder states for 1–4 sentences, no reader. First check indexed targets, centroid recovery, and direct-vector interpolation; inspect nearest-neighbor margins and duplicate centroids. Then fit the actual writer for the remaining time, capped at 60 updates, evaluating after updates. Require >99% overall target agreement and >95% on search-changed slots, with falling violations and bounded displacement. Failure means diagnose gradients/update size or parameter coupling locally; it does not justify GPU spending.

Keep search at send time now. Distillation is worth revisiting only if this gate passes and held-out visual decoding improves enough to justify replacing the approximately one-second search.
