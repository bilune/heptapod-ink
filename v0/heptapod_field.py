#!/usr/bin/env python3
"""Heptapod Ink: a tiny, trainable experiment in non-sequential 2D semantics.

Only NumPy and Matplotlib are needed. There are no text tokens in the model.
Each input and output is a two-dimensional binary relation field.

This is an original research prototype inspired by Arrival, NOT canonical
Heptapod B and NOT a model of the fictional aliens' language.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compose_exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean relation composition: c[i,k] = exists j: a[i,j] & b[j,k]."""
    return (a.astype(np.int16) @ b.astype(np.int16) > 0).astype(np.float32)


def random_graphs(rng: np.random.Generator, count: int, n: int, density: float):
    a = (rng.random((count, n, n)) < density).astype(np.float32)
    b = (rng.random((count, n, n)) < density).astype(np.float32)
    return a, b


def aligned_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Prepare unordered 2D relational evidence, not a token sequence.

    For every output cell (i,k), compare all possible intermediate entities j.
    The same learned gate is applied at every (i,j,k); this parameter sharing
    lets us evaluate on more entities than existed in the training examples.
    """
    batch, n, _ = a.shape
    left = np.broadcast_to(a[:, :, None, :], (batch, n, n, n))
    right = np.broadcast_to(b.transpose(0, 2, 1)[:, None, :, :], (batch, n, n, n))
    return np.stack((left, right), axis=-1)


class SpatialComposer:
    """Trainable binary gate + permutation-invariant noisy-OR over entities.

    The gate is an MLP: two numeric relation values -> 16 hidden units -> a
    probability. The entire network is trained end to end on composed relation
    *fields* using binary cross-entropy and NumPy backpropagation.
    """

    def __init__(self, seed: int = 7, hidden: int = 16, lr: float = 0.025):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.4, size=(2, hidden)).astype(np.float64)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = rng.normal(0, 0.2, size=(hidden,)).astype(np.float64)
        self.b2 = np.array(-1.0, dtype=np.float64)
        self.parameters = ("w1", "b1", "w2", "b2")
        self.m = {name: np.zeros_like(getattr(self, name)) for name in self.parameters}
        self.v = {name: np.zeros_like(getattr(self, name)) for name in self.parameters}
        self.step = 0
        self.lr = lr

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))

    def forward(self, a: np.ndarray, b: np.ndarray, *, cache: bool = False):
        shape = a.shape
        assert a.shape == b.shape and a.ndim == 3 and shape[-1] == shape[-2]
        x = aligned_pairs(a, b).astype(np.float64)
        hidden = np.tanh(x @ self.w1 + self.b1)
        local = self.sigmoid(hidden @ self.w2 + self.b2)
        # The evidence from different potential intermediary entities is
        # aggregated simultaneously; there is no left-to-right reading order.
        field = 1.0 - np.prod(1.0 - local, axis=-1)
        if cache:
            return field, (x, hidden, local)
        return field

    def fit_step(self, a: np.ndarray, b: np.ndarray):
        target = compose_exact(a, b)
        predicted, (x, hidden, local) = self.forward(a, b, cache=True)
        eps = 1e-10
        clipped = np.clip(predicted, eps, 1.0 - eps)
        loss = -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean()

        # For p = 1 - prod_j(1-q_j), the BCE gradient w.r.t. each gate logit is
        # q_j * (p - target) / p. Its shape is (batch, source, target, mediator).
        output_gradient = (predicted - target) / np.clip(predicted, eps, 1.0)
        dz = local * output_gradient[..., None] / target.size
        grads = {
            "w2": hidden.reshape(-1, hidden.shape[-1]).T @ dz.reshape(-1),
            "b2": np.array(dz.sum()),
        }
        dh = dz[..., None] * self.w2 * (1 - hidden * hidden)
        grads["w1"] = x.reshape(-1, 2).T @ dh.reshape(-1, hidden.shape[-1])
        grads["b1"] = dh.sum(axis=tuple(range(dh.ndim - 1)))

        self.step += 1
        for name in self.parameters:
            grad = np.clip(grads[name], -3.0, 3.0)
            self.m[name] = 0.9 * self.m[name] + 0.1 * grad
            self.v[name] = 0.999 * self.v[name] + 0.001 * grad * grad
            mhat = self.m[name] / (1.0 - 0.9**self.step)
            vhat = self.v[name] / (1.0 - 0.999**self.step)
            setattr(self, name, getattr(self, name) - self.lr * mhat / (np.sqrt(vhat) + 1e-8))
        return float(loss)

    def infer(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim == 2:
            return self.infer(a[None], b[None])[0]
        return (self.forward(a, b) >= 0.5).astype(np.float32)

    def gate_table(self):
        examples = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        z = np.tanh(examples @ self.w1 + self.b1) @ self.w2 + self.b2
        return {f"{int(a)}{int(b)}": round(float(p), 6) for (a, b), p in zip(examples, self.sigmoid(z))}

    def save(self, filename: Path):
        np.savez(filename, **{name: getattr(self, name) for name in self.parameters})

    @classmethod
    def load(cls, filename: Path) -> "SpatialComposer":
        model = cls()
        with np.load(filename) as weights:
            for name in model.parameters:
                setattr(model, name, weights[name])
        return model


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    actual = actual.astype(bool)
    prediction = prediction.astype(bool)
    true_positive = int((actual & prediction).sum())
    false_positive = int((~actual & prediction).sum())
    false_negative = int((actual & ~prediction).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "cell_accuracy": round(float((actual == prediction).mean()), 5),
        "matrix_exact_match": round(float(np.all(actual == prediction, axis=(-2, -1)).mean()), 5),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 5),
    }


def draw_field(ax, field: np.ndarray, label: str, *, color: str = "#152331"):
    """A deterministic radial rendering of an N x N semantic field.

    Angle identifies the relation's source entity. Radius identifies its
    target entity. Each mark is one directed relation. The marks are readable
    and reversible by construction, but this visual grammar is OUR invention.
    """
    n = field.shape[0]
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    angles = np.linspace(0, 2 * np.pi, 600)
    ax.plot(0.98 * np.cos(angles), 0.98 * np.sin(angles), color="#cad4d9", lw=1.2)
    ax.plot(0.30 * np.cos(angles), 0.30 * np.sin(angles), color="#e5e9ed", lw=1)
    for source in range(n):
        theta = 2 * np.pi * (source + 0.5) / n + np.pi / 2
        for target in range(n):
            if field[source, target] < 0.5:
                continue
            radius = 0.38 + 0.52 * (target + 0.5) / n
            arc = np.linspace(theta - 0.38 * np.pi / n, theta + 0.38 * np.pi / n, 25)
            ax.plot(radius * np.cos(arc), radius * np.sin(arc), color=color, lw=4.0, solid_capstyle="round")
            ax.plot(
                [radius * np.cos(theta), (radius + 0.09) * np.cos(theta)],
                [radius * np.sin(theta), (radius + 0.09) * np.sin(theta)],
                color=color, lw=2.4, solid_capstyle="round",
            )
    ax.set_title(label, fontsize=11, color="#152331", pad=12)


def save_demo(model: SpatialComposer, output: Path, seed: int):
    rng = np.random.default_rng(seed + 100)
    n = 6
    # Find an example with neither an empty nor a saturated field.
    for _ in range(100):
        a, b = random_graphs(rng, 1, n, 0.27)
        target = compose_exact(a, b)[0]
        if 6 <= target.sum() <= 24:
            break
    predicted = model.infer(a, b)[0]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#fafbfc")
    for ax in axes.ravel():
        ax.set_facecolor("#fafbfc")
    items = [(a[0], "A · trusts"), (b[0], "B · helps"), (target, "A ∘ B · truth"), (predicted, "A ∘ B · model")]
    for i, (field, title) in enumerate(items):
        draw_field(axes[0, i], field, title)
        axes[1, i].imshow(field, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        for edge in (np.arange(n + 1) - 0.5):
            axes[1, i].axhline(edge, color="#cfd9de", lw=0.3)
            axes[1, i].axvline(edge, color="#cfd9de", lw=0.3)
    fig.suptitle("HEPTAPOD FIELD  ·  Relations as geometry, not text", fontsize=14, y=0.99)
    fig.text(0.5, 0.025, "The circular drawing is a deterministic render of a 2D semantic matrix; it is NOT the official Arrival alphabet.", ha="center", fontsize=9, color="#596875")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def run(seed: int, steps: int, batch: int, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    model = SpatialComposer(seed=seed)
    history = []
    for step in range(steps):
        # Train exclusively on 4-entity worlds with varied densities.
        density = float(rng.uniform(0.13, 0.60))
        a, b = random_graphs(rng, batch, 4, density)
        loss = model.fit_step(a, b)
        if step == 0 or (step + 1) % max(1, steps // 20) == 0:
            history.append({"step": step + 1, "loss": round(loss, 6)})

    checks = {}
    for n, density, title in [(4, 0.3, "in_distribution_4"), (7, 0.3, "unseen_size_7"), (12, 0.22, "unseen_size_12"), (7, 0.52, "dense_world_7")]:
        a, b = random_graphs(rng, 700, n, density)
        checks[title] = metrics(compose_exact(a, b), model.infer(a, b))

    # Test whether the model supports repeated, non-sequential 2D composition.
    a, b = random_graphs(rng, 500, 7, 0.23)
    c = (rng.random((500, 7, 7)) < 0.23).astype(np.float32)
    exact_twice = compose_exact(compose_exact(a, b), c)
    predicted_twice = model.infer(model.infer(a, b), c)
    checks["two_hops_composed_twice_7"] = metrics(exact_twice, predicted_twice)

    # A meaningful negative control: scramble intermediary identities in B.
    permuted_b = b[:, ::-1, :]
    control = metrics(compose_exact(a, b), model.infer(a, permuted_b))

    result = {
        "hypothesis": "Can a trained spatial operator manipulate 2D relational meaning and generalize to more entities without text tokens?",
        "training_size": 4,
        "training_updates": steps,
        "learned_gate": model.gate_table(),
        "metrics": checks,
        "intermediary_identity_scramble_control": control,
        "history": history,
        "limitations": [
            "Synthetic binary relations, not natural-language meanings.",
            "The 2D relational grammar and aggregator are engineered, not discovered.",
            "Radial drawings are a reversible visualization, not visual inputs to the trained model.",
            "Heptapod B does not exist as a complete, learnable official grammar.",
        ],
    }
    model.save(output / "model.npz")
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_demo(model, output / "demo.png", seed)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot([h["step"] for h in history], [h["loss"] for h in history], color="#244d69", lw=2)
    ax.set(xlabel="Training updates", ylabel="Loss (BCE)", title="Learning 2D relation composition")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"gate": result["learned_gate"], "metrics": checks, "scrambled_control": control}, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("out"))
    args = parser.parse_args()
    run(args.seed, args.steps, args.batch, args.output)


if __name__ == "__main__":
    main()
