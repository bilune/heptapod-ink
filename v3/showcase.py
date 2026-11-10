"""Write sentences as logograms with a trained model, export PNGs, read the PNGs back."""
import sys, random
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, "v3")
import heptapod as V

V.OUT = Path(sys.argv[1]).resolve()
dev = V.device()
model = V.load(dev)
test = V.load_split("test")
rng = random.Random(7)
sentences = ["A girl liked the donut .", "The donut was liked by a girl .",
             "Emma gave the cake to a dog .", "A dog gave the cake to Emma .",
             "The boy needed to sleep .", "Liam said that a cat ate the cake on the table ."]
sentences += [r["text"] for r in rng.sample(test, 6)]
out = V.OUT / "showcase"; out.mkdir(exist_ok=True)

def read_png(path):
    g = np.asarray(Image.open(path).convert("L").resize((V.RES, V.RES), Image.LANCZOS)) / 255.0
    img = torch.tensor(1 - g, dtype=torch.float32, device=dev)[None]
    return model.decode(model.reader.generate(img, model.bos, model.eos))[0]

fig, axes = plt.subplots(3, 4, figsize=(18, 15.5), facecolor="white")
with torch.no_grad():
    for i, (ax, s) in enumerate(zip(axes.flat, sentences)):
        rc, mc = model.writer(*model.read_text(model.tokenize([s])))
        hi = V.paint(rc.cpu(), mc.cpu(), 1024)[0].numpy()
        path = out / f"{i:02d}.png"
        plt.imsave(path, np.repeat((1 - hi)[..., None], 3, -1))
        got = read_png(path)
        ok = got == V.normalize(s)
        ax.imshow(1 - hi, cmap="gray", vmin=0, vmax=1); ax.set_axis_off()
        ax.set_title(f"{V.normalize(s)}", fontsize=10.5, color="#222", wrap=True)
        ax.text(0.5, -0.05, ("✓ " if ok else "✗ ") + got, transform=ax.transAxes, ha="center",
                fontsize=10, color="#1a7f37" if ok else "#c0392b", wrap=True)
        print(("OK " if ok else "XX ") + V.normalize(s) + "  ->  " + got)
fig.suptitle("Heptapod Ink v3 (freeze25) · top: what it was asked to write · bottom: what it read back from the exported PNG",
             fontsize=13)
fig.tight_layout()
fig.savefig(out / "showcase.png", dpi=80, facecolor="white")
print(out / "showcase.png")
