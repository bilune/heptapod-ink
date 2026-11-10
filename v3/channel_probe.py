"""How much can the ring carry? Random slot choices -> logogram -> a reader
learns to recover each slot's mark and ring segment. No writer involved.
Accuracy per slot tells the usable capacity of a given ring layout."""
import math, sys, time, json
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "v3")
import heptapod as V
INK = V.INK

def probe(slots, mark_p, scale=1.0, steps=1500, batch=48, res=192):
    V.MARK_SCALE = scale
    V.RES = res
    INK.SLOTS, INK.PITCH_DEG = slots, 360 / slots
    V.SLOTS = slots
    rings, marks = V.upsample_library()
    dev = V.device()
    rend = INK.RingMarkRenderer(rings, marks, 1.0, res).to(dev)
    nr, nm = len(rings), len(marks)

    class R(nn.Module):
        def __init__(s):
            super().__init__()
            s.r = V.Reader(10)
            s.ring = nn.Linear(256 * 48, slots * nr)
            s.mark = nn.Linear(256 * 48, slots * nm)
        def forward(s, img):
            h = s.r.memory(img).flatten(1)
            return s.ring(h).view(-1, slots, nr), s.mark(h).view(-1, slots, nm)

    model = R().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    def sample(b):
        ri = torch.randint(0, nr, (b, slots), device=dev)
        mi = torch.randint(1, nm, (b, slots), device=dev) * (torch.rand(b, slots, device=dev) < mark_p)
        return ri, mi
    for step in range(steps):
        ri, mi = sample(batch)
        img = rend(F.one_hot(ri, nr).float(), F.one_hot(mi, nm).float())
        lr, lm = model(V.channel(img, True))
        loss = F.cross_entropy(lr.reshape(-1, nr), ri.reshape(-1)) + F.cross_entropy(lm.reshape(-1, nm), mi.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ri, mi = sample(512)
        lr, lm = model(rend(F.one_hot(ri, nr).float(), F.one_hot(mi, nm).float()))
        racc = (lr.argmax(-1) == ri).float().mean().item()
        macc = (lm.argmax(-1) == mi).float().mean().item()
        mark_present = mi > 0
        macc_present = (lm.argmax(-1)[mark_present] == mi[mark_present]).float().mean().item()
    return {"slots": slots, "res": res, "mark_p": mark_p, "mark_scale": scale, "ring_acc": round(racc, 3), "mark_acc": round(macc, 3),
            "mark_acc_when_present": round(macc_present, 3), "ring_chance": round(1 / nr, 3)}

if __name__ == "__main__":
    import math as _m
    for slots, p, sc, res in [(24, 0.5, 0.25, 192), (36, 0.5, 0.25, 192), (48, 0.5, 0.25, 192), (48, 0.5, 0.25, 288)]:
        t = time.time()
        r = probe(slots, p, sc, res=res)
        print(json.dumps(r), round(time.time() - t), "s", flush=True)
