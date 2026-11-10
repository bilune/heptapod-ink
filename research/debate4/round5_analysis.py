"""Frozen, CPU-only round-5 analyses. Each invocation has a 165-second deadline.

Run from repository root with .venv/bin/python, modes: base, ring, stats,
decoder, sensitivity, decoder_controls, figure.
No fitting of model parameters; baseline quantizers are fixed random projections.
"""
import os
import signal
import time

os.environ.update(HEPTAPOD_DEVICE="cpu", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  MPLCONFIGDIR="/tmp/heptapod-round5-mpl", TOKENIZERS_PARALLELISM="false")
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("165s deadline")))
signal.alarm(165)
START = time.monotonic()
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "v4"))
import semantic_probe as P
import heptapod_t5 as H


def corpus():
    rng = random.Random(505)
    rows = []
    for split in ["dev", "mixed_val"]:
        data = json.loads((ROOT / "v3/data_tatoeba" / f"{split}.json").read_text())
        unique = list(dict.fromkeys(r["sentence"] for r in data))
        rows.extend({"sentence": s, "source": split} for s in rng.sample(unique, 192))
    frames = []
    for pos, quads in P.WORDS.items():
        for frame in P.FRAMES[pos]:
            for q in quads:
                frames.append({"pos": pos, "frame": frame, "words": q})
                rows.extend({"sentence": frame.format(w), "source": "frame"} for w in q)
    rows.extend({"sentence": s, "source": "triplet"} for t in P.PARA for s in t)
    return rows, frames


def pool(states, mask):
    return (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)


@torch.inference_mode()
def embeddings(mode):
    from transformers import T5EncoderModel, AutoTokenizer
    rows, frames = corpus()
    texts = [r["sentence"] for r in rows]
    vecs, codes, qvecs, lengths = [], [], [], []
    if mode == "base":
        tok = AutoTokenizer.from_pretrained(H.T5, local_files_only=True)
        enc = T5EncoderModel.from_pretrained(H.T5, local_files_only=True).eval()
    else:
        m, _ = H.load(ROOT / "v4/runs/s2/model.pt", torch.device("cpu"))
        m.eval()
        assert m.quant == "vq"
        tok, enc = m.tok, m.encoder
    for start in range(0, len(texts), 16):
        b = tok(texts[start:start+16], return_tensors="pt", padding=True, truncation=True, max_length=64)
        states = enc(**b).last_hidden_state
        vecs.append(pool(states, b.attention_mask).numpy())
        lengths.extend(b.attention_mask.sum(1).tolist())
        if mode == "ring":
            raw = m.writer(states, b.attention_mask.bool()).float()
            ids = m.assign(raw)
            codes.append(m.fsq_to_mark[ids].numpy())
            qvecs.append(m.books[torch.arange(36)[None], ids].flatten(1).numpy())
        if start % 128 == 0:
            print(mode, start, "seconds", round(time.monotonic()-START, 1), flush=True)
    result = dict(vectors=np.concatenate(vecs), lengths=np.array(lengths))
    if codes:
        result.update(codes=np.concatenate(codes), quantized=np.concatenate(qvecs))
    np.savez_compressed(OUT / f"round5_{mode}.npz", **result)
    (OUT / "round5_items.json").write_text(json.dumps({"rows": rows, "frames": frames}, indent=1))
    print("saved", mode, len(rows), "seconds", time.monotonic()-START, flush=True)


def cosine(x):
    x = x.astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.clip(1 - x @ x.T, 0, 2)


def hamming(c):
    return (c[:, None] != c[None, :]).sum(-1).astype(float)


def stats():
    from scipy.stats import spearmanr, pearsonr, norm
    b, f = [np.load(OUT / f"round5_{s}.npz") for s in ["base", "ring"]]
    n = 384
    x = b["vectors"][:n].astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    db, df = cosine(x), cosine(f["vectors"][:n])
    dh = hamming(f["codes"][:n])
    dq = cosine(f["quantized"][:n])
    ij = np.triu_indices(n, 1)
    rho = lambda a, b: float(spearmanr(a, b).statistic)
    result = {"n_sentences": n, "n_pairs": len(ij[0]), "sampling_seed": 505,
              "nominal_bits": float(36*np.log2(81)),
              "rsa": {"ring_base": rho(dh[ij], db[ij]), "ring_finetuned": rho(dh[ij], df[ij]),
                      "base_finetuned": rho(db[ij], df[ij]), "codebook_base": rho(dq[ij], db[ij])}}
    def entropy(c):
        return float(sum(-sum(p*np.log2(p) for p in np.bincount(col)[np.bincount(col)>0]/len(col)) for col in c.T))
    result["sum_slot_empirical_entropies"] = {"ring": entropy(f["codes"][:n])}
    baselines = {}
    all_dist = {}
    for method in ["ternary4", "scalar81", "argmax81"]:
        scores, entropies, distances = [], [], []
        for seed in range(20):
            rng = np.random.default_rng(50500+seed)
            if method == "ternary4":
                z = (x @ rng.normal(size=(512, 144))).reshape(n, 36, 4)
                digits = np.digitize(z, [-norm.ppf(2/3), norm.ppf(2/3)])
                code = (digits * (3**np.arange(4))).sum(-1)
            elif method == "scalar81":
                code = np.digitize(x @ rng.normal(size=(512, 36)), norm.ppf(np.arange(1,81)/81))
            else:
                proj = torch.tensor(rng.normal(size=(512,36*81)), dtype=torch.float32)
                code = (torch.tensor(x, dtype=torch.float32) @ proj).reshape(n,36,81).argmax(-1).numpy()
            d = hamming(code)
            scores.append(rho(d[ij], db[ij]))
            entropies.append(entropy(code))
            distances.append(d)
        baselines[method] = {"mean_rho": float(np.mean(scores)), "seed_range": [min(scores), max(scores)],
                             "scores": scores, "mean_sum_slot_entropy": float(np.mean(entropies))}
        all_dist[method] = np.stack(distances)
        print(method, baselines[method]["mean_rho"], flush=True)
    result["baselines"] = baselines
    rng = np.random.default_rng(505)
    null = []
    for _ in range(200):
        perm = rng.permutation(n)
        null.append(rho(dh[perm[:,None],perm][ij], db[ij]))
    result["whole_code_permutation_null"] = {"mean": float(np.mean(null)), "range95": np.percentile(null,[2.5,97.5]).tolist()}
    boot = []
    # Resample sentence identities, not pair entries. Omit duplicate-identity pairs.
    for _ in range(200):
        ids = rng.integers(n, size=n)
        a, c = ids[ij[0]], ids[ij[1]]
        keep = a != c
        a, c = a[keep], c[keep]
        boot.append(rho(dh[a,c], db[a,c]))
    result["ring_base_sentence_bootstrap_ci95"] = np.percentile(boot, [2.5,97.5]).tolist()
    for split, ix in [("dev", np.arange(192)), ("mixed_val", np.arange(192,384))]:
        sub = np.triu_indices(len(ix), 1)
        a, c = ix[sub[0]], ix[sub[1]]
        result["rsa"][split] = {"ring": rho(dh[a,c],db[a,c]),
                                "ternary4_mean": float(np.mean([rho(d[a,c],db[a,c]) for d in all_dist["ternary4"]]))}
    # Existing minimal-pair panel, independent of the natural-sentence RSA above.
    fullb, fullf = cosine(b["vectors"]), cosine(f["vectors"])
    codes = f["codes"]
    start = 384
    frame_dist = {}
    for role, k in [("synonym",1),("contrast",2),("unrelated",3)]:
        a = start + 4*np.arange(92)
        c = a+k
        frame_dist[role] = {"base_mean": float(fullb[a,c].mean()), "ft_mean": float(fullf[a,c].mean()),
                            "hamming_mean": float((codes[a]!=codes[c]).sum(1).mean())}
    result["frame_distances"] = frame_dist
    start += 368
    a = start + 3*np.arange(12)
    triples = {}
    for name, d in [("base",fullb),("finetuned",fullf)]:
        para, rev = d[a,a+1], d[a,a+2]
        triples[name] = {"paraphrase_mean": float(para.mean()), "reversal_mean": float(rev.mean()),
                          "reversal_closer_count": int((rev<para).sum()), "paraphrase": para.tolist(), "reversal": rev.tolist()}
    para, rev = (codes[a]!=codes[a+1]).sum(1), (codes[a]!=codes[a+2]).sum(1)
    triples["ring"] = {"paraphrase_mean": float(para.mean()), "reversal_mean": float(rev.mean()),
                       "reversal_closer_count": int((rev<para).sum()), "ties": int((rev==para).sum()),
                       "paraphrase": para.tolist(), "reversal": rev.tolist()}
    result["triplets"] = triples
    # Partial rank association with surface token overlap and length, not causal attribution.
    from scipy.stats import rankdata
    rows, _ = corpus()
    import re
    sets = [set(re.findall(r"[a-z]+", r["sentence"].lower())) for r in rows[:n]]
    lex = np.array([1-len(sets[a]&sets[c])/len(sets[a]|sets[c]) for a,c in zip(*ij)])
    ln = b["lengths"][:n]
    controls = np.stack([np.ones(len(lex)), rankdata(db[ij]), rankdata(abs(ln[ij[0]]-ln[ij[1]])),
                         rankdata(ln[ij[0]]+ln[ij[1]])], axis=1)
    resid = lambda y: y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
    result["partial_ring_lexical_given_base_and_lengths"] = float(pearsonr(resid(rankdata(dh[ij])),resid(rankdata(lex))).statistic)
    result["elapsed_seconds"] = time.monotonic()-START
    (OUT / "round5_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


@torch.inference_mode()
def decoder():
    """Symmetric word-evidence loss when exchanging rings within a shared frame.

    This measures reader discrimination, NOT a context-only decoder prior.
    """
    import torch.nn.functional as F
    from transformers.modeling_outputs import BaseModelOutput
    from scipy.stats import spearmanr, rankdata, pearsonr
    rows, frames = corpus()
    b, f = [np.load(OUT / f"round5_{s}.npz") for s in ["base", "ring"]]
    m, _ = H.load(ROOT / "v4/runs/s2/model.pt", torch.device("cpu"))
    m.eval()
    codes = torch.tensor(f["codes"][384:752])
    memory = m.memory(F.one_hot(codes, m.script.n_marks).float())
    texts = [r["sentence"] for r in rows[384:752]]
    positions = []
    for i, s in enumerate(texts):
        word = frames[i//4]["words"][i%4]
        pre = s[:s.index(word)].rstrip()
        before = m.tok(pre, add_special_tokens=False).input_ids
        upto = m.tok(pre+" "+word, add_special_tokens=False).input_ids
        full = m.tok(s).input_ids
        assert upto[:len(before)] == before and full[:len(upto)] == upto
        positions.append((len(before),len(upto)))
    tasks = [(i,j) for frame in range(92) for i in range(frame*4,frame*4+4) for j in range(frame*4,frame*4+4)]
    lp = np.zeros((92,4,4))  # frame, candidate text, ring
    for start in range(0,len(tasks),16):
        chunk = tasks[start:start+16]
        ids = m.tokenize([texts[i] for i,j in chunk])
        labels = ids["input_ids"].masked_fill(~ids["attention_mask"].bool(),-100)
        logits = m.t5(encoder_outputs=BaseModelOutput(last_hidden_state=memory[[j for i,j in chunk]]), labels=labels).logits
        logp = F.log_softmax(logits.float(),-1).gather(-1,ids["input_ids"][...,None]).squeeze(-1)
        for row,(i,j) in enumerate(chunk):
            lo,hi = positions[i]
            lp[i//4,i%4,j%4] = float(logp[row,lo:hi].sum())
        if start % 256 == 0:
            print("decoder",start,"seconds",round(time.monotonic()-START,1),flush=True)
    db, df = cosine(b["vectors"][384:752]), cosine(f["vectors"][384:752])
    dh = hamming(f["codes"][384:752])
    measures = []
    for frame in range(92):
        for k in [1,2,3]:
            a,c = 4*frame,4*frame+k
            gap = .5*(lp[frame,0,0]-lp[frame,0,k]+lp[frame,k,k]-lp[frame,k,0])
            measures.append([frame,k,dh[a,c],db[a,c],df[a,c],gap])
    z=np.array(measures)
    r = lambda a,b: float(spearmanr(a,b).statistic)
    result={"n_comparisons":len(z),"ring_vs_decoder_discrimination_rho":r(z[:,2],z[:,5]),
            "base_vs_decoder_discrimination_rho":r(z[:,3],z[:,5]),
            "ft_vs_decoder_discrimination_rho":r(z[:,4],z[:,5]),
            "local_ring_base_rho":r(z[:,2],z[:,3]),
            "local_ring_base_pearson":float(pearsonr(z[:,2],z[:,3]).statistic)}
    controls=np.stack([np.ones(len(z)),rankdata(z[:,3]),rankdata(z[:,4])],axis=1)
    resid=lambda y:y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
    result["partial_ring_decoder_given_both_encoders"]=float(pearsonr(resid(rankdata(z[:,2])),resid(rankdata(z[:,5]))).statistic)
    rng=np.random.default_rng(505)
    boots=[]
    for _ in range(1000):
        # Word quadruple cluster: four adjective/noun frames or three verb frames.
        groupids=np.array([list(P.WORDS).index(fr["pos"])*10+P.WORDS[fr["pos"]].index(tuple(fr["words"])) for fr in frames])
        groups=np.unique(groupids)
        inds=np.concatenate([np.flatnonzero(np.repeat(groupids,3)==g) for g in rng.choice(groups,len(groups))])
        t=z[inds]
        ctrl=np.stack([np.ones(len(t)),rankdata(t[:,3]),rankdata(t[:,4])],axis=1)
        u=rankdata(t[:,2]);v=rankdata(t[:,5])
        boots.append(float(pearsonr(u-ctrl@np.linalg.lstsq(ctrl,u,rcond=None)[0],v-ctrl@np.linalg.lstsq(ctrl,v,rcond=None)[0]).statistic))
    result["partial_wordquad_cluster_ci95"]=np.percentile(boots,[2.5,97.5]).tolist()
    result["elapsed_seconds"]=time.monotonic()-START
    result["comparison_columns"]=["frame","role_index","hamming","base_cosdist","ft_cosdist","symmetric_word_logprob_loss"]
    result["comparisons"]=measures
    (OUT/"round5_decoder.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="comparisons"},indent=2))


def sensitivity():
    from scipy.stats import spearmanr, norm
    b, f = [np.load(OUT / f"round5_{s}.npz") for s in ["base", "ring"]]
    n=384
    ij=np.triu_indices(n,1)
    dh=hamming(f["codes"][:n]);df=cosine(f["vectors"][:n])
    r=lambda a,b:float(spearmanr(a,b).statistic)
    x=f["vectors"][:n].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True)
    result={"ring_ft":r(dh[ij],df[ij]),"codebook_ft":r(cosine(f["quantized"][:n])[ij],df[ij]),
            "ring_hamming_percentiles":np.percentile(dh[ij],[0,10,25,50,75,90,100]).tolist(),
            "ring_hamming_36_fraction":float((dh[ij]==36).mean())}
    for method in ["ternary4","argmax81"]:
        scores=[]
        for seed in range(20):
            rng=np.random.default_rng(50500+seed)
            if method=="ternary4":
                z=(x@rng.normal(size=(512,144))).reshape(n,36,4)
                code=(np.digitize(z,[-norm.ppf(2/3),norm.ppf(2/3)])*(3**np.arange(4))).sum(-1)
            else:
                code=(torch.tensor(x,dtype=torch.float32)@torch.tensor(rng.normal(size=(512,36*81)),dtype=torch.float32)).reshape(n,36,81).argmax(-1).numpy()
            scores.append(r(hamming(code)[ij],df[ij]))
        result[method]={"mean_rho":float(np.mean(scores)),"seed_range":[min(scores),max(scores)]}
    # Same metric for local/global geometry, to avoid conflating Pearson and Spearman.
    a=384+4*np.repeat(np.arange(92),3);c=a+np.tile([1,2,3],92)
    result["local_base_rho"]=r((f["codes"][a]!=f["codes"][c]).sum(1),cosine(b["vectors"])[a,c])
    result["elapsed_seconds"]=time.monotonic()-START
    (OUT/"round5_sensitivity.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


def decoder_controls():
    from scipy.stats import rankdata, pearsonr, spearmanr
    from transformers import AutoTokenizer
    rows,frames=corpus()
    z=np.array(json.loads((OUT/"round5_decoder.json").read_text())["comparisons"])
    tok=AutoTokenizer.from_pretrained(H.T5,local_files_only=True)
    # Add candidate-word piece counts and contrast-class controls to both geometries.
    counts=np.array([[len(tok(w,add_special_tokens=False).input_ids) for w in fr["words"]] for fr in frames])
    a=np.repeat(counts[:,0],3);c=counts[:,1:].flatten()
    extra=np.stack([a+c,abs(a-c),(z[:,1]==2).astype(float),(z[:,1]==3).astype(float)],axis=1)
    def partial(ix):
        t=z[ix]
        ctrl=np.column_stack([np.ones(len(ix)),rankdata(t[:,3]),rankdata(t[:,4]),extra[ix]])
        u,v=rankdata(t[:,2]),rankdata(t[:,5])
        return float(pearsonr(u-ctrl@np.linalg.lstsq(ctrl,u,rcond=None)[0],v-ctrl@np.linalg.lstsq(ctrl,v,rcond=None)[0]).statistic)
    groupids=np.repeat([list(P.WORDS).index(fr["pos"])*10+P.WORDS[fr["pos"]].index(tuple(fr["words"])) for fr in frames],3)
    groups=np.unique(groupids);rng=np.random.default_rng(505)
    boot=[];diff=[]
    for _ in range(1000):
        ids=np.concatenate([np.flatnonzero(groupids==g) for g in rng.choice(groups,len(groups))])
        boot.append(partial(ids))
        t=z[ids]
        diff.append(float(spearmanr(t[:,2],t[:,5]).statistic-spearmanr(t[:,2],t[:,3]).statistic))
    result={"partial_after_encoders_piece_counts_role":partial(np.arange(len(z))),
            "wordquad_cluster_ci95":np.percentile(boot,[2.5,97.5]).tolist(),
            "decoder_minus_base_rho_difference_ci95":np.percentile(diff,[2.5,97.5]).tolist(),
            "n_word_quadruples":len(groups),"elapsed_seconds":time.monotonic()-START}
    (OUT/"round5_decoder_controls.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    r=json.loads((OUT/"round5_results.json").read_text())
    s=json.loads((OUT/"round5_sensitivity.json").read_text())
    d=json.loads((OUT/"round5_decoder.json").read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold","svg.fonttype":"none"})
    fig,axes=plt.subplots(1,3,figsize=(14,6.2),gridspec_kw={"width_ratios":[1.05,1.2,1]})
    blue,orange,gray="#215C87","#CB6132","#737D85"
    ax=axes[0]
    names=["Matched word edits\n276 pairs / 24 word sets","Natural sentences\n73,536 pairs / 384 sentences"]
    vals=[s["local_base_rho"],r["rsa"]["ring_base"]]
    ax.bar([0,1],vals,color=[blue,blue],width=.55)
    lo,hi=r["ring_base_sentence_bootstrap_ci95"]
    ax.errorbar(1,vals[1],yerr=[[vals[1]-lo],[hi-vals[1]]],fmt="none",color="#19232A",capsize=4)
    for i,v in enumerate(vals):ax.text(i,v+.025,f"{v:.3f}",ha="center",fontweight="bold")
    ax.set(xticks=[0,1],xticklabels=names,ylim=(0,.85),ylabel="Spearman correlation: ring vs base T5")
    ax.set_title("A  Local similarity is not a global map",loc="left",fontsize=11,pad=18)
    ax.text(.02,.95,"Ring error bar: 95% sentence bootstrap CI",transform=ax.transAxes,fontsize=8.5,color=gray,va="top")
    ax=axes[1]
    targets=["Base encoder","Fine-tuned encoder"]
    ring=[r["rsa"]["ring_base"],r["rsa"]["ring_finetuned"]]
    randoms=[r["baselines"]["argmax81"],s["argmax81"]]
    for j in range(2):
        for offset,v,color in [(-.17,ring[j],blue),(.17,randoms[j]["mean_rho"],orange)]:
            ax.bar(j+offset,v,width=.3,color=color)
            ax.text(j+offset,v+.025,f"{v:.3f}",ha="center",fontsize=10,fontweight="bold")
        lo,hi=randoms[j]["seed_range"];v=randoms[j]["mean_rho"]
        ax.errorbar(j+.17,v,yerr=[[v-lo],[hi-v]],fmt="none",color="#19232A",capsize=4)
    ax.set(xticks=[0,1],xticklabels=targets,ylim=(0,.85),ylabel="Spearman correlation on natural sentences")
    ax.set_title("B  Equal nominal capacity, different geometry",loc="left",fontsize=11,pad=18)
    ax.legend(handles=[Line2D([0],[0],lw=8,color=blue,label="Learned ring"),Line2D([0],[0],lw=8,color=orange,label="Random projection + argmax")],loc="upper left",frameon=False,fontsize=9)
    ax.text(.02,-.24,"36 slots × 81 symbols in both codes\nOrange whiskers: range over 20 random seeds",transform=ax.transAxes,fontsize=9,color=gray)
    ax=axes[2]
    for i,(name,key) in enumerate([("Base T5","base"),("Fine-tuned T5","finetuned"),("Ring","ring")]):
        t=r["triplets"][key]
        ratio=t["reversal_mean"]/t["paraphrase_mean"]
        ax.plot([ratio,1],[i,i],color="#B8C2C9",lw=2)
        ax.scatter([ratio],[i],color=orange,s=70,zorder=3)
        ax.scatter([1],[i],color=blue,s=70,zorder=3)
        ax.text(ratio,i-.20,f"{ratio:.2f}",ha="center",fontsize=10,color=orange,fontweight="bold")
    ax.axvline(1,color="#BBC2C7",ls=":",zorder=0)
    ax.set(yticks=range(3),yticklabels=["Base T5","Fine-tuned T5","Ring"],xlim=(0,1.12),ylim=(2.5,-.6),xlabel="Distance relative to paraphrase")
    ax.set_title("C  Base T5 shares the role bias",loc="left",fontsize=11,pad=18)
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="",color=orange,label="Role reversal"),Line2D([0],[0],marker="o",ls="",color=blue,label="Paraphrase = 1")],loc="lower left",frameon=False,fontsize=9)
    ax.text(.02,-.27,"12 triplets · ratio of mean distances\nCosine for T5; Hamming for ring",transform=ax.transAxes,fontsize=9,color=gray)
    for ax in axes:
        ax.grid(axis="y" if ax is not axes[2] else "x",color="#E6E9EC",linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle("A visual message from T5 preserves some local relations, but reshapes its geometry",x=.055,ha="left",fontsize=17,fontweight="bold",y=.99)
    fig.text(.055,.905,"Frozen s2 writer · attention-masked mean pooling (including EOS) · CPU only · exploratory development data",fontsize=10,color=gray)
    fig.subplots_adjust(left=.055,right=.99,top=.79,bottom=.30,wspace=.4)
    fig.text(.055,.045,"Hamming ignores distances between learned mark embeddings: 31.7% of natural pairs differ in all 36 slots.\nThese are representation comparisons, not reconstruction scores; random projection codes have no trained reader.",fontsize=10,color=gray)
    for ext in ["png","svg"]:
        fig.savefig(OUT/f"round5_figure.{ext}",dpi=180,facecolor="white")
    print("saved figure",round(time.monotonic()-START,2))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ["base", "ring"]:
        embeddings(mode)
    elif mode == "stats":
        stats()
    elif mode == "decoder":
        decoder()
    elif mode == "sensitivity":
        sensitivity()
    elif mode == "decoder_controls":
        decoder_controls()
    elif mode == "figure":
        figure()
