#!/usr/bin/env python3
"""Live training dashboard for the Modal runs.

Every 2 minutes: pull each run's logs from Modal, parse the progress lines and
rewrite v3/runs/dashboard.html (a self-contained page that reloads itself).

  python3 v3/dashboard.py            # loop until every run has finished
  python3 v3/dashboard.py --once     # one refresh
"""
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PAGE = RUNS / "dashboard.html"
STEPS = {"r4_fresh_reader": 24000, "r3_aug_verbs": 20000, "r2_control": 12000, "r2_aug": 12000, "base": 30000, "narrow": 30000, "long": 80000, "freeze40": 30000, "freeze25": 30000}
LABELS = {"r4_fresh_reader": "run 4 · new reader from scratch", "r3_aug_verbs": "run 3 · more steps + verbs", "r2_control": "reader · original data", "r2_aug": "reader · with augmented sentences", "base": "base · marks at 40%", "narrow": "narrow · marks at 25%", "long": "long · 80,000 steps",
          "freeze40": "freeze40 · frozen, marks at 40%", "freeze25": "freeze25 · frozen, marks at 25%"}


def apps():
    out = {}
    for f in sorted(RUNS.glob("modal_*.launch.log")):
        m = re.search(r"ap-[A-Za-z0-9]+", f.read_text())
        if m:
            out[f.name[len("modal_"):-len(".launch.log")]] = m.group(0)
    return out


def fetch(app):
    try:
        return subprocess.run(["modal", "app", "logs", app], capture_output=True, text=True, timeout=12).stdout
    except subprocess.TimeoutExpired as e:
        return e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")


def parse(text):
    rows, final = [], None
    for line in text.splitlines():
        if line.startswith("{'step'"):
            try:
                rows.append(ast.literal_eval(line))
            except Exception:
                pass
    m = re.search(r'"dev": \{\s*"exact": ([0-9.]+)', text)
    t = re.search(r'"test": \{\s*"exact": ([0-9.]+)', text)
    g = re.search(r'"gen": \{\s*"exact": ([0-9.]+)', text)
    if t:
        final = {"dev": float(m.group(1)) if m else None, "test": float(t.group(1)),
                 "gen": float(g.group(1)) if g else None}
    return rows, final


def snapshot():
    from concurrent.futures import ThreadPoolExecutor
    data = {}
    found = apps()
    with ThreadPoolExecutor(len(found) or 1) as pool:
        texts = dict(zip(found, pool.map(fetch, found.values())))
    cache_file = RUNS / "dashboard_cache.json"
    try:
        cache = json.loads(cache_file.read_text())
    except Exception:
        cache = {}
    for name, app in found.items():
        new_rows, final = parse(texts[name])
        # Modal only returns the latest log lines: accumulate history locally.
        merged = {str(r["step"]): r for r in cache.get(name, {}).get("rows", [])}
        merged.update({str(r["step"]): r for r in new_rows})
        rows = [merged[k] for k in sorted(merged, key=int)]
        final = final or cache.get(name, {}).get("final")
        cache[name] = {"rows": rows, "final": final}
        hard_start = next((r["step"] for r in rows if r.get("hard")), None)
        data[name] = {
            "label": LABELS.get(name, name), "total": STEPS.get(name), "app": app,
            "step": rows[-1]["step"] if rows else 0,
            "hard_start": hard_start or int(0.4 * STEPS.get(name, 30000)),
            "dev": [[r["step"], r["dev_exact"]] for r in rows if "dev_exact" in r],
            "info": [[r["step"], r["loss_right_vs_wrong_logogram"][0], r["loss_right_vs_wrong_logogram"][1]]
                     for r in rows if "loss_right_vs_wrong_logogram" in r],
            "loss": [[r["step"], r["loss"]] for r in rows if "loss" in r],
            "final": final,
        }
    cache_file.write_text(json.dumps(cache))
    return data


def render(data):
    stamp = time.strftime("%H:%M")
    PAGE.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__STAMP__", stamp))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>Heptapod live</title>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--page:#f3f2ef;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e1;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--band:#efeee9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--page:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8d8c86;--grid:#2c2c2a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--band:#232321}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#121211;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8d8c86;--grid:#2c2c2a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--band:#232321}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Inter",system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:22px;margin:0 0 4px;font-weight:650}.sub{color:var(--ink2);margin:0 0 22px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:18px}
.tile{background:var(--surface);border-radius:12px;padding:14px 16px}
.tile .name{display:flex;align-items:center;gap:8px;color:var(--ink2);font-size:13px}
.dot{width:10px;height:10px;border-radius:50%}
.tile .big{font-size:34px;font-weight:650;letter-spacing:-.02em;margin-top:2px}
.tile .meta{color:var(--muted);font-size:13px}
.bar{height:4px;border-radius:2px;background:var(--grid);margin-top:10px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ink2)}
.card{background:var(--surface);border-radius:12px;padding:16px 16px 10px;margin-bottom:14px;position:relative}
.card h2{font-size:15px;margin:0;font-weight:600}.card p{margin:2px 0 8px;color:var(--ink2);font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin-bottom:4px}
.legend span{display:flex;align-items:center;gap:6px}
svg{width:100%;height:auto;display:block;overflow:visible}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--surface);font-size:12px;padding:6px 8px;border-radius:6px;white-space:nowrap;display:none;z-index:2}
details{margin-top:6px;color:var(--ink2);font-size:13px}table{border-collapse:collapse;width:100%;margin-top:6px}td,th{padding:3px 6px;text-align:right;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
.foot{color:var(--muted);font-size:12px}
</style></head><body><main>
<h1>Heptapod Ink v3 · live training</h1>
<p class="sub">Reader retrained on the frozen freeze25 code (starting point: 71% on validation). Refreshes itself every 15 seconds · last data __STAMP__</p>
<div class="tiles" id="tiles"></div>
<div class="card"><h2>Unseen sentences, reconstructed exactly</h2>
<p>Before each color's dotted line the run blends pieces; after it, it has to pick whole pieces.</p>
<div class="legend" id="lg1"></div><svg id="c1" viewBox="0 0 960 320"></svg><div class="tip" id="t1"></div>
<details><summary>Show table</summary><div id="tb1"></div></details></div>
<div class="card"><h2>Training loss (updates live)</h2>
<p>Each point is 100 steps, about 6 seconds of H100. Lower is better.</p>
<div class="legend" id="lg3"></div><svg id="c3" viewBox="0 0 960 240"></svg><div class="tip" id="t3"></div></div>
<div class="card"><h2>How much information the drawing carries</h2>
<p>Loss gap between reading the wrong logogram and the right one (higher = the drawing says more).</p>
<div class="legend" id="lg2"></div><svg id="c2" viewBox="0 0 960 260"></svg><div class="tip" id="t2"></div></div>
<p class="foot">Each run is one H100 GPU on Modal. Mid-training numbers use 512 validation sentences; the final result uses all of them.</p>
</main>
<script>
const D=__DATA__;const ORDER=["r2_control","r2_aug","r3_aug_verbs","r4_fresh_reader"].filter(k=>D[k]);const COL={base:"var(--s1)",narrow:"var(--s2)",long:"var(--s3)",freeze40:"var(--s4)",freeze25:"var(--s1)",r2_control:"var(--s2)",r2_aug:"var(--s3)",r3_aug_verbs:"var(--s4)",r4_fresh_reader:"var(--s5)"};
const pct=v=>(v*100).toFixed(1)+"%";const fmt=n=>n.toLocaleString("en-US");
// tiles
document.getElementById("tiles").innerHTML=ORDER.map(k=>{const r=D[k];const best=r.dev.length?Math.max(...r.dev.map(d=>d[1])):0;
const last=r.dev.length?r.dev[r.dev.length-1][1]:0;const done=!!r.final;const prog=Math.min(1,r.step/r.total);
const phase=done?"done":(r.step>=r.hard_start?"whole pieces":"blending");
return `<div class="tile"><div class="name"><span class="dot" style="background:${COL[k]}"></span>${r.label}</div>
<div class="big">${done?pct(r.final.test):pct(last)}</div>
<div class="meta">${done?`final test · validation ${pct(r.final.dev)} · COGS gen ${pct(r.final.gen)}`:`latest eval · best ${pct(best)}`}</div>
<div class="meta">step ${fmt(r.step)} of ${fmt(r.total)} · ${phase}</div><div class="bar"><i style="width:${prog*100}%"></i></div></div>`}).join("");
function legend(id){document.getElementById(id).innerHTML=ORDER.map(k=>`<span><span class="dot" style="background:${COL[k]}"></span>${D[k].label}</span>`).join("")}
legend("lg1");legend("lg2");legend("lg3");
function chart(svgId,tipId,series,yMax,yFmt,bands){
 const svg=document.getElementById(svgId),W=960,H=+svg.getAttribute("viewBox").split(" ")[3],L=52,R=86,T=12,B=30;
 const xMax=Math.max(...ORDER.map(k=>D[k].total));const x=v=>L+(W-L-R)*v/xMax,y=v=>T+(H-T-B)*(1-v/yMax);
 let s="";
 if(bands)ORDER.forEach(k=>{const hs=D[k].hard_start;s+=`<line x1="${x(hs)}" x2="${x(hs)}" y1="${T}" y2="${H-B}" stroke="${COL[k]}" stroke-width="1.5" stroke-dasharray="3 4" opacity=".8"/>`});
 for(let i=0;i<=4;i++){const v=yMax*i/4;s+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" font-size="12" fill="var(--muted)">${yFmt(v)}</text>`}
 for(let i=0;i<=4;i++){const v=xMax*i/4;s+=`<text x="${x(v)}" y="${H-8}" text-anchor="middle" font-size="12" fill="var(--muted)">${fmt(Math.round(v))}</text>`}
 ORDER.forEach(k=>{const pts=series(k);if(!pts.length)return;
  s+=`<polyline fill="none" stroke="${COL[k]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${pts.map(p=>x(p[0])+","+y(p[1])).join(" ")}"/>`;
  if(pts.length<40)pts.forEach(p=>{s+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="4" fill="${COL[k]}" stroke="var(--surface)" stroke-width="2"/>`});});
 // end labels, nudged apart so they never overlap
 const ends=ORDER.map(k=>{const p=series(k);return p.length?{k,px:x(p[p.length-1][0]),py:y(p[p.length-1][1]),v:p[p.length-1][1]}:null}).filter(Boolean).sort((a,b)=>a.py-b.py);
 for(let i=1;i<ends.length;i++)if(ends[i].py-ends[i-1].py<15)ends[i].py=ends[i-1].py+15;
 ends.forEach(e=>{s+=`<text x="${e.px+8}" y="${e.py+4}" font-size="12" fill="var(--ink2)">${e.k} ${yFmt(e.v)}</text>`});
 s+=`<line id="${svgId}x" y1="${T}" y2="${H-B}" stroke="var(--muted)" stroke-width="1" opacity="0"/><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent" id="${svgId}h"/>`;
 svg.innerHTML=s;const tip=document.getElementById(tipId),hl=document.getElementById(svgId+"x");
 document.getElementById(svgId+"h").addEventListener("mousemove",e=>{const r=svg.getBoundingClientRect();const sx=(e.clientX-r.left)*W/r.width;
  const step=(sx-L)/(W-L-R)*xMax;let lines=[];ORDER.forEach(k=>{const pts=series(k);if(!pts.length)return;
   const p=pts.reduce((a,b)=>Math.abs(b[0]-step)<Math.abs(a[0]-step)?b:a);lines.push(`<b>${k}</b> step ${fmt(p[0])}: ${yFmt(p[1])}`)});
  hl.setAttribute("x1",sx);hl.setAttribute("x2",sx);hl.setAttribute("opacity",".6");tip.innerHTML=lines.join("<br>");tip.style.display="block";
  const card=svg.parentElement.getBoundingClientRect();tip.style.left=Math.min(e.clientX-card.left+12,card.width-200)+"px";tip.style.top=(e.clientY-card.top+12)+"px"});
 document.getElementById(svgId+"h").addEventListener("mouseleave",()=>{tip.style.display="none";hl.setAttribute("opacity","0")});
}
chart("c1","t1",k=>D[k].dev,1,v=>Math.round(v*100)+"%",true);
chart("c3","t3",k=>D[k].loss.filter(p=>p[0]>=100),3,v=>v.toFixed(1),true);
const gap=k=>D[k].info.map(r=>[r[0],Math.max(0,r[2]-r[1])]);
chart("c2","t2",gap,Math.max(1,Math.ceil(Math.max(0,...ORDER.flatMap(k=>gap(k).map(p=>p[1]))))),v=>v.toFixed(1),true);
// table
const steps=[...new Set(ORDER.flatMap(k=>D[k].dev.map(d=>d[0])))].sort((a,b)=>a-b);
document.getElementById("tb1").innerHTML=`<table><tr><th>step</th>${ORDER.map(k=>`<th>${k}</th>`).join("")}</tr>`+steps.map(s=>`<tr><td>${fmt(s)}</td>${ORDER.map(k=>{const d=D[k].dev.find(x=>x[0]===s);return `<td>${d?pct(d[1]):""}</td>`}).join("")}</tr>`).join("")+"</table>";
</script></body></html>"""


if __name__ == "__main__":
    once = "--once" in sys.argv
    while True:
        data = snapshot()
        render(data)
        print(time.strftime("%H:%M"), {k: (v["step"], v["dev"][-1][1] if v["dev"] else None) for k, v in data.items()}, flush=True)
        if once or all(v["final"] for v in data.values()):
            break
        time.sleep(3)
