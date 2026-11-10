#!/usr/bin/env python3
"""Live view of the data-mix experiment (mixed vs control arms). http://localhost:8792
Reads `modal run` client logs from $HEPTAPOD_LIVE_LOGS (default: v4/runs/live_logs/).
"""
import ast, http.server, json
import os
from pathlib import Path
S = Path(os.environ.get("HEPTAPOD_LIVE_LOGS", Path(__file__).resolve().parent / "runs" / "live_logs"))
ARMS = {"New mix": S / "arm_mixed.log", "Control (Tatoeba only)": S / "arm_control.log"}
BASE, GATE = 16.4, 26.4

def rows(p):
    out = []
    if p.exists():
        for line in p.read_text(errors="ignore").splitlines():
            if line.startswith("{"):
                try: out.append(ast.literal_eval(line))
                except Exception: pass
    return out

def data():
    d = {}
    for name, p in ARMS.items():
        rs = rows(p)
        pts = [{"step": r["step"], "sec": r["sec"], "ewt": r["sym_fresh_natural"]["exact_cased"] * 100,
                "ewt_words": r["sym_fresh_natural"]["word_acc"] * 100, "tat": r["sym_dev"]["exact_cased"] * 100}
               for r in rs if "sym_fresh_natural" in r]
        last = next((r for r in reversed(rs) if "step" in r), {})
        txt = p.read_text(errors="ignore") if p.exists() else ""
        d[name] = {"pts": pts, "step": last.get("step"), "sec": last.get("sec"),
                   "done": "exit code" in txt or "done (continued)" in txt}
    eyes = [{"step": r["step"], "ewt": r["vis_fresh_natural"]["exact_cased"] * 100, "words": r["vis_fresh_natural"]["word_acc"] * 100,
             "slot": r["vis_fresh_natural"].get("slot_acc", 0) * 100, "tat": r["vis_dev"]["exact_cased"] * 100}
            for r in rows(S / "s2eyes.log") if "vis_fresh_natural" in r]
    panels, cur = [], None
    txt = (S / "s2eyes.log").read_text(errors="ignore") if (S / "s2eyes.log").exists() else ""
    for line in txt.splitlines():
        if "'cmd'" in line and "search.py" in line:
            cur = ("web (A)" if "fresh_natural --panel A" in line else "web, confirmation (B)" if "--panel B" in line else "Tatoeba")
        elif cur and line[:1].isdigit() and "/" in line.split(" ")[0]:
            try:
                done, rest = line.split(" ", 1)
                m = ast.literal_eval(rest[:rest.rindex("}") + 1])
                panels = [p for p in panels if p["name"] != cur] + [{"name": cur, "done": done, **{k: round(v * 100, 1) for k, v in m.items()}}]
            except Exception:
                pass
    fresh = {"orig": None, "pts": [], "training": None}
    for r in rows(S / "fresh.log"):
        if r.get("event") == "original reader":
            fresh["orig"] = {k: r[k]["exact"] * 100 for k in ("web", "tatoeba")}
        elif "web" in r and "size" in r:
            fresh["pts"].append({"size": r["size"], "web": r["web"]["exact"] * 100, "tat": r["tatoeba"]["exact"] * 100,
                                 "web_s": r.get("web_searched_rings", {}).get("exact", 0) * 100})
        elif "size" in r and "step" in r:
            fresh["training"] = {"size": r["size"], "step": r["step"], "sec": r["sec"]}
    fl = S / "fresh.log"
    fresh["status"] = ("done" if fl.exists() and "'event': 'all done'" in fl.read_text(errors="ignore") else
                       "training readers" if fresh["orig"] else
                       "preparing: the writer rereads the 256 test sentences (about 5 min)" if fl.exists() else "not started")
    prog = json.loads((Path(__file__).resolve().parent / "progress.json").read_text())
    now = {p["name"]: p for p in panels}
    if prog and "web (A)" in now:  # live values for the current milestone while it is being measured
        prog[-1]["web_plain"], prog[-1]["web_reread"] = now["web (A)"]["plain"], now["web (A)"]["search"]
    if prog and "Tatoeba" in now:
        prog[-1]["tat_plain"], prog[-1]["tat_reread"] = now["Tatoeba"]["plain"], now["Tatoeba"]["search"]
    return {"arms": d, "base": BASE, "gate": GATE, "eyes": eyes, "panels": panels, "progress": prog, "fresh": fresh}

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Heptapod new data</title><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e4e1da;--a:#2f6fb3;--b:#c2581d;--g:#9a958c;--ok:#2e8a5e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1f1e1c;--ink:#ecebe7;--muted:#a19d95;--line:#33312e;--a:#6ea8e6;--b:#ec8a52;--g:#77736b;--ok:#5cc493}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 18px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}.k{color:var(--muted);font-size:13px}.v{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}.ok{color:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}.card h2{font-size:16px;margin:0 0 2px}.card p{color:var(--muted);margin:0 0 10px;font-size:13px}.chart{position:relative;height:300px}</style></head>
<body><main><h1>New data vs control</h1><p class=sub id=sub>loading…</p><div class=tiles id=tiles></div><div class=card><h2>Is it a language? A new reader learns to read the ring</h2><p id=fsub>Reader from scratch (base T5, new marks), trained on N examples; reads the real image. Dotted line: the original reader.</p><div class=chart style="height:320px"><canvas id=fresh></canvas></div></div>
<div class=card><h2>Project progress</h2><p>Exact sentences reading the real image, at each milestone. Web = real sentences from the internet (what we are trying to improve). Dotted line: Codex's "surprising" target for web with rereading (95%).</p><div class=chart style="height:360px"><canvas id=prog></canvas></div></div>
<div class=card><h2>What matters: exact sentences reading the real image</h2><p>This morning (v4 model) vs now (new data). Plain writer, rereading writer, and rereading with image noise. 256 sentences per panel.</p><div class=chart style="height:340px"><canvas id=bars></canvas></div></div>

<div class=card><h2>Eyes adapting to the new code (reading the image)</h2><p>Exact web sentences, case-sensitive, and recognized marks.</p><div class=chart><canvas id=eyes></canvas></div></div>
<div class=card><h2>With rereading, reading the real image</h2><p>256 sentences per panel. Exact (case-insensitive), plain writer vs rereading writer.</p><div id=panels>waiting…</div></div>
<div class=card><h2>Web sentences (EWT), exact, case-sensitive</h2><p>No image, 256 sentences. Gray dotted: current model (16.4%). Green dotted: gate to continue (26.4%).</p><div class=chart><canvas id=ewt></canvas></div></div>
<div class=card><h2>Common sentences (Tatoeba), exact</h2><p>To check that what already worked does not get worse.</p><div class=chart><canvas id=tat></canvas></div></div></main>
<script>const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function mk(id){return new Chart(document.getElementById(id),{type:'line',data:{datasets:[]},options:{animation:false,maintainAspectRatio:false,parsing:false,
scales:{x:{type:'linear',title:{display:true,text:'step',color:css('--muted')},ticks:{color:css('--muted')},grid:{color:css('--line')}},y:{ticks:{color:css('--muted'),callback:v=>v+'%'},grid:{color:css('--line')}}},
plugins:{legend:{labels:{color:css('--ink'),usePointStyle:true,boxWidth:8}},tooltip:{callbacks:{label:c=>c.dataset.label+': '+c.parsed.y.toFixed(1)+'%'}}}}})}
const C={ewt:mk('ewt'),tat:mk('tat'),eyes:mk('eyes')};
const FR=new Chart(document.getElementById('fresh'),{type:'line',data:{datasets:[]},options:{animation:false,maintainAspectRatio:false,parsing:false,
scales:{x:{type:'logarithmic',title:{display:true,text:'training examples',color:css('--muted')},ticks:{color:css('--muted')},grid:{color:css('--line')}},y:{min:0,max:100,ticks:{color:css('--muted'),callback:v=>v+'%'},grid:{color:css('--line')}}},
plugins:{legend:{labels:{color:css('--ink'),usePointStyle:true,boxWidth:8}}}}});
const P=new Chart(document.getElementById('prog'),{type:'line',data:{labels:[],datasets:[]},options:{animation:false,maintainAspectRatio:false,spanGaps:true,
scales:{y:{min:0,max:100,ticks:{color:css('--muted'),callback:v=>v+'%'},grid:{color:css('--line')}},x:{ticks:{color:css('--ink')},grid:{display:false}}},
plugins:{legend:{labels:{color:css('--ink'),usePointStyle:true,boxWidth:8}},tooltip:{callbacks:{label:c=>c.dataset.label+': '+(c.parsed.y==null?'–':c.parsed.y.toFixed(1)+'%')}}}}});
const MORNING={'web (A)':{plain:20.7,search:66.8,search_pert:61.7},'web, confirmation (B)':{plain:20.7,search:66.8,search_pert:61.7},'Tatoeba':{plain:68.8,search:96.5,search_pert:94.5}};
const B=new Chart(document.getElementById('bars'),{type:'bar',data:{labels:[],datasets:[]},options:{animation:false,maintainAspectRatio:false,
scales:{y:{min:0,max:100,ticks:{color:css('--muted'),callback:v=>v+'%'},grid:{color:css('--line')}},x:{ticks:{color:css('--ink')},grid:{display:false}}},
plugins:{legend:{labels:{color:css('--ink'),usePointStyle:true,boxWidth:8}},tooltip:{callbacks:{label:c=>c.dataset.label+': '+(c.parsed.y==null?'–':c.parsed.y.toFixed(1)+'%')}}}}});
const L=(label,data,color,dash)=>({label,data,borderColor:color,backgroundColor:color,borderWidth:2,pointRadius:dash?0:3,borderDash:dash?[5,5]:[]});
async function tick(){const d=await (await fetch('data.json',{cache:'no-store'})).json();const names=Object.keys(d.arms);const cols=[css('--a'),css('--b')];
let maxs=1000;names.forEach(n=>d.arms[n].pts.forEach(p=>maxs=Math.max(maxs,p.step)));
C.ewt.data.datasets=names.map((n,i)=>L(n,d.arms[n].pts.map(p=>({x:p.step,y:p.ewt})),cols[i])).concat([L('current',[{x:0,y:d.base},{x:maxs,y:d.base}],css('--g'),1),L('gate',[{x:0,y:d.gate},{x:maxs,y:d.gate}],css('--ok'),1)]);
C.tat.data.datasets=names.map((n,i)=>L(n,d.arms[n].pts.map(p=>({x:p.step,y:p.tat})),cols[i]));C.ewt.update();C.tat.update();
{const f=d.fresh;const xs=[1000,200000];const L2=(l,data,c,dash)=>({label:l,data,borderColor:c,backgroundColor:c,borderWidth:2,pointRadius:dash?0:4,borderDash:dash?[5,5]:[]});
 FR.data.datasets=[L2('new · web',f.pts.map(p=>({x:p.size,y:p.web})),css('--a')),L2('new · Tatoeba',f.pts.map(p=>({x:p.size,y:p.tat})),css('--b')),L2('new · web, reread rings',f.pts.map(p=>({x:p.size,y:p.web_s})),css('--ok'))]
  .concat(f.orig?[L2('original · web',xs.map(x=>({x,y:f.orig.web})),css('--a'),1),L2('original · Tatoeba',xs.map(x=>({x,y:f.orig.tatoeba})),css('--b'),1)]:[]);FR.update();
 document.getElementById('fsub').textContent='Reader from scratch (base T5, new marks), trained on N examples; reads the real image. Dotted: the original reader.'+` · Status: ${f.status}`+(f.training?` · training on ${f.training.size.toLocaleString()} examples, step ${f.training.step}`:'');}
{const pr=d.progress;P.data.labels=pr.map(p=>p.label);const ln=(l,k,c,dash,w)=>({label:l,data:pr.map(p=>p[k]),borderColor:c,backgroundColor:c,borderWidth:w||2,pointRadius:4,borderDash:dash?[4,4]:[]});
 P.data.datasets=[ln('Web · reread','web_reread',css('--a'),0,3),ln('Web · plain','web_plain',css('--a'),1),ln('Tatoeba · reread','tat_reread',css('--b'),0,3),ln('Tatoeba · plain','tat_plain',css('--b'),1),
 {label:'web target (95%)',data:pr.map(()=>95),borderColor:css('--ok'),borderWidth:1,pointRadius:0,borderDash:[2,4]}];P.update();}
{const names=['web (A)','web, confirmation (B)','Tatoeba'];const now=Object.fromEntries(d.panels.map(p=>[p.name,p]));
 const labs=[];const keys=[['plain','plain'],['search','reread'],['search_pert','reread with noise']];
 names.forEach(n=>keys.forEach(([k,l])=>labs.push(n.replace(', confirmation',' confirm.')+' · '+l)));
 B.data.labels=labs;
 B.data.datasets=[{label:'this morning (v4)',data:names.flatMap(n=>keys.map(([k])=>MORNING[n][k])),backgroundColor:css('--g'),borderRadius:4},
  {label:'now',data:names.flatMap(n=>keys.map(([k])=>now[n]?now[n][k]:null)),backgroundColor:css('--a'),borderRadius:4}];B.update();}
C.eyes.data.datasets=[L('web exact',d.eyes.map(p=>({x:p.step,y:p.ewt})),cols[0]),L('recognized marks',d.eyes.map(p=>({x:p.step,y:p.slot})),css('--ok'))];C.eyes.update();
document.getElementById('panels').innerHTML=d.panels.length?'<table style="width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums">'+'<tr style="color:var(--muted);text-align:left"><th>panel</th><th>progress</th><th>plain</th><th>reread</th><th>reread with noise</th><th>wrong image</th></tr>'+d.panels.map(p=>`<tr><td>${p.name}</td><td>${p.done}</td><td>${p.plain}%</td><td><b>${p.search}%</b></td><td>${p.search_pert}%</td><td>${p.shuffled}%</td></tr>`).join('')+'</table>':'waiting for the eyes to finish…';
let t='';names.forEach(n=>{const a=d.arms[n],p=a.pts.at(-1);t+=`<div class=tile><div class=k>${n}${a.done?' · done':''}</div><div class="v ${p&&p.ewt>=d.gate?'ok':''}">${p?p.ewt.toFixed(1)+'%':'–'}</div><div class=k>web exact · step ${a.step??'–'} · ${a.sec?Math.round(a.sec/60)+' min':''}</div></div>`});
const e=d.eyes.at(-1);t=`<div class=tile style="border-color:var(--a)"><div class=k>RUNNING · eyes reading the image</div><div class=v>${e?e.ewt.toFixed(1)+'%':'–'}</div><div class=k>web exact · step ${e?e.step:'–'} · marks ${e?e.slot.toFixed(1)+'%':'–'}</div></div>`+t;
document.getElementById('tiles').innerHTML=t;document.getElementById('sub').textContent='Updates every 10 s';}
tick();setInterval(tick,10000);</script></body></html>"""

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b, k = (json.dumps(data()).encode(), "application/json") if self.path.startswith("/data.json") else (PAGE.encode(), "text/html; charset=utf-8")
        self.send_response(200); self.send_header("Content-Type", k); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass

if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", 8792), H).serve_forever()
