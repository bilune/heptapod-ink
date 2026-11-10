#!/usr/bin/env python3
"""Live view of the v4 Modal runs, read from the local launch logs.

Serves a page on http://localhost:8791 that refreshes every 10 s.
The logs are the `modal run` client output (`modal run ... | tee <dir>/v4_vq.log`),
read from $HEPTAPOD_LIVE_LOGS (default: v4/runs/live_logs/).

  python3 v4/live.py
"""
import ast
import http.server
import json
import sys
import os
from pathlib import Path

SCRATCH = Path(os.environ.get("HEPTAPOD_LIVE_LOGS", Path(__file__).resolve().parent / "runs" / "live_logs"))
FIRST = SCRATCH / "v4_vq.log"          # warm-up + first VQ stretch (stopped at step ~15000)
CONT = SCRATCH / "v4_vq_ent2.log"      # continuation from the step-13000 checkpoint, then eyes and final eval
RESUMED_AT = 13000
S2_SEC = 2400
PORT = 8791


def rows(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("{"):
            try:
                out.append(ast.literal_eval(line))
            except Exception:
                pass
    return out


def pick(r, key):
    m = r.get(key)
    return None if m is None else {"exact": m["exact"], "words": m["word_acc"], "slot": m.get("slot_acc")}


def data():
    paper, eyes, events, last = [], [], [], None
    for src, offset in ((FIRST, 0), (CONT, RESUMED_AT)):
        for r in rows(src):
            if r.get("stage") == 1 and "step" in r:
                step = r["step"] + offset
                if src == FIRST and step > RESUMED_AT:
                    continue  # the continuation restarted from this point
                if "cont_dev" in r and "sym_dev" not in r:
                    paper.append({"step": step, "kind": "continuous", "nat": pick(r, "cont_dev"), "rand": pick(r, "cont_dev_random")})
                elif "sym_dev" in r:
                    paper.append({"step": step, "kind": "marks", "nat": pick(r, "sym_dev"), "rand": pick(r, "sym_dev_random")})
            if src == CONT and r.get("stage") == 2 and "vis_dev" in r:
                eyes.append({"step": r["step"], "nat": pick(r, "vis_dev"), "rand": pick(r, "vis_dev_random")})
            if src == CONT and "event" in r and r["event"] != "start":
                events.append({k: v for k, v in r.items() if k in ("stage", "step", "event")})
            if src == CONT:
                last = r
    text = CONT.read_text(errors="ignore") if CONT.exists() else ""
    stage = (last or {}).get("stage")
    done = "exit code" in text
    report = next((r for r in reversed(rows(CONT)) if r.get("event") == "report"), None)
    status = "done" if done else "final evaluation" if any(e.get("event") == "done" and e.get("stage") == 2 for e in events) \
        else "eyes reading the code" if stage == 2 else "on paper" if stage == 1 else "starting"
    progress = None
    if stage == 2 and last and "sec" in last:
        progress = min(1.0, last["sec"] / S2_SEC)
    return {"paper": paper, "eyes": eyes, "events": events, "status": status, "progress": progress,
            "last": {k: last.get(k) for k in ("stage", "step", "loss", "slot_acc", "sec")} if last else None,
            "report": report}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heptapod live</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e4e1da;--a:#2f6fb3;--b:#c2581d;--c:#2e8a5e;--goal:#9a958c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1f1e1c;--ink:#ecebe7;--muted:#a19d95;--line:#33312e;--a:#6ea8e6;--b:#ec8a52;--c:#5cc493;--goal:#77736b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .k{color:var(--muted);font-size:13px}.tile .v{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums}.tile .g{color:var(--muted);font-size:12px}
.ok{color:var(--c)}.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:8px}.bar i{display:block;height:100%;background:var(--a)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
.card h2{font-size:16px;margin:0 0 2px}.card p{color:var(--muted);margin:0 0 10px;font-size:13px}.chart{position:relative;height:280px}
</style></head><body><main>
<h1>Heptapod v4 · live</h1><p class="sub" id="sub">loading…</p>
<div class="tiles" id="tiles"></div>
<div class="card"><h2>Reading the image (eyes)</h2><p>Over 512 sentences from each set. Dotted lines: target goals.</p><div class="chart"><canvas id="eyes"></canvas></div></div>
<div class="card"><h2>On paper (no image)</h2><p>Up to step 4,000 the channel is continuous; after that each slot uses one of 81 marks.</p><div class="chart"><canvas id="paper"></canvas></div></div>
</main><script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct=x=>x==null?'–':Math.round(x*100)+'%';
function series(rows,get){return rows.map(r=>({x:r.step,y:get(r)==null?null:get(r)*100}))}
function goal(y,rows){if(!rows.length)return[];const xs=rows.map(r=>r.step);return[{x:Math.min(...xs),y},{x:Math.max(...xs),y}]}
function make(id){return new Chart(document.getElementById(id),{type:'line',data:{datasets:[]},options:{animation:false,maintainAspectRatio:false,parsing:false,
 interaction:{mode:'nearest',axis:'x',intersect:false},scales:{x:{type:'linear',title:{display:true,text:'step',color:css('--muted')},ticks:{color:css('--muted')},grid:{color:css('--line')}},
 y:{min:0,max:100,ticks:{color:css('--muted'),callback:v=>v+'%'},grid:{color:css('--line')}}},
 plugins:{legend:{labels:{color:css('--ink'),usePointStyle:true,boxWidth:8}},tooltip:{callbacks:{label:c=>c.dataset.label+': '+(c.parsed.y==null?'–':c.parsed.y.toFixed(1)+'%')}}}}})}
const charts={eyes:make('eyes'),paper:make('paper')};
function sets(rows){const a=css('--a'),b=css('--b'),c=css('--c'),g=css('--goal');const L=(label,data,color,dash)=>({label,data,borderColor:color,backgroundColor:color,borderWidth:2,pointRadius:dash?0:3,borderDash:dash?[5,5]:[],spanGaps:true});
 return[L('Normal · exact sentences',series(rows,r=>r.nat&&r.nat.exact),a),L('Normal · words',series(rows,r=>r.nat&&r.nat.words),c),L('Random · words',series(rows,r=>r.rand&&r.rand.words),b),
 L('target exact 30%',goal(30,rows),a,1),L('target words 60%',goal(60,rows),c,1),L('target random 30%',goal(30,rows),b,1)]}
function tile(k,v,g,ok){return `<div class="tile"><div class="k">${k}</div><div class="v ${ok?'ok':''}">${v}</div><div class="g">${g}</div></div>`}
async function tick(){try{const d=await (await fetch('data.json',{cache:'no-store'})).json();
 const e=d.eyes.at(-1),p=d.paper.at(-1),src=e||p;
 document.getElementById('sub').textContent=`Stage: ${d.status}`+(d.last?` · step ${d.last.step??'–'}`:'')+` · updates every 10 s`;
 let t='';if(src){t+=tile('Normal · exact',pct(src.nat.exact),'target 30%',src.nat.exact>=.3);t+=tile('Normal · words',pct(src.nat.words),'target 60%',src.nat.words>=.6);
 t+=tile('Random · words',pct(src.rand.words),'target 30%',src.rand.words>=.3);if(e)t+=tile('Recognized marks',pct(e.nat.slot),'eyes, normal sentences',false)}
 if(d.progress!=null)t+=`<div class="tile"><div class="k">Eye training</div><div class="v">${Math.round(d.progress*100)}%</div><div class="bar"><i style="width:${d.progress*100}%"></i></div></div>`;
 document.getElementById('tiles').innerHTML=t;
 charts.eyes.data.datasets=sets(d.eyes);charts.eyes.update();charts.paper.data.datasets=sets(d.paper);charts.paper.update()}catch(err){document.getElementById('sub').textContent='no data: '+err}}
tick();setInterval(tick,10000);
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/data.json"):
            body, kind = json.dumps(data()).encode(), "application/json"
        else:
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps(data(), indent=1)[:2000])
    else:
        print(f"http://localhost:{PORT}")
        http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
