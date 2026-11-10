#!/usr/bin/env python3
"""Live view of the locked final exam. http://localhost:8792
Reads `modal run` client logs from $HEPTAPOD_LIVE_LOGS (default: v4/runs/live_logs/).
"""
import ast, http.server, json
import os
from pathlib import Path
S = Path(os.environ.get("HEPTAPOD_LIVE_LOGS", Path(__file__).resolve().parent / "runs" / "live_logs"))
SPLITS = {"locked_natural": "Real sentences (4 new corpora)", "locked_random": "Random combinations", "locked_uniform10": "10 fully random words"}

def data():
    out = []
    for sp, name in SPLITS.items():
        p = S / f"exam_{sp}.log"
        row = {"name": name, "done": 0, "total": {"locked_uniform10": 1000}.get(sp, 2000), "m": None, "finished": False}
        if p.exists():
            for line in p.read_text(errors="ignore").splitlines():
                if line[:1].isdigit() and "/" in line.split(" ")[0]:
                    try:
                        a, rest = line.split(" ", 1)
                        row["done"] = int(a.split("/")[0])
                        row["m"] = {k: round(v * 100, 1) for k, v in ast.literal_eval(rest[:rest.rindex("}") + 1]).items()}
                    except Exception:
                        pass
                if "'event': 'all done'" in line:
                    row["finished"] = True
        out.append(row)
    return out

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Locked exam</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e4e1da;--a:#2f6fb3;--b:#c2581d;--g:#9a958c;--ok:#2e8a5e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1f1e1c;--ink:#ecebe7;--muted:#a19d95;--line:#33312e;--a:#6ea8e6;--b:#ec8a52;--g:#77736b;--ok:#5cc493}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 6px}.sub{color:var(--muted);margin:0 0 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.row{display:grid;grid-template-columns:1fr;gap:6px}.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--a)}
.chart{position:relative;height:360px}.k{color:var(--muted);font-size:13px}</style></head>
<body><main><h1>Locked exam: run once, on sentences nobody has used</h1><p class=sub>Final system: new data + adapted eyes. Exact sentences reading the real image (16-bit PNG). Updates every 10 s.</p>
<div class=card id=prog></div><div class=card><div class=chart><canvas id=c></canvas></div></div></main>
<script>const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C=new Chart(document.getElementById('c'),{type:'bar',data:{labels:[],datasets:[]},options:{animation:false,maintainAspectRatio:false,scales:{y:{min:0,max:100,ticks:{color:css('--muted'),callback:v=>v+'%'},grid:{color:css('--line')}},x:{ticks:{color:css('--ink')},grid:{display:false}}},plugins:{legend:{labels:{color:css('--ink'),usePointStyle:true,boxWidth:8}}}}});
async function tick(){const d=await (await fetch('data.json',{cache:'no-store'})).json();
document.getElementById('prog').innerHTML=d.map(r=>`<div class=row style="margin-bottom:10px"><div><b>${r.name}</b> <span class=k>${r.done} of ${r.total}${r.finished?' · done':''}</span></div><div class=bar><i style="width:${100*r.done/r.total}%"></i></div></div>`).join('');
C.data.labels=d.map(r=>r.name);const ds=(l,k,c)=>({label:l,data:d.map(r=>r.m?r.m[k]:null),backgroundColor:c,borderRadius:4});
C.data.datasets=[ds('plain','plain',css('--g')),ds('reread','search',css('--a')),ds('reread with noise','search_pert',css('--b')),ds('wrong image','shuffled',css('--line'))];C.update();}
tick();setInterval(tick,10000);</script></body></html>"""

class Hd(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b, k = (json.dumps(data()).encode(), "application/json") if self.path.startswith("/data.json") else (PAGE.encode(), "text/html; charset=utf-8")
        self.send_response(200); self.send_header("Content-Type", k); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass

if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", 8792), Hd).serve_forever()
