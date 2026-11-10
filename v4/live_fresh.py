#!/usr/bin/env python3
"""Live view of the fresh-reader experiment only. http://localhost:8792"""
import http.server, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_mix as LM

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>New reader</title><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e4e1da;--a:#2f6fb3;--b:#c2581d;--ok:#2e8a5e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1f1e1c;--ink:#ecebe7;--muted:#a19d95;--line:#33312e;--a:#6ea8e6;--b:#ec8a52;--ok:#5cc493}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 6px}.sub{color:var(--muted);margin:0 0 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.chart{position:relative;height:420px}table{width:100%;border-collapse:collapse;margin-top:14px;font-variant-numeric:tabular-nums}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:500}</style></head>
<body><main><h1>Is it a language? A new reader learns to read the ring</h1>
<p class=sub>Reader from scratch (base T5 and new marks, nothing inherited), trained on N examples. Reads the real image; exact sentences. Dotted: the original reader that invented the code. <b id=st></b></p>
<div class=card><div class=chart><canvas id=c></canvas></div><table id=t></table></div></main>
<script>const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C=new Chart(document.getElementById('c'),{type:'line',data:{datasets:[]},options:{animation:false,maintainAspectRatio:false,parsing:false,
scales:{x:{type:'logarithmic',min:800,max:250000,title:{display:true,text:'training examples',color:css('--muted')},ticks:{color:css('--muted')},grid:{color:css('--line')}},y:{min:0,max:100,ticks:{color:css('--muted'),callback:v=>v+'%'},grid:{color:css('--line')}}},
plugins:{legend:{labels:{color:css('--ink'),usePointStyle:true,boxWidth:8}},tooltip:{callbacks:{label:c=>c.dataset.label+': '+c.parsed.y.toFixed(1)+'%'}}}}});
const L=(l,data,c,dash)=>({label:l,data,borderColor:c,backgroundColor:c,borderWidth:2,pointRadius:dash?0:5,borderDash:dash?[6,5]:[]});
async function tick(){const f=await (await fetch('data.json',{cache:'no-store'})).json();const xs=[800,250000];
C.data.datasets=[L('new · web',f.pts.map(p=>({x:p.size,y:p.web})),css('--a')),L('new · Tatoeba',f.pts.map(p=>({x:p.size,y:p.tat})),css('--b')),L('new · web, reread rings',f.pts.map(p=>({x:p.size,y:p.web_s})),css('--ok'))]
.concat(f.orig?[L('original · web',xs.map(x=>({x,y:f.orig.web})),css('--a'),1),L('original · Tatoeba',xs.map(x=>({x,y:f.orig.tatoeba})),css('--b'),1)]:[]);C.update();
document.getElementById('st').textContent='Status: '+f.status+(f.training&&f.status!='done'?` · training on ${f.training.size.toLocaleString('en')} examples, step ${f.training.step}`:'');
document.getElementById('t').innerHTML='<tr><th>examples</th><th>web</th><th>Tatoeba</th><th>web, reread rings</th></tr>'+(f.orig?`<tr><td>original reader</td><td>${f.orig.web.toFixed(1)}%</td><td>${f.orig.tatoeba.toFixed(1)}%</td><td>–</td></tr>`:'')+f.pts.map(p=>`<tr><td>${p.size.toLocaleString('en')}</td><td>${p.web.toFixed(1)}%</td><td>${p.tat.toFixed(1)}%</td><td>${p.web_s.toFixed(1)}%</td></tr>`).join('');}
tick();setInterval(tick,10000);</script></body></html>"""

class Hd(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b, k = (json.dumps(LM.data()["fresh"]).encode(), "application/json") if self.path.startswith("/data.json") else (PAGE.encode(), "text/html; charset=utf-8")
        self.send_response(200); self.send_header("Content-Type", k); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass

if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", 8792), Hd).serve_forever()
