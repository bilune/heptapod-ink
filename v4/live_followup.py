#!/usr/bin/env python3
"""Live view of the follow-up tests (cliff + fresh reader). http://localhost:8792
Reads `modal run` client logs from $HEPTAPOD_LIVE_LOGS (default: v4/runs/live_logs/).
"""
import ast, http.server, json
import os
from pathlib import Path
L = Path(os.environ.get("HEPTAPOD_LIVE_LOGS", Path(__file__).resolve().parent / "runs" / "live_logs")) / "followup.log"

def data():
    rows = []
    if L.exists():
        for line in L.read_text(errors="ignore").splitlines():
            if line.startswith("{"):
                try: rows.append(ast.literal_eval(line))
                except Exception: pass
    return rows

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Follow-up tests</title>
<style>:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1c1a;--muted:#6b6862;--line:#e4e1da;--a:#2f6fb3}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1f1e1c;--ink:#ecebe7;--muted:#a19d95;--line:#33312e;--a:#6ea8e6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 14px;font-size:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:500}</style></head>
<body><main><h1>Follow-up tests</h1><p class=sub id=st>Updates every 10 s.</p>
<div class=card><h2>1. The cliff: 9 to 12 word sentences, simple vs complex words</h2><p class=sub>Exact sentences. "Forced" = the reader gets each piece right when given the correct previous one.</p><table id=c></table></div>
<div class=card><h2>2. New reader on equal terms (20,000 examples)</h2><p class=sub>Shared table (as before) vs per-slot table (like the original). Exact web sentences.</p><table id=f></table><p class=sub id=fp></p></div></main>
<script>const pc=x=>x==null?'–':(x*100).toFixed(1)+'%';
async function tick(){const r=await (await fetch('data.json',{cache:'no-store'})).json();
const c=r.filter(x=>x.test=='cliff');document.getElementById('c').innerHTML='<tr><th>group</th><th>mean pieces</th><th>no image</th><th>with image</th><th>forced</th><th>reread 20 rounds</th><th>reread 40 rounds</th><th>reread 40, forced</th></tr>'+
c.map(x=>`<tr><td>${x.group=='light'?'simple words':'complex words'}</td><td>${x.mean_tokens}</td><td>${pc(x.symbolic_greedy.exact)}</td><td>${pc(x.visual_greedy.exact)}</td><td>${pc(x.teacher_forced_exact_visual)}</td><td>${pc(x.search20_visual.exact)}</td><td>${pc(x.search40_visual.exact)}</td><td>${pc(x.search40_teacher_forced_visual)}</td></tr>`).join('')||'<tr><td>running…</td></tr>';
const f=r.filter(x=>x.test=='fresh'&&x.plain_rings);document.getElementById('f').innerHTML='<tr><th>new reader</th><th>plain rings</th><th>words</th><th>reread rings</th><th>words</th></tr>'+
f.map(x=>`<tr><td>${x.reader=='shared_table'?'shared table':'per-slot table'}</td><td>${pc(x.plain_rings.exact)}</td><td>${pc(x.plain_rings.word_acc)}</td><td>${pc(x.searched_rings.exact)}</td><td>${pc(x.searched_rings.word_acc)}</td></tr>`).join('')+'<tr><td>original reader</td><td>41.8%</td><td>76.1%</td><td>80.5%</td><td>91.1%</td></tr>';
const tr=r.filter(x=>x.test=='fresh'&&x.step).at(-1);document.getElementById('fp').textContent=tr?`training: ${tr.reader=='shared_table'?'shared table':'per-slot table'}, step ${tr.step} of 6000`:'';
document.getElementById('st').textContent='Updates every 10 s. '+(r.some(x=>x.event=='all done')?'Done.':'Running…');}
tick();setInterval(tick,10000);</script></body></html>"""

class Hd(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b, k = (json.dumps(data()).encode(), "application/json") if self.path.startswith("/data.json") else (PAGE.encode(), "text/html; charset=utf-8")
        self.send_response(200); self.send_header("Content-Type", k); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass

if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", 8792), Hd).serve_forever()
