import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./models/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const $ = (id) => document.getElementById(id);
const S = {};
let cfg, tok, gpu = false;
const P = new URLSearchParams(location.search);
const CHUNK = +(P.get("chunk") || 32);       // candidate codes scored per decoder call
const ROUNDS = +(P.get("rounds") || 20);     // max rereading rounds
const K = +(P.get("k") || 8);               // alternatives tried per slot (nearest first)
const MULTI = +(P.get("multi") || 5);        // up to this many improving moves (distinct slots) accepted per round
const EARLY = (P.get("early") || "1") === "1"; // stop as soon as the ring reads back exactly

// Stage 1 (~64 MB): what is needed to DRAW a ring. Stage 2 (~103 MB, in the background): the reader,
// needed to reread (improve) a ring and to read images.
let readerReady = null;
async function load() {
  cfg = await (await fetch("models/config.json")).json();
  tok = await AutoTokenizer.from_pretrained("tokenizer");
  $("go").textContent = "Loading writer…";
  [S.writer, S.render] = await Promise.all([
    ort.InferenceSession.create("models/writer_alt_mix.onnx", { executionProviders: ["wasm"] }),
    ort.InferenceSession.create("models/render.onnx", { executionProviders: ["wasm"] })]);
  $("go").textContent = "Write";
  $("go").disabled = false;
  window.heptapodWriterReady = true;
  readerReady = loadReader();
}

async function loadReader() {
  $("loading").textContent = "loading the reader in the background (for rereading and reading images)…";
  const ep = (P.get("gpu") || "1") === "1" && navigator.gpu ? ["webgpu"] : ["wasm"];
  const mk = (f, e) => ort.InferenceSession.create(`models/${f}.onnx`, { executionProviders: e });
  [S.eyes, S.memory] = await Promise.all([mk("eyes_q8pc", ["wasm"]), mk("memory", ["wasm"])]);
  try {
    [S.decoder, S.symMemory] = await Promise.all([mk("decoder_fp16", ep), mk("sym_memory", ep)]);
    gpu = ep[0] === "webgpu";
  } catch (e) {
    [S.decoder, S.symMemory] = await Promise.all([mk("decoder_fp16", ["wasm"]), mk("sym_memory", ["wasm"])]);
  }
  $("loading").textContent = "";
  $("go").textContent = "Write and read";
  window.heptapodReady = true;
  window.heptapodGPU = gpu;
}

const i64 = (arr, dims) => new ort.Tensor("int64", BigInt64Array.from(arr, (x) => BigInt(x)), dims);
const marksOf = (sym) => sym.map((s) => cfg.sym_to_mark[s]);

async function writeAlt(text) {
  const enc = tok(text, { padding: "max_length", max_length: cfg.max_tokens, truncation: true });
  const ids = Array.from(enc.input_ids.data, Number), am = Array.from(enc.attention_mask.data, Number);
  const a = (await S.writer.run({ input_ids: i64(ids, [1, cfg.max_tokens]), attention_mask: i64(am, [1, cfg.max_tokens]) })).alt.data;
  const AK = cfg.alt_k, alt = [];
  for (let s = 0; s < cfg.slots; s++) alt.push(Array.from(a.slice(s * AK, s * AK + K), Number));
  return alt;                                                   // [36][K] symbols, nearest first
}

async function renderInk(sym) {
  return (await S.render.run({ marks: i64(marksOf(sym), [1, cfg.slots]) })).ink;
}

function draw(ink, changed = []) {
  const c = $("ring"), g = c.getContext("2d"), img = g.createImageData(cfg.res, cfg.res);
  for (let i = 0; i < ink.data.length; i++) {
    const v = Math.round(255 * (1 - Math.min(1, Math.max(0, ink.data[i]))));
    img.data.set([v, v, v, 255], 4 * i);
  }
  g.putImageData(img, 0, 0);
}

async function readInk(ink) {
  const probs = (await S.eyes.run({ ink })).probs;
  const memory = (await S.memory.run({ probs })).memory;
  const seq = [cfg.decoder_start_token_id];
  for (let t = 0; t < cfg.max_tokens; t++) {
    const ids = i64(seq, [1, seq.length]);
    const logits = (await S.decoder.run({ decoder_input_ids: ids, labels: i64(seq.map(() => 0), [1, seq.length]), memory })).next_logits.data;
    let best = 0;
    for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
    if (best === cfg.eos_token_id) break;
    seq.push(best);
  }
  return tok.decode(seq.slice(1), { skip_special_tokens: true }).trim();
}

async function scoreAll(cands, labels) {
  const out = new Float32Array(cands.length), L = labels.length;
  const exact = new Uint8Array(cands.length);                 // 1 if greedy reading would reproduce every token
  const decIn = [cfg.decoder_start_token_id, ...labels.slice(0, -1)];
  for (let i = 0; i < cands.length; i += CHUNK) {
    const part = cands.slice(i, i + CHUNK), n = part.length;
    const memory = (await S.symMemory.run({ sym: i64(part.flat(), [n, cfg.slots]) })).memory;
    const res = await S.decoder.run({ decoder_input_ids: i64(Array(n).fill(decIn).flat(), [n, L]),
                                      labels: i64(Array(n).fill(labels).flat(), [n, L]), memory });
    const lp = res.label_logprob.data, top = res.label_is_top.data;
    for (let r = 0; r < n; r++) {
      let s = 0, all = 1;
      for (let t = 0; t < L; t++) { s += lp[r * L + t]; if (!top[r * L + t]) all = 0; }
      out[i + r] = -s / L; exact[i + r] = all;
    }
  }
  out.exact = exact;
  return out;
}

const words = (s) => s.toLowerCase().match(/[a-z0-9']+/g) || [];
const same = (a, b) => a.toLowerCase().replace(/\s+/g, " ").trim() === b.toLowerCase().replace(/\s+/g, " ").trim();
// the reader learned that sentences start with a capital: aim the rereading at that form
const target = (t) => t.replace(/^\s*([a-z])/, (m, c) => c.toUpperCase());
function editDist(a, b) {
  const d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0]; d[0] = i;
    for (let j = 1; j <= b.length; j++) { const t = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] !== b[j - 1])); prev = t; }
  }
  return d[b.length];
}

async function reread(text, alt, plainRead) {
  const labels = Array.from(tok(text).input_ids.data, Number);       // includes EOS
  let cur = alt.map((a) => a[0]);
  if (EARLY && same(plainRead, text)) return { sym: cur, moves: 0, rounds: 0, read: plainRead };
  let best = (await scoreAll([cur], labels))[0];
  let moves = 0, rounds = 0, lastRead = null;
  for (let r = 0; r < ROUNDS; r++) {
    rounds++;
    const cands = [], where = [];
    for (let s = 0; s < cfg.slots; s++) for (const sym of alt[s]) if (sym !== cur[s]) { const c = cur.slice(); c[s] = sym; cands.push(c); where.push([s, sym]); }
    const t0 = performance.now();
    const sc = await scoreAll(cands, labels);
    // best improving single moves on distinct slots
    const order = Array.from(sc.keys()).sort((a, b) => sc[a] - sc[b]);
    const picks = [], used = new Set();
    for (const i of order) { if (!(sc[i] < best - 1e-4) || picks.length >= MULTI) break; if (!used.has(where[i][0])) { used.add(where[i][0]); picks.push(i); } }
    if (!picks.length) break;
    let next = cands[picks[0]], nextScore = sc[picks[0]], nMoves = 1, nextExact = sc.exact[picks[0]];
    if (picks.length > 1) {                     // try applying the top moves together (cumulative prefixes)
      const combos = [];
      for (let p = 2; p <= picks.length; p++) { const c = cur.slice(); for (const i of picks.slice(0, p)) c[where[i][0]] = where[i][1]; combos.push(c); }
      const cs = await scoreAll(combos, labels);
      for (let p = 0; p < cs.length; p++) if (cs[p] < nextScore - 1e-4) { next = combos[p]; nextScore = cs[p]; nMoves = p + 2; nextExact = cs.exact[p]; }
    }
    cur = next; best = nextScore; moves += nMoves;
    const ink = await renderInk(cur);
    draw(ink);
    $("stage").textContent = `rereading… round ${r + 1}: ${moves} mark${moves > 1 ? "s" : ""} changed (${Math.round(performance.now() - t0)} ms per round, ${gpu ? "WebGPU" : "CPU"})`;
    // the scorer already tells whether the symbolic reading is exact; confirm through the eyes only then
    if (EARLY && nextExact) { lastRead = await readInk(ink); if (same(lastRead, text)) break; lastRead = null; }
  }
  return { sym: cur, moves, rounds, read: lastRead };
}

function show(el, reading, text, label) {
  const ok = same(reading, text);
  el.innerHTML = `${label}<span class="${ok ? "ok" : "bad"}">${ok ? "✓" : "✗"} ${reading.replace(/</g, "&lt;")}</span>`;
  return ok;
}

async function run() {
  const text = $("text").value.trim();
  if (!text) return;
  $("go").disabled = true; $("read").textContent = ""; $("plain").textContent = ""; $("info").textContent = "";
  const t0 = performance.now();
  const alt = await writeAlt(target(text));          // write the same (capitalised) sentence the rereading aims at
  const plainSym = alt.map((a) => a[0]);
  const plainInk = await renderInk(plainSym);
  draw(plainInk);
  $("stage").textContent = "first attempt, no rereading";
  window.heptapodFirstRing = { text, ms: performance.now() - t0 };
  if (!window.heptapodReady) { $("read").textContent = "waiting for the reader to reread and read…"; await readerReady; }
  const plainRead = await readInk(plainInk);
  show($("read"), plainRead, text, "");
  const t1 = performance.now();
  const { sym, moves, rounds, read } = await reread(target(text), alt, plainRead);
  const ink = await renderInk(sym);
  let reading = read ?? await readInk(ink);
  let finalSym = sym;
  if (editDist(words(text), words(reading)) > editDist(words(text), words(plainRead))) {   // never worse than the first try
    finalSym = plainSym; reading = plainRead; draw(plainInk);
  } else draw(ink);
  const t2 = performance.now();
  $("stage").textContent = `after rereading (${moves} mark${moves === 1 ? "" : "s"} changed)`;
  const ok = show($("read"), reading, text, "");
  show($("plain"), plainRead, text, "without rereading: ");
  $("info").textContent = `write ${Math.round(t1 - t0)} ms · reread ${((t2 - t1) / 1000).toFixed(1)} s on ${gpu ? "WebGPU" : "CPU (no WebGPU)"}`;
  $("go").disabled = false;
  window.heptapodResult = { text, reading, plainRead, ok, moves, rounds, gpu, ms: [t1 - t0, t2 - t1], sym: finalSym };
}

async function readFile(file) {
  if (!window.heptapodReady) { $("read").textContent = "waiting for the reader…"; await readerReady; }
  const bmp = await createImageBitmap(file);
  const c = $("ring"), g = c.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, cfg.res, cfg.res);
  g.drawImage(bmp, 0, 0, cfg.res, cfg.res);
  const px = g.getImageData(0, 0, cfg.res, cfg.res).data, ink = new Float32Array(cfg.res * cfg.res);
  for (let i = 0; i < ink.length; i++) ink[i] = 1 - (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
  $("stage").textContent = "reading the image"; $("read").textContent = "…"; $("plain").textContent = "";
  $("read").textContent = await readInk(new ort.Tensor("float32", ink, [1, cfg.res, cfg.res]));
  $("info").textContent = "read from the image";
}

$("go").onclick = run;
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && !$("go").disabled) run(); });
$("drop").onclick = () => $("file").click();
$("file").onchange = (e) => e.target.files[0] && readFile(e.target.files[0]);
$("drop").ondragover = (e) => e.preventDefault();
$("drop").ondrop = (e) => { e.preventDefault(); e.dataTransfer.files[0] && readFile(e.dataTransfer.files[0]); };
await load();
const q = new URLSearchParams(location.search).get("auto");
if (q) { $("text").value = q; await run(); }
