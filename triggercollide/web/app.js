"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let state = { scan: null };

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function setStatus(text, isError = false) {
  const el = $("#status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

async function runScan(path, body, button) {
  button.disabled = true;
  setStatus("Scanning headers…");
  try {
    body.same_family_only = !$("#all-families").checked;
    const data = await api(path, body);
    state.scan = data;
    render();
    const s = data.summary;
    setStatus(`Scanned ${s.loras} LoRAs in ${s.seconds ?? 0}s.` + (s.errors ? ` ${s.errors} could not be read.` : ""));
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    button.disabled = false;
  }
}

function render() {
  const d = state.scan;
  const s = d.summary;
  const labels = { sd15: "SD 1.5", sd2: "SD 2", sdxl: "SDXL", sd3: "SD 3", flux: "Flux", other: "Other", unknown: "Unknown" };
  const fam = Object.entries(s.families || {}).map(([k, v]) => `${labels[k] || k} ${v}`).join(" · ");
  $("#summary").innerHTML = [
    ["", s.loras, "LoRAs scanned"],
    ["", s.with_triggers, "with trigger words"],
    ["high", s.severity.high, "high clashes"],
    ["medium", s.severity.medium, "medium clashes"],
    ["low", s.severity.low, "low overlaps"],
    ["", s.errors, "unreadable files"],
  ].map(([c, n, l]) => `<div class="card ${c}"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("")
    + `<div class="card"><div class="l">Base models</div><div>${esc(fam || "none")}</div></div>`;
  $("#summary").classList.remove("hidden");
  $("#tabs").classList.remove("hidden");
  $("#dl-csv").href = `/api/scan/${d.scan_id}/collisions.csv`;
  $("#dl-json").href = `/api/scan/${d.scan_id}.json`;
  showTab($(".tab.active").dataset.tab);
  renderHeatmap();
  renderCollisions();
  renderLibrary();
}

function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tab-pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== name || !state.scan));
}

function color(score) {
  if (score <= 0) return "#1b2441";
  const t = Math.min(1, score / 2);
  const a = [51, 214, 194], b = [255, 181, 71], c = [255, 93, 115];
  const mix = (x, y, k) => x.map((v, i) => Math.round(v + (y[i] - v) * k));
  const rgb = t < 0.5 ? mix(a, b, t * 2) : mix(b, c, (t - 0.5) * 2);
  return `rgb(${rgb.join(",")})`;
}

function renderHeatmap() {
  const m = state.scan.matrix;
  const n = m.labels.length;
  if (!n) { $("#heatmap").innerHTML = '<p class="muted">No collisions found. Nice and clean.</p>'; return; }
  const cell = Math.max(10, Math.min(30, Math.floor(560 / n)));
  const pad = 170;
  const short = (t) => (t.length > 24 ? t.slice(0, 23) + "…" : t);
  const size = pad + n * cell + 4;
  let svg = `<svg viewBox="0 0 ${size} ${size}" style="max-width:${size}px" xmlns="http://www.w3.org/2000/svg">`;
  m.labels.forEach((label, i) => {
    svg += `<text x="${pad - 6}" y="${pad + i * cell + cell * 0.7}" fill="#93a0c4" font-size="${Math.min(12, cell * 0.6)}" text-anchor="end">${esc(short(label))}</text>`;
    svg += `<text transform="translate(${pad + i * cell + cell * 0.7},${pad - 6}) rotate(-60)" fill="#93a0c4" font-size="${Math.min(12, cell * 0.6)}">${esc(short(label))}</text>`;
  });
  m.scores.forEach((row, i) => row.forEach((v, j) => {
    svg += `<rect class="hm" data-i="${i}" data-j="${j}" x="${pad + j * cell}" y="${pad + i * cell}" width="${cell - 2}" height="${cell - 2}" rx="3" fill="${i === j ? "#0e1428" : color(v)}"/>`;
  }));
  svg += "</svg>";
  $("#heatmap").innerHTML = svg;
  $$("#heatmap rect.hm").forEach((r) => {
    r.addEventListener("mousemove", (e) => {
      const i = +r.dataset.i, j = +r.dataset.j;
      if (i === j) return hideTip();
      const v = m.scores[i][j];
      tip(e, `<b>${esc(m.labels[i])}</b> × <b>${esc(m.labels[j])}</b><br>${v ? "score " + v : "no overlap"}`);
    });
    r.addEventListener("mouseleave", hideTip);
  });
}

function tip(e, html) {
  const t = $("#tip");
  t.innerHTML = html;
  t.style.left = e.clientX + 14 + "px";
  t.style.top = e.clientY + 14 + "px";
  t.classList.remove("hidden");
}
function hideTip() { $("#tip").classList.add("hidden"); }

function renderCollisions() {
  const q = $("#col-filter").value.trim().toLowerCase();
  const rows = state.scan.collisions.filter((c) => !q || JSON.stringify([c.a_name, c.b_name, c.findings]).toLowerCase().includes(q));
  $("#col-list").innerHTML = rows.slice(0, 300).map((c) => `
    <li class="${c.severity}"><div><span class="sev ${c.severity}">${c.severity}</span><span class="pair">${esc(c.a_name)} ↔ ${esc(c.b_name)}</span>
    <span class="muted"> · ${c.score}</span></div>
    ${c.findings.map((f) => `<div class="why">${esc(f.detail)}</div>`).join("")}
    ${c.shared_tags.length ? `<div class="why">Shared tags: ${c.shared_tags.slice(0, 8).map(esc).join(", ")}</div>` : ""}</li>`).join("")
    || '<li class="muted">No collisions match.</li>';
}

function renderLibrary() {
  const q = $("#lib-filter").value.trim().toLowerCase();
  const rows = state.scan.library.filter((r) => !q || JSON.stringify([r.name, r.filename, r.family_label, r.triggers]).toLowerCase().includes(q));
  $("#lib-table tbody").innerHTML = rows.map((r) => `<tr>
    <td><div>${esc(r.name)}</div><div class="muted">${esc(r.filename)}${r.trained_name && r.trained_name !== r.name ? " · trained as " + esc(r.trained_name) : ""}</div>${r.error ? `<div class="err">${esc(r.error)}</div>` : ""}</td>
    <td><span class="fam ${r.family}">${esc(r.family_label)}</span><div class="muted">${esc(r.family_source)}</div></td>
    <td>${r.triggers.map((t) => `<span class="chip trig" title="${esc(t.source)}">${esc(t.text)}</span>`).join("") || '<span class="muted">none found</span>'}</td>
    <td>${r.top_tags.slice(0, 6).map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</td>
    <td>${r.collisions}</td></tr>`).join("");
}

async function checkWorkflow() {
  const out = $("#wf-result");
  let wf;
  try { wf = JSON.parse($("#wf-text").value); } catch (e) { out.innerHTML = `<p class="err">That is not valid JSON: ${esc(e.message)}</p>`; return; }
  try {
    const r = await api("/api/workflow/check", { scan_id: state.scan.scan_id, workflow: wf });
    out.innerHTML = `<h2>${r.stacked.length} LoRAs stacked · total weight ${r.total_weight}</h2>
      <div class="stack">${r.stacked.map((s) => `<span class="chip ${s.enabled ? "trig" : ""}">${esc(s.lora)} @ ${s.strength_model}${s.enabled ? "" : " (off)"}</span>`).join("")}</div>
      ${r.issues.map((i) => `<div class="issue ${i.level}"><span class="sev ${i.level}">${i.level}</span>${esc(i.message)}</div>`).join("") || '<div class="issue">No problems found.</div>'}
      ${r.missing_triggers.map((m) => `<div class="issue low"><span class="sev low">hint</span>${esc(m.lora)}: the prompt has none of ${m.triggers.map(esc).join(", ")}</div>`).join("")}`;
  } catch (err) {
    out.innerHTML = `<p class="err">${esc(err.message)}</p>`;
  }
}

function init() {
  $$(".seg-btn").forEach((b) => b.addEventListener("click", () => {
    $$(".seg-btn").forEach((x) => x.classList.toggle("active", x === b));
    $$(".src-pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== b.dataset.src));
  }));
  $$(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  $("#scan-demo").addEventListener("click", (e) => runScan("/api/scan/demo", {}, e.target));
  $("#scan-folder").addEventListener("click", (e) => runScan("/api/scan/folder", { path: $("#folder-path").value, recursive: $("#folder-recursive").checked }, e.target));
  $("#scan-comfy").addEventListener("click", (e) => runScan("/api/scan/comfy", { url: $("#comfy-url").value }, e.target));
  $("#col-filter").addEventListener("input", renderCollisions);
  $("#lib-filter").addEventListener("input", renderLibrary);
  $("#wf-check").addEventListener("click", checkWorkflow);
  $("#wf-demo").addEventListener("click", async () => { $("#wf-text").value = JSON.stringify(await api("/api/demo/workflow"), null, 2); });
  $("#wf-file").addEventListener("change", async (e) => { const f = e.target.files[0]; if (f) $("#wf-text").value = await f.text(); });
  api("/api/meta").then((m) => {
    $("#version").textContent = "v" + m.version;
    if (m.comfy_url) $("#comfy-url").value = m.comfy_url;
    if (m.lora_folder) $("#folder-path").value = m.lora_folder;
  }).catch(() => {});
  const params = new URLSearchParams(location.search);
  if (params.get("tab")) showTab(params.get("tab"));
  if (params.get("demo") === "1") {
    runScan("/api/scan/demo", {}, $("#scan-demo")).then(async () => {
      if (params.get("tab")) showTab(params.get("tab"));
      if (params.get("tab") === "workflow") { $("#wf-demo").click(); setTimeout(checkWorkflow, 300); }
    });
  }
}
init();
