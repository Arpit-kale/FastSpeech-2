/* FastSpeech 2 Studio — front-end logic (vanilla JS + Chart.js) */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const COLORS = { duration: "#2a78d6", pitch: "#eb6834", energy: "#1baf7a" };
  const state = { lang: "english", models: [], sentences: {}, result: null, variable: "duration", chart: null, history: [] };

  // ------------------------------------------------------------ helpers
  const api = async (path, opts = {}) => {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
    return body;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const showError = (msg) => { const e = $("#error"); e.textContent = msg; e.hidden = !msg; };

  // ------------------------------------------------------------ navigation
  $$(".tab").forEach((b) => b.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t === b));
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${b.dataset.view}`));
    if (b.dataset.view === "results") loadResults(state.resultsLang || "english");
  }));

  // ------------------------------------------------------------ models
  async function refreshModels() {
    try {
      state.models = await api("/api/models");
    } catch { state.models = []; }
    $("#model-status").innerHTML = state.models.map((m) => {
      const cls = m.loaded ? "ok" : m.available ? "warn" : "";
      const label = m.loaded ? `step ${m.step ?? "?"}` : m.available ? "ready" : "not trained";
      return `<span class="pill ${cls}" title="${esc(m.checkpoint || "no checkpoint")}"><span class="dot"></span>${esc(m.label)} · ${label}</span>`;
    }).join("");
    fillSpeakers();
  }

  function fillSpeakers() {
    const m = state.models.find((x) => x.language === state.lang);
    const sel = $("#speaker");
    const current = sel.value;
    sel.innerHTML = `<option value="">Default${m?.default_speaker ? ` (${esc(m.default_speaker)})` : ""}</option>` +
      (m?.speakers || []).map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
    sel.value = current;
  }

  // ------------------------------------------------------------ language + samples
  function setLanguage(lang) {
    state.lang = lang;
    $$("#lang-switch button").forEach((b) => b.classList.toggle("on", b.dataset.lang === lang));
    const ta = $("#text");
    ta.lang = lang === "telugu" ? "te" : "en";
    ta.placeholder = lang === "telugu" ? "తెలుగులో వాక్యం టైప్ చేయండి…" : "Type an English sentence…";
    renderSamples();
    fillSpeakers();
  }
  $$("#lang-switch button").forEach((b) => b.addEventListener("click", () => setLanguage(b.dataset.lang)));

  function renderSamples() {
    const list = state.sentences[state.lang] || [];
    $("#samples").innerHTML = list.map((s, i) =>
      `<button class="chip" data-i="${i}" title="${esc(s.focus)}"><b>${esc(s.id)}</b>${esc(s.text.slice(0, 34))}${s.text.length > 34 ? "…" : ""}</button>`).join("");
    $$("#samples .chip").forEach((c) => c.addEventListener("click", () => {
      $("#text").value = list[+c.dataset.i].text;
      updateCount();
    }));
  }

  const updateCount = () => { $("#char-count").textContent = `${$("#text").value.length} / 600`; };
  $("#text").addEventListener("input", updateCount);

  // ------------------------------------------------------------ sliders
  const sliders = [["dur", "dur-out"], ["pitch", "pitch-out"], ["energy", "energy-out"]];
  sliders.forEach(([id, out]) => {
    const el = $(`#${id}`);
    const upd = () => { $(`#${out}`).textContent = `${(+el.value).toFixed(2)}×`; };
    el.addEventListener("input", upd); upd();
  });
  $("#reset").addEventListener("click", () => {
    sliders.forEach(([id]) => { $(`#${id}`).value = 1; $(`#${id}`).dispatchEvent(new Event("input")); });
  });

  // ------------------------------------------------------------ synthesis
  async function synthesize() {
    const text = $("#text").value.trim();
    if (!text) { showError("Please enter some text."); return; }
    showError("");
    const btn = $("#go");
    btn.disabled = true; btn.classList.add("loading"); $(".btn-label", btn).textContent = "Synthesizing";
    try {
      const body = {
        language: state.lang, text, speaker: $("#speaker").value || null,
        duration_control: +$("#dur").value, pitch_control: +$("#pitch").value, energy_control: +$("#energy").value,
      };
      const r = await api("/api/synthesize", { method: "POST", body: JSON.stringify(body) });
      state.result = r;
      renderResult(r);
      addHistory(r);
      refreshModels();
    } catch (e) {
      showError(e.message);
    } finally {
      btn.disabled = false; btn.classList.remove("loading"); $(".btn-label", btn).textContent = "Synthesize";
    }
  }
  $("#go").addEventListener("click", synthesize);
  $("#text").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") synthesize(); });

  function wavUrl(b64) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  }

  function renderResult(r) {
    $("#empty").hidden = true; $("#result").hidden = false;
    const url = wavUrl(r.audio_wav_base64);
    r._url = url;
    const player = $("#player");
    player.src = url; player.play().catch(() => {});
    const dl = $("#download"); dl.href = url; dl.hidden = false;
    const tokens = r.phonemes.filter((p) => p !== "sp" && p !== "sil").length;
    const stats = [
      ["Audio", `${r.audio_seconds.toFixed(2)} s`],
      ["Phonemes", tokens],
      ["Inference", `${Math.round(r.inference_ms)} ms`],
      ["Vocoder", r.vocoder === "hifigan" ? "HiFi-GAN" : "Griffin-Lim"],
    ];
    $("#stats").innerHTML = stats.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`).join("");
    $("#normalized").textContent = r.normalized;
    drawMel(r.mel_preview);
    drawChart();
    $("#words").innerHTML = r.words.filter((w) => w.end > w.start && !/^[.,!?;:]$/.test(w.word)).map((w) => {
      const ms = r.durations_ms.slice(w.start, w.end).reduce((a, b) => a + b, 0);
      return `<div class="word"><div class="w">${esc(w.word)} <span class="muted small">${Math.round(ms)} ms</span></div>` +
        `<div class="p">${esc(r.phonemes.slice(w.start, w.end).join(" "))}</div></div>`;
    }).join("");
  }

  // Magma-like colour map for the mel preview
  const STOPS = [[0, [0, 0, 4]], [0.25, [81, 18, 124]], [0.5, [183, 55, 121]], [0.75, [252, 137, 97]], [1, [252, 253, 191]]];
  function cmap(v) {
    for (let i = 1; i < STOPS.length; i++) {
      if (v <= STOPS[i][0]) {
        const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i];
        const t = (v - a) / (b - a);
        return ca.map((c, j) => Math.round(c + (cb[j] - c) * t));
      }
    }
    return STOPS[STOPS.length - 1][1];
  }
  function drawMel(mel) {
    const cv = $("#mel");
    const T = mel.length, F = mel[0].length;
    cv.width = T; cv.height = F;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(T, F);
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < F; f++) {
        const [r, g, b] = cmap(mel[t][f]);
        const idx = ((F - 1 - f) * T + t) * 4;
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  $$("#var-tabs button").forEach((b) => b.addEventListener("click", () => {
    state.variable = b.dataset.var;
    $$("#var-tabs button").forEach((x) => x.classList.toggle("on", x === b));
    drawChart();
  }));

  function drawChart() {
    const r = state.result;
    if (!r || typeof Chart === "undefined") return;
    const v = state.variable;
    const labels = r.phonemes;
    const data = v === "duration" ? r.durations_ms : v === "pitch" ? r.pitch_hz : r.energy;
    const yTitle = v === "duration" ? "duration (ms)" : v === "pitch" ? "F0 (Hz)" : "energy (L2 norm)";
    const color = COLORS[v];
    if (state.chart) state.chart.destroy();
    const wordOf = (i) => r.words.find((w) => i >= w.start && i < w.end)?.word || "";
    state.chart = new Chart($("#chart"), {
      type: v === "duration" ? "bar" : "line",
      data: {
        labels,
        datasets: [{
          data, label: yTitle, backgroundColor: color, borderColor: color, borderWidth: 2,
          borderRadius: 4, stepped: v !== "duration" ? "middle" : false, pointRadius: 2.5, pointHoverRadius: 5,
          pointBackgroundColor: color, maxBarThickness: 18,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#18181a", padding: 10, displayColors: false,
            callbacks: {
              title: (items) => `${items[0].label}  ·  ${wordOf(items[0].dataIndex)}`,
              label: (item) => `${yTitle}: ${(+item.raw).toFixed(1)}`,
            },
          },
        },
        scales: {
          x: { grid: { display: false }, ticks: { font: { family: "JetBrains Mono", size: 10 }, color: "#55534e", maxRotation: 90, autoSkip: labels.length > 70 } },
          y: { title: { display: true, text: yTitle, color: "#55534e" }, grid: { color: "#efeee9" }, ticks: { color: "#55534e" } },
        },
      },
    });
  }

  // ------------------------------------------------------------ history
  function addHistory(r) {
    state.history.unshift(r);
    state.history = state.history.slice(0, 8);
    $("#history").innerHTML = state.history.map((h) =>
      `<li><span class="tag">${h.language === "telugu" ? "TE" : "EN"}</span><span>${esc(h.text)}</span><audio controls src="${h._url}"></audio></li>`).join("");
  }

  // ------------------------------------------------------------ results tab
  $$("#res-lang button").forEach((b) => b.addEventListener("click", () => loadResults(b.dataset.lang)));
  async function loadResults(lang) {
    state.resultsLang = lang;
    $$("#res-lang button").forEach((b) => b.classList.toggle("on", b.dataset.lang === lang));
    let data;
    try { data = await api(`/api/results/${lang}`); } catch (e) { data = { losses: {}, tests: [] }; }
    const fig = (src, cap) => `<figure><img loading="lazy" src="${src}" alt="${esc(cap)}" /><figcaption>${esc(cap)}</figcaption></figure>`;
    $("#losses").innerHTML = Object.keys(data.losses).length
      ? Object.entries(data.losses).map(([k, v]) => fig(v, k.replace("_", " "))).join("")
      : `<div class="placeholder" style="grid-column:1/-1">No loss plots yet — run <code>${lang === "telugu" ? "Telugu" : "English"}/train.py</code>.</div>`;
    $("#tests").innerHTML = data.tests.length ? data.tests.map((t) => `
      <div class="test">
        <div class="test-head"><span class="test-id">${esc(t.id)}</span><span class="test-text" lang="${lang === "telugu" ? "te" : "en"}">${esc(t.text || "")}</span></div>
        <div class="test-focus">${esc(t.focus || "")}</div>
        <audio controls preload="none" src="${t.audio}"></audio>
        <div class="test-plots">${["overview", "duration", "pitch", "energy"].filter((k) => t.plots[k]).map((k) => fig(t.plots[k], `${t.id} — ${k}`)).join("")}</div>
      </div>`).join("")
      : `<div class="placeholder">No test outputs yet — run <code>${lang === "telugu" ? "Telugu" : "English"}/inference.py</code>.</div>`;
  }

  // ------------------------------------------------------------ boot
  (async () => {
    try { state.sentences = await api("/api/test-sentences"); } catch { state.sentences = {}; }
    setLanguage("english");
    $("#text").value = state.sentences.english?.[0]?.text || "";
    updateCount();
    refreshModels();
  })();
})();
