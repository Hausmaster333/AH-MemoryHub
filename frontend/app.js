const API_BASE = window.__AH_API__ || "";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const SVG = "http://www.w3.org/2000/svg";

const state = {
  view: "observatory",
  result: null,
  trace: [],
  tick: 0,
  animationSteps: [],
  selectedUid: null,
  memory: { labels: new Map(), sections: new Map(), predicates: new Map(), hypernodes: new Map(), links: new Map() },
  candidates: [],
  selectedCandidate: 0,
  playing: null,
  resizeFrame: null,
};

const demoCandidates = [
  {
    predicate: "CAUSE",
    assertion: "Перегрев насоса вызвал остановку агрегата в насосном зале.",
    confidence: .92,
    span: "0:62",
    template: "CAUSE(SUBJECT, OBJECT, LOCATION)",
    bindings: { SUBJECT: "перегрев насоса", OBJECT: "остановка агрегата", LOCATION: "насосный зал" },
  },
  {
    predicate: "FOLLOW",
    assertion: "Остановка агрегата привела к снижению давления.",
    confidence: .86,
    span: "63:112",
    template: "FOLLOW(SUBJECT, OBJECT)",
    bindings: { SUBJECT: "остановка агрегата", OBJECT: "снижение давления" },
  },
  {
    predicate: "USES_TOOL",
    assertion: "После этого оператор применил ручной ключ.",
    confidence: .78,
    span: "113:158",
    template: "USES_TOOL(SUBJECT, TOOL)",
    bindings: { SUBJECT: "оператор", TOOL: "ручной ключ" },
  },
];

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || data.error || data;
    throw new Error(detail.message || detail.code || `HTTP ${response.status}`);
  }
  return data;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("is-visible");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("is-visible"), 2800);
}

function setOnline(online, revision = "—") {
  const element = $("#system-state");
  element.classList.toggle("is-online", online);
  element.classList.toggle("is-offline", !online);
  element.lastElementChild.textContent = online ? "AH Core доступен" : "Backend недоступен";
  $("#run-state").textContent = `${online ? "ready" : "offline"} · rev ${revision}`;
}

function showView(name, updateHash = true) {
  state.view = name;
  $$(".view").forEach((view) => view.classList.toggle("is-active", view.id === `view-${name}`));
  $$("[data-view]").forEach((button) => button.classList.toggle("is-active", button.dataset.view === name));
  if (updateHash) history.replaceState(null, "", `#${name}`);
  if (name === "evaluation" && $("#conformance-score").textContent === "—") runEvaluation();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function ignitionConfig() {
  return {
    initial_life_ticks: Number($("#initial-life").value),
    decay_lambda: Number($("#decay").value),
    working_memory_threshold: Number($("#threshold").value),
    hebbian_eta: Number($("#hebbian").value),
    rhythm_hz: Number($("#rhythm").value),
    max_ticks: Number($("#max-ticks").value),
    epsilon: .0001,
  };
}

async function ensureDemo() {
  const ingestions = await api("/api/v1/ingestions");
  if (!ingestions.count) await api("/api/v1/demo/seed", { method: "POST", body: "{}" });
}

async function loadMemoryMap() {
  const exported = await api("/api/v1/memory/export", { method: "POST", body: "{}" });
  const dump = exported.dump || exported;
  state.memory.labels.clear();
  state.memory.sections.clear();
  state.memory.predicates.clear();
  state.memory.hypernodes.clear();
  state.memory.links.clear();
  for (const symbol of dump.S || []) {
    const text = symbol.sensory_representations?.find((item) => item.modality === "text")?.value || symbol.uid;
    state.memory.labels.set(symbol.uid, text);
    state.memory.sections.set(symbol.uid, "S");
  }
  for (const section of ["C", "P", "H"]) {
    for (const item of dump[section] || []) {
      state.memory.sections.set(item.uid, section);
      const payload = item.payload || {};
      if (payload.predicate_ref) {
        state.memory.labels.set(item.uid, payload.predicate_ref);
      }
    }
  }
  for (const section of ["C", "P", "H"]) {
    for (const item of dump[section] || []) {
      const payload = item.payload || {};
      if (!payload.template_ref) continue;
      const predicate = payload.template_ref.replace(/^tpl_/, "").toUpperCase();
      const bindings = (payload.role_bindings || []).map((binding) => ({
        role: binding.role_id,
        uid: binding.target_ref?.target_uid,
        label: state.memory.labels.get(binding.target_ref?.target_uid) || binding.target_ref?.target_uid || "—",
      }));
      const byRole = Object.fromEntries(bindings.map((binding) => [binding.role, binding.label]));
      const label = predicate === "CAUSE" && byRole.SUBJECT && byRole.OBJECT
        ? `${byRole.SUBJECT} → ${byRole.OBJECT}`
        : payload.evidence?.[0]?.exact_text || predicate;
      const hypernode = { uid: item.uid, section, predicate, bindings, evidence: payload.evidence || [], label };
      state.memory.hypernodes.set(item.uid, hypernode);
      state.memory.predicates.set(item.uid, predicate);
      state.memory.labels.set(item.uid, label);
    }
  }
  for (const link of dump.L || []) {
    state.memory.predicates.set(link.uid, link.type_id || "LINK");
    state.memory.links.set(link.uid, link);
  }
}

function compactLabel(uid) {
  const label = state.memory.labels.get(uid) || uid.replace(/^[a-z]+_/, "");
  return label.length > 46 ? `${label.slice(0, 44)}…` : label;
}

function nodeType(uid) {
  if (uid.startsWith("h_")) return "N";
  return state.memory.sections.get(uid) || (uid.startsWith("s_") ? "S" : "C");
}

function uniqueGraph(result) {
  const path = new Set(result.minimal_path || []);
  const evidenceText = new Set((result.evidence || []).map((item) => item.exact_text));
  let hypernodes = [...state.memory.hypernodes.values()].filter((item) => path.has(item.uid));
  const exact = hypernodes.filter((item) => item.evidence.some((ev) => evidenceText.has(ev.exact_text)));
  if (exact.length) hypernodes = exact;
  const signatures = new Set();
  hypernodes = hypernodes.filter((item) => {
    const signature = `${item.predicate}:${item.bindings.map((binding) => `${binding.role}:${binding.label.toLowerCase()}`).join("|")}`;
    if (signatures.has(signature)) return false;
    signatures.add(signature);
    return true;
  }).slice(0, 2);
  const nodes = [];
  const nodeSet = new Set();
  const edges = [];
  const addNode = (uid) => { if (uid && !nodeSet.has(uid)) { nodeSet.add(uid); nodes.push(uid); } };
  for (const hypernode of hypernodes) {
    addNode(hypernode.uid);
    for (const binding of hypernode.bindings.filter((item) => ["SUBJECT", "OBJECT", "LOCATION", "RESULT"].includes(item.role))) {
      addNode(binding.uid);
      edges.push({ source: hypernode.uid, target: binding.uid, type: binding.role, impulseType: "hypernode" });
    }
  }
  for (const link of state.memory.links.values()) {
    const source = link.source_ref?.target_uid;
    const target = link.target_ref?.target_uid;
    if (nodeSet.has(source) && nodeSet.has(target)) edges.push({ source, target, type: link.type_id, impulseType: "associative" });
  }
  return { nodes, edges };
}

function firstTraceTick(edge) {
  const matches = state.trace.filter((item) => item.source_uid === edge.source && item.target_uid === edge.target && item.impulse_type === edge.impulseType);
  return matches.length ? Math.min(...matches.map((item) => item.tick)) : 0;
}

function nodeFirstTick(uid) {
  const matches = state.trace.filter((item) => item.source_uid === uid || item.target_uid === uid);
  return matches.length ? Math.min(...matches.map((item) => item.tick)) : 0;
}

function edgeKey(edge) { return `${edge.source}>${edge.target}:${edge.impulseType}`; }

function buildAnimationSteps() {
  const graph = uniqueGraph(state.result || {});
  const hypernodeUid = graph.nodes.find((uid) => nodeType(uid) === "N");
  const hypernode = state.memory.hypernodes.get(hypernodeUid);
  if (!hypernode) return [];
  const binding = (role) => hypernode.bindings.filter((item) => item.role === role && graph.nodes.includes(item.uid));
  const objects = [...binding("OBJECT"), ...binding("RESULT")];
  const subjects = binding("SUBJECT");
  const roleEdges = (roles) => graph.edges.filter((edge) => edge.source === hypernodeUid && roles.includes(edge.type));
  const associative = graph.edges.filter((edge) => edge.impulseType === "associative");
  const seedUids = graph.nodes.filter((uid) => (state.result?.seed_uids || []).includes(uid));
  const targetUids = seedUids.length ? seedUids : objects.map((item) => item.uid);
  const steps = [
    {
      label: `Цель: ${objects[0]?.label || compactLabel(targetUids[0] || hypernodeUid)}`,
      nodeUids: targetUids,
      edgeKeys: [],
      focusUid: targetUids[0] || hypernodeUid,
      traceTick: targetUids.length ? Math.min(...targetUids.map(nodeFirstTick)) : 0,
    },
    {
      label: `${hypernode.predicate} · утверждение найдено`,
      nodeUids: [hypernodeUid],
      edgeKeys: roleEdges(["OBJECT", "RESULT"]).map(edgeKey),
      focusUid: hypernodeUid,
      traceTick: nodeFirstTick(hypernodeUid),
    },
  ];
  if (subjects.length) steps.push({
    label: `Причина: ${subjects[0].label}`,
    nodeUids: subjects.map((item) => item.uid),
    edgeKeys: roleEdges(["SUBJECT", "CAUSE"]).map(edgeKey),
    focusUid: subjects[0].uid,
    traceTick: nodeFirstTick(subjects[0].uid),
  });
  if (associative.length) steps.push({
    label: "Причинная связь подтверждена",
    nodeUids: [],
    edgeKeys: associative.map(edgeKey),
    focusUid: hypernodeUid,
    traceTick: Math.min(...associative.map(firstTraceTick)),
  });
  return steps;
}

function addTextLines(group, text, y, className = "") {
  const words = text.split(/\s+/);
  const lines = [];
  let line = "";
  for (const word of words) {
    if (`${line} ${word}`.trim().length > 24 && line) { lines.push(line); line = word; }
    else line = `${line} ${word}`.trim();
  }
  if (line) lines.push(line);
  const textNode = svgElement("text", { y: y - ((lines.length - 1) * 7), class: className });
  lines.slice(0, 3).forEach((value, index) => {
    const tspan = svgElement("tspan", { x: 0, dy: index ? 15 : 0 });
    tspan.textContent = value;
    textNode.append(tspan);
  });
  group.append(textNode);
}

function svgElement(name, attrs = {}) {
  const element = document.createElementNS(SVG, name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function renderGraph() {
  const svg = $("#graph");
  svg.replaceChildren();
  if (!state.result?.minimal_path?.length) {
    $("#graph-empty").hidden = false;
    return;
  }
  $("#graph-empty").hidden = true;
  const { nodes, edges } = uniqueGraph(state.result);
  if (!nodes.length) {
    $("#graph-empty").hidden = false;
    return;
  }
  const rect = $("#graph-stage").getBoundingClientRect();
  const width = Math.max(600, rect.width || 900);
  const height = Math.max(300, rect.height || 360);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const positions = new Map();
  const hypernodes = nodes.filter((uid) => nodeType(uid) === "N");
  hypernodes.forEach((uid, index) => positions.set(uid, { x: width * .5, y: height * ((index + 1) / (hypernodes.length + 1)) }));
  const slots = { SUBJECT: 0, OBJECT: 0, LOCATION: 0, RESULT: 0 };
  for (const edge of edges.filter((item) => item.impulseType === "hypernode")) {
    if (positions.has(edge.target)) continue;
    const side = ["OBJECT", "RESULT"].includes(edge.type) ? .79 : .21;
    const count = edges.filter((item) => item.impulseType === "hypernode" && item.type === edge.type).length;
    const index = slots[edge.type]++;
    if (width < 640) {
      const order = edge.type === "SUBJECT" ? 0 : ["OBJECT", "RESULT"].includes(edge.type) ? 2 : 3;
      positions.set(edge.target, { x: width * .5, y: 66 + order * ((height - 132) / 3) });
    } else positions.set(edge.target, { x: width * side, y: height * ((index + 1) / (count + 1)) });
  }
  const completedSteps = state.animationSteps.slice(0, state.tick + 1);
  const activeNodes = new Set(completedSteps.flatMap((step) => step.nodeUids));
  const activeEdges = new Set(completedSteps.flatMap((step) => step.edgeKeys));
  const currentStep = state.animationSteps[state.tick] || { nodeUids: [], edgeKeys: [] };
  const currentNodes = new Set(currentStep.nodeUids);
  const currentEdges = new Set(currentStep.edgeKeys);
  edges.forEach((edge) => {
    const from = positions.get(edge.source);
    const to = positions.get(edge.target);
    if (!from || !to) return;
    const key = edgeKey(edge);
    const isActive = activeEdges.has(key);
    const isCurrent = currentEdges.has(key);
    const line = svgElement("path", { d: `M ${from.x} ${from.y} C ${(from.x + to.x) / 2} ${from.y}, ${(from.x + to.x) / 2} ${to.y}, ${to.x} ${to.y}`, class: `graph-link ${isActive ? "is-active" : "is-context"} ${isCurrent ? "is-current" : ""}` });
    const title = svgElement("title");
    title.textContent = edge.type;
    line.append(title);
    svg.append(line);
    const label = svgElement("text", { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 - 7, class: "graph-edge-label" });
    label.textContent = edge.type;
    svg.append(label);
  });
  nodes.forEach((uid) => {
    const pos = positions.get(uid);
    const type = nodeType(uid);
    const color = getComputedStyle(document.documentElement).getPropertyValue(`--${type.toLowerCase()}`).trim() || "#1261d8";
    const isActive = activeNodes.has(uid);
    const isCurrent = currentNodes.has(uid);
    const group = svgElement("g", { class: `graph-node ${state.selectedUid === uid ? "is-selected" : ""} ${isActive ? "is-active" : "is-future"} ${isCurrent ? "is-current" : ""}`, transform: `translate(${pos.x} ${pos.y})`, tabindex: "0", role: "button", "aria-label": `${type} ${compactLabel(uid)}` });
    const nodeWidth = type === "N" ? 230 : 190;
    group.append(svgElement("rect", { x: -nodeWidth / 2, y: -42, width: nodeWidth, height: 84, rx: 11, stroke: color, class: "node-shape" }));
    group.append(svgElement("circle", { cx: -nodeWidth / 2 + 18, cy: -25, r: 10, fill: color, stroke: "none" }));
    const marker = svgElement("text", { x: -nodeWidth / 2 + 18, y: -22, class: "node-type" }); marker.textContent = type; group.append(marker);
    addTextLines(group, compactLabel(uid), -4, "node-label");
    const meta = svgElement("text", { y: 29, class: "node-meta" });
    meta.textContent = type === "N" ? `${state.memory.predicates.get(uid)} · ${state.memory.sections.get(uid)} · утверждение` : "S · первичный символ";
    group.append(meta);
    group.addEventListener("click", () => selectNode(uid));
    group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") selectNode(uid); });
    svg.append(group);
  });
}

function selectNode(uid, openInspector = true) {
  state.selectedUid = uid;
  const relatedTrace = state.trace.filter((item) => item.target_uid === uid || item.source_uid === uid);
  const engineTick = state.animationSteps[state.tick]?.traceTick;
  const trace = relatedTrace.find((item) => item.tick === engineTick) || relatedTrace[0];
  const hypernode = state.memory.hypernodes.get(uid);
  const owner = hypernode || [...state.memory.hypernodes.values()].find((item) => item.bindings.some((binding) => binding.uid === uid) && item.evidence.some((ev) => (state.result?.evidence || []).some((resultEv) => resultEv.exact_text === ev.exact_text)));
  const evidence = owner?.evidence?.[0] || state.result?.evidence?.[0];
  const type = nodeType(uid);
  const typeBadge = $("#selected-type");
  typeBadge.textContent = type;
  typeBadge.className = `type-dot type-${type.toLowerCase()}`;
  $("#selected-title").textContent = compactLabel(uid);
  $("#selected-uid").textContent = `uid: ${uid}`;
  const section = state.memory.sections.get(uid) || type;
  const sectionNames = { S: "S · первичные символы", C: "C · общая память", P: "P · частная память", H: "H · история" };
  $("#selected-section").textContent = sectionNames[section] || section;
  $("#selected-tick").textContent = trace?.tick ?? state.tick;
  $("#selected-activation").textContent = trace ? `${trace.next_excitation.toFixed(3)} / ${trace.activation.toFixed(3)}` : "—";
  $("#selected-weight").textContent = trace?.next_weight != null ? trace.next_weight.toFixed(3) : "n/a";
  $("#source-quote").textContent = evidence?.exact_text || "Для этого элемента нет отдельного source span.";
  $("#source-coord").textContent = evidence ? `${evidence.document_uid} · span ${evidence.start_offset}:${evidence.end_offset}` : "document — · span —";
  const bindings = $("#selected-bindings");
  if (owner) {
    bindings.replaceChildren(...owner.bindings.map((binding) => {
      const row = document.createElement("div");
      row.innerHTML = "<b></b><span></span>";
      $("b", row).textContent = binding.role;
      $("span", row).textContent = binding.label;
      return row;
    }));
  } else {
    const empty = document.createElement("span"); empty.textContent = "У элемента нет RoleBindings."; bindings.replaceChildren(empty);
  }
  $("#relation-type").textContent = trace?.impulse_type || "selected";
  $("#relation-parent").textContent = `parent: ${trace?.parent_trace ?? "root"}`;
  renderGraph();
  if (openInspector && window.innerWidth < 1280) $("#inspector").classList.add("is-open");
}

function updateTick(value) {
  state.tick = Number(value);
  $("#tick-range").value = state.tick;
  const step = state.animationSteps[state.tick];
  $("#tick-label").textContent = step ? `шаг ${state.tick + 1} из ${state.animationSteps.length} · engine tick ${step.traceTick}` : "путь не построен";
  $$("#tick-marks button").forEach((button) => button.classList.toggle("is-active", Number(button.dataset.tick) === state.tick));
  if (step?.focusUid) selectNode(step.focusUid, false); else renderGraph();
}

function configureTimeline() {
  state.animationSteps = buildAnimationSteps();
  const maxStep = Math.max(0, state.animationSteps.length - 1);
  $("#tick-range").max = maxStep;
  $("#tick-marks").replaceChildren(...state.animationSteps.map((step, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.tick = i;
    button.innerHTML = `<b>${i + 1}</b><span></span>`;
    $("span", button).textContent = step.label;
    button.addEventListener("click", () => updateTick(i));
    return button;
  }));
  updateTick(0);
}

function setQueryLoading(loading) {
  $("#answer-block").classList.toggle("is-loading", loading);
  $("#query-form button").disabled = loading;
  if (loading) {
    $("#answer-status").textContent = "Строим доказуемый ответ";
    $("#answer-text").textContent = "Собираем неизменяемый срез памяти и запускаем контур активации.";
    $("#trace-badge").textContent = "running";
  }
}

async function runQuery(question = $("#query-input").value.trim()) {
  if (!question) return;
  setQueryLoading(true);
  $("#answer-block").classList.remove("is-error");
  try {
    const result = await api("/api/v1/queries", {
      method: "POST",
      body: JSON.stringify({ question, profile: $("#profile").value, ignition: ignitionConfig() }),
    });
    state.result = result;
    state.trace = result.trace || [];
    await loadMemoryMap();
    $("#answer-status").textContent = result.status === "answered" ? "Ответ по доказательствам" : "Недостаточно доказательств";
    $("#answer-text").textContent = result.status === "answered" ? result.answer : "В AH-памяти нет достаточного подтверждённого пути. Ответ не был дополнен знаниями модели.";
    $("#trace-badge").textContent = result.trace_complete ? `полная трасса · ${state.trace.length} событий` : "трасса недоступна";
    $("#path-state").textContent = result.trace_complete ? "путь подтверждён" : "неполный путь";
    $("#run-state").textContent = `${result.profile || $("#profile").value} · ${result.run_uid ? result.run_uid.slice(0, 14) : "no run"}`;
    configureTimeline();
    const first = result.working_memory?.[0] || result.minimal_path?.find((uid) => !uid.startsWith("l_"));
    if (first) selectNode(first, false); else renderGraph();
  } catch (error) {
    state.result = null; state.trace = [];
    $("#answer-block").classList.add("is-error");
    $("#answer-status").textContent = "BACKEND UNAVAILABLE";
    $("#answer-text").textContent = "Не удалось связаться с AH Core. Проверьте процесс backend и повторите запрос.";
    $("#trace-badge").textContent = error.message;
    setOnline(false);
    renderGraph();
  } finally {
    setQueryLoading(false);
  }
}

function deriveCandidates(text) {
  let offset = 0;
  const sentences = text.match(/[^.!?]+[.!?]?/g)?.map((item) => item.trim()).filter(Boolean) || [];
  return sentences.slice(0, 6).map((assertion) => {
    const start = text.indexOf(assertion, offset); const end = start + assertion.length; offset = end;
    const low = assertion.toLowerCase();
    const predicate = /вызвал|из-за|привел|привела|привело/.test(low) ? "CAUSE" : /после|затем/.test(low) ? "FOLLOW" : /ключ|инструмент|применил/.test(low) ? "USES_TOOL" : "OBSERVED";
    const parts = assertion.replace(/[.!?]$/, "").split(/\s+(?:вызвал[аио]?|привел[аио]? к|после этого|применил[аио]?)\s+/i);
    const bindings = { SUBJECT: parts[0] || assertion, OBJECT: parts[1] || assertion };
    const location = assertion.match(/в\s+([а-яё\s-]+?(?:зале|цехе|помещении))/i); if (location) bindings.LOCATION = location[1];
    const tool = assertion.match(/(?:ключ|инструмент)\w*(?:\s+[а-яё-]+)?/i); if (tool) bindings.TOOL = tool[0];
    return { predicate, assertion, confidence: .78, span: `${start}:${end}`, template: `${predicate}(${Object.keys(bindings).join(", ")})`, bindings };
  });
}

function renderCandidates() {
  const list = $("#candidate-list");
  list.replaceChildren(...state.candidates.map((candidate, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `candidate-card ${index === state.selectedCandidate ? "is-selected" : ""}`;
    button.innerHTML = `<span class="predicate">${candidate.predicate}</span><span class="candidate-copy"><b></b><code></code></span><span class="confidence"></span>`;
    $("b", button).textContent = candidate.assertion;
    $("code", button).textContent = `Hyperedge template · ${candidate.template}`;
    $(".confidence", button).textContent = `${Math.round(candidate.confidence * 100)}% · span ${candidate.span}`;
    button.addEventListener("click", () => { state.selectedCandidate = index; renderCandidates(); });
    return button;
  }));
  $("#candidate-count").textContent = `${state.candidates.length} candidates`;
  const selected = state.candidates[state.selectedCandidate];
  if (!selected) {
    $("#candidate-title").textContent = "Нет выбранного кандидата";
    $("#binding-list").replaceChildren();
    return;
  }
  $("#candidate-title").textContent = `${selected.predicate} · выбранный кандидат`;
  $("#candidate-span").textContent = `source span ${selected.span} · confidence ${selected.confidence.toFixed(2)}`;
  $("#candidate-quote").textContent = selected.assertion;
  $("#binding-list").replaceChildren(...Object.entries(selected.bindings).map(([role, value]) => {
    const row = document.createElement("div"); row.className = "binding-row";
    const marker = document.createElement("span"); marker.textContent = role[0]; marker.title = role;
    const label = document.createElement("b"); label.textContent = `${role} · ${value}`;
    row.append(marker, label); return row;
  }));
}

async function admitDocument(event) {
  event.preventDefault();
  const text = $("#ingestion-text").value.trim();
  if (!text) return;
  const button = $("#admit-candidate"); button.disabled = true;
  try {
    const result = await api("/api/v1/ingestions", { method: "POST", body: JSON.stringify({ text, source_name: "ingestion-review" }) });
    $("#admission-result").textContent = `${result.candidates.length} candidates · ${result.accepted.length} admitted · ${result.rejected.length} rejected`;
    toast(`AH Core принял ${result.accepted.length} фактов`);
  } catch (error) {
    $("#admission-result").textContent = `validation error · ${error.message}`;
  } finally { button.disabled = false; }
}

function percent(value) { return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—"; }

async function runEvaluation() {
  const button = $("#run-evaluation"); button.disabled = true; button.textContent = "Считаем…";
  try {
    const result = await api("/api/v1/evaluations", { method: "POST", body: "{}" });
    const metrics = result.metrics;
    $("#conformance-score").textContent = result.fixtures.rabbit.conformant ? `${result.fixtures.rabbit.accepted}/${result.fixtures.rabbit.facts}` : "FAIL";
    const values = [
      [percent(metrics.M1.weighted_f1), "weighted F1"],
      [percent(metrics.M2.explain_score), metrics.M2.trace_complete ? "trace complete" : "trace incomplete"],
      [percent(metrics.M3.gc_efficiency), `${metrics.M3.deleted} deleted`],
      [metrics.M4.status, "external LLM"],
      [metrics.M5.status, "SLM / frontier"],
    ];
    $$("#metric-grid article").forEach((card, index) => { $("strong", card).textContent = values[index][0]; $("small", card).textContent = values[index][1]; });
    toast(`Evaluation computed · ${result.elapsed_ms} ms`);
  } catch (error) { toast(`Evaluation: ${error.message}`); }
  finally { button.disabled = false; button.textContent = "Пересчитать"; }
}

function openParams(open) {
  $("#params-drawer").classList.toggle("is-open", open);
  $("#params-drawer").setAttribute("aria-hidden", String(!open));
  $("#drawer-backdrop").classList.toggle("is-open", open);
}

function bindEvents() {
  $$("[data-view]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
  $("#query-form").addEventListener("submit", (event) => { event.preventDefault(); runQuery(); });
  $("#tick-range").addEventListener("input", (event) => updateTick(event.target.value));
  $("#tick-play").addEventListener("click", () => {
    if (state.playing) { clearInterval(state.playing); state.playing = null; $("#tick-play").textContent = "Продолжить"; return; }
    if (!state.trace.length) return;
    $("#tick-play").textContent = "Пауза";
    if (state.tick >= Number($("#tick-range").max)) updateTick(0);
    state.playing = setInterval(() => {
      const next = state.tick + 1;
      if (next > Number($("#tick-range").max)) { clearInterval(state.playing); state.playing = null; $("#tick-play").textContent = "Повторить"; return; }
      updateTick(next);
    }, 900);
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden && state.playing) $("#tick-play").click(); });
  $("#inspector-close").addEventListener("click", () => $("#inspector").classList.remove("is-open"));
  $("#params-open").addEventListener("click", () => openParams(true));
  $("#params-top-open").addEventListener("click", () => openParams(true));
  $("#params-mobile-open").addEventListener("click", () => openParams(true));
  $("#params-close").addEventListener("click", () => openParams(false));
  $("#drawer-backdrop").addEventListener("click", () => openParams(false));
  $("#profile").addEventListener("change", (event) => { $("#profile-label").textContent = event.target.value; });
  $("#preview-candidates").addEventListener("click", () => { state.candidates = deriveCandidates($("#ingestion-text").value); state.selectedCandidate = 0; renderCandidates(); });
  $("#ingestion-form").addEventListener("submit", admitDocument);
  $("#reject-candidate").addEventListener("click", () => { if (!state.candidates.length) return; state.candidates.splice(state.selectedCandidate, 1); state.selectedCandidate = Math.max(0, state.selectedCandidate - 1); renderCandidates(); $("#admission-result").textContent = `${state.candidates.length} candidates · 0 admitted`; });
  $("#run-evaluation").addEventListener("click", runEvaluation);
  $("#gc-preview").addEventListener("click", async () => {
    try { const result = await api("/api/v1/gc/preview", { method: "POST", body: "{}" }); $("#gc-output").textContent = `${result.orphan_count_before} collectible · token ${result.preview_token}`; }
    catch (error) { $("#gc-output").textContent = error.message; }
  });
  window.addEventListener("hashchange", () => { const next = location.hash.slice(1); if (["observatory", "ingestion", "evaluation"].includes(next)) showView(next, false); });
  window.addEventListener("resize", () => { cancelAnimationFrame(state.resizeFrame); state.resizeFrame = requestAnimationFrame(renderGraph); });
}

async function boot() {
  bindEvents();
  state.candidates = demoCandidates.map((item) => ({ ...item, bindings: { ...item.bindings } }));
  renderCandidates();
  const initialView = location.hash.slice(1);
  if (["observatory", "ingestion", "evaluation"].includes(initialView)) showView(initialView, false);
  try {
    const health = await api("/health");
    setOnline(true, health.revision);
    await ensureDemo();
    await runQuery();
  } catch (error) {
    setOnline(false);
    $("#answer-status").textContent = "BACKEND UNAVAILABLE";
    $("#answer-text").textContent = "Интерфейс готов, но AH Core не отвечает. Запустите backend и обновите страницу.";
    $("#trace-badge").textContent = error.message;
  }
}

boot();
