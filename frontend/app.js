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
  memory: { labels: new Map(), sections: new Map(), predicates: new Map(), templates: new Map(), hypernodes: new Map(), links: new Map() },
  memoryExport: null,
  memoryFilter: "ALL",
  memoryPositions: [],
  memorySelectedUid: null,
  candidates: [],
  rejectionLog: [],
  coverageWarnings: [],
  sourceGroups: new Map(),
  previewUid: null,
  selectedCandidate: 0,
  playing: null,
  resizeFrame: null,
};

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
  if (name === "memory") refreshMemoryView();
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
  state.memoryExport = exported;
  const dump = exported.dump || exported;
  state.memory.labels.clear();
  state.memory.sections.clear();
  state.memory.predicates.clear();
  state.memory.templates.clear();
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
      if (payload.predicate_ref) state.memory.templates.set(item.uid, payload);
      const ownLabel = payload.properties?.find((property) => property.name === "label")?.value;
      if (ownLabel) state.memory.labels.set(item.uid, String(ownLabel));
    }
  }
  const refTarget = (reference) => typeof reference === "string" ? reference : reference?.target_uid;
  const predicateLabel = (templateUid) => {
    const predicateUid = refTarget(state.memory.templates.get(templateUid)?.predicate_ref);
    return state.memory.labels.get(predicateUid) || predicateUid?.replace(/^pred_/, "").toUpperCase() || templateUid;
  };
  for (const section of ["C", "P", "H"]) {
    for (const item of dump[section] || []) {
      const payload = item.payload || {};
      if (payload.kind === "S" || payload.kind === "M") state.memory.labels.set(item.uid, `${payload.kind}* → ${state.memory.labels.get(payload.target_uid) || payload.target_uid}`);
      else if (payload.predicate_ref) state.memory.labels.set(item.uid, predicateLabel(item.uid));
      else if (payload.function_id) state.memory.labels.set(item.uid, payload.function_id);
      else if (payload.ordered_members) state.memory.labels.set(item.uid, `${payload.list_type || "LIST"} · ${payload.ordered_members.length}`);
    }
  }
  for (const section of ["C", "P", "H"]) {
    for (const item of dump[section] || []) {
      const payload = item.payload || {};
      if (!payload.template_ref) continue;
      const templateUid = refTarget(payload.template_ref);
      const predicate = predicateLabel(templateUid);
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
  if (state.view === "memory") requestAnimationFrame(() => { updateMemoryStats(); renderMemoryGraph(); });
  return exported;
}

function memoryKind(payload = {}) {
  if (payload.kind === "S") return "S*";
  if (payload.kind === "M") return "M*";
  if (payload.template_ref) return "N";
  if (payload.predicate_ref) return "T";
  if (payload.ordered_members) return "LIST";
  if (payload.function_id) return "FUNCTION";
  return "M";
}

function completeMemoryGraph() {
  const dump = state.memoryExport?.dump || {};
  const nodes = [];
  for (const symbol of dump.S || []) nodes.push({ uid: symbol.uid, section: "S", kind: "S", label: state.memory.labels.get(symbol.uid) || symbol.uid, raw: symbol });
  for (const section of ["C", "P", "H"]) {
    for (const item of dump[section] || []) {
      const kind = memoryKind(item.payload);
      nodes.push({ uid: item.uid, section, kind, label: state.memory.labels.get(item.uid) || item.uid, raw: item });
    }
  }
  const known = new Set(nodes.map((node) => node.uid));
  const edges = [];
  for (const node of nodes) {
    const payload = node.raw.payload || {};
    if ((node.kind === "S*" || node.kind === "M*") && known.has(payload.target_uid)) edges.push({ source: node.uid, target: payload.target_uid, type: node.kind, kind: "reference" });
    const predicateTarget = payload.predicate_ref?.target_uid;
    if (predicateTarget && known.has(predicateTarget)) edges.push({ source: node.uid, target: predicateTarget, type: "PREDICATE", kind: "reference" });
    const templateTarget = payload.template_ref?.target_uid;
    if (templateTarget && known.has(templateTarget)) edges.push({ source: node.uid, target: templateTarget, type: "T*", kind: "reference" });
  }
  for (const node of nodes.filter((item) => item.kind === "N")) {
    for (const binding of node.raw.payload.role_bindings || []) {
      const target = binding.target_ref?.target_uid;
      if (known.has(target)) edges.push({ source: node.uid, target, type: binding.role_id, kind: "role" });
    }
  }
  for (const link of dump.L || []) {
    const source = link.source_ref?.target_uid;
    const target = link.target_ref?.target_uid;
    if (known.has(source) && known.has(target)) edges.push({ source, target, type: link.type_id || "L", kind: "link", uid: link.uid });
  }
  return { nodes, edges };
}

function updateMemoryStats() {
  const exported = state.memoryExport;
  if (!exported) return;
  const dump = exported.dump;
  const values = [dump.S, dump.C, dump.P, dump.H, dump.L].map((items) => items?.length || 0);
  $$("#memory-stats article:not(.memory-revision) strong").forEach((element, index) => { element.textContent = values[index]; });
  $("#memory-revision").textContent = `rev ${dump.revision} · tick ${dump.current_tick}`;
  $("#memory-checksum").textContent = exported.sha256;
}

function renderMemoryInspector(uid) {
  const graph = completeMemoryGraph();
  const node = graph.nodes.find((item) => item.uid === uid);
  if (!node) return;
  state.memorySelectedUid = uid;
  const badge = $("#memory-selected-type");
  badge.textContent = node.kind;
  badge.className = `type-dot type-${node.kind === "N" ? "n" : node.section.toLowerCase()}`;
  $("#memory-selected-label").textContent = node.label;
  $("#memory-selected-uid").textContent = `uid: ${uid}`;
  $("#memory-selected-section").textContent = node.section;
  $("#memory-selected-degree").textContent = graph.edges.filter((edge) => edge.source === uid || edge.target === uid).length;
  const templateUid = node.raw.payload?.template_ref?.target_uid;
  const predicateUid = node.raw.payload?.predicate_ref?.target_uid;
  $("#memory-selected-predicate").textContent = node.kind === "N"
    ? state.memory.labels.get(templateUid) || templateUid || "—"
    : state.memory.labels.get(predicateUid) || predicateUid || "—";
  $("#memory-selected-kind").textContent = node.kind;
  const hypernode = state.memory.hypernodes.get(uid);
  const bindings = $("#memory-selected-bindings");
  if (hypernode?.bindings.length) {
    bindings.replaceChildren(...hypernode.bindings.map((binding) => {
      const row = document.createElement("div"); row.innerHTML = "<b></b><span></span>";
      $("b", row).textContent = binding.role; $("span", row).textContent = binding.label; return row;
    }));
  } else { const empty = document.createElement("span"); empty.textContent = "У элемента нет RoleBindings."; bindings.replaceChildren(empty); }
  const evidence = hypernode?.evidence?.[0] || node.raw.payload?.evidence?.[0];
  $("#memory-selected-source").textContent = evidence?.exact_text || "Для выбранного элемента нет отдельного source span.";
  $("#memory-selected-coord").textContent = evidence ? `${evidence.document_uid} · span ${evidence.start_offset}:${evidence.end_offset}` : "document — · span —";
  renderMemoryGraph();
}

function renderMemoryGraph() {
  const canvas = $("#memory-canvas");
  const wrap = $("#memory-canvas-wrap");
  if (!canvas || !state.memoryExport) return;
  const graph = completeMemoryGraph();
  let nodes = graph.nodes;
  if (state.memoryFilter !== "ALL") {
    const focused = new Set(graph.nodes.filter((node) => node.section === state.memoryFilter).map((node) => node.uid));
    const visibleWithContext = new Set(focused);
    graph.edges.forEach((edge) => { if (focused.has(edge.source) || focused.has(edge.target)) { visibleWithContext.add(edge.source); visibleWithContext.add(edge.target); } });
    nodes = graph.nodes.filter((node) => visibleWithContext.has(node.uid));
  }
  const visible = new Set(nodes.map((node) => node.uid));
  const edges = graph.edges.filter((edge) => visible.has(edge.source) && visible.has(edge.target));
  $("#memory-empty").hidden = Boolean(nodes.length);
  $("#memory-visible-count").textContent = `${nodes.length} узлов · ${edges.length} рёбер`;
  const rect = wrap.getBoundingClientRect();
  const width = Math.max(320, rect.width || 900);
  const height = Math.max(480, rect.height || 620);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  const sections = ["S", "C", "P", "H"].filter((section) => nodes.some((node) => node.section === section));
  const laneWidth = width / sections.length;
  const colors = { S: "#257f7a", C: "#c45b35", P: "#2367d1", H: "#7651a8", N: "#2f7e91" };
  state.memoryPositions = [];
  sections.forEach((section, sectionIndex) => {
    const laneNodes = nodes.filter((node) => node.section === section);
    const columns = Math.max(1, Math.ceil(Math.sqrt(laneNodes.length * Math.max(.45, laneWidth / height))));
    const rows = Math.max(1, Math.ceil(laneNodes.length / columns));
    ctx.fillStyle = "#656a67"; ctx.font = "10px Consolas"; ctx.fillText(`${section} · ${laneNodes.length}`, sectionIndex * laneWidth + 14, 22);
    if (sectionIndex) { ctx.strokeStyle = "#d9d6cf"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(sectionIndex * laneWidth, 0); ctx.lineTo(sectionIndex * laneWidth, height); ctx.stroke(); }
    laneNodes.forEach((node, index) => {
      const column = index % columns; const row = Math.floor(index / columns);
      const x = sectionIndex * laneWidth + (column + 1) * laneWidth / (columns + 1);
      const y = 44 + (row + 1) * (height - 64) / (rows + 1);
      state.memoryPositions.push({ uid: node.uid, x, y, radius: laneNodes.length > 180 ? 3 : laneNodes.length > 60 ? 4 : 6 });
    });
  });
  const positions = new Map(state.memoryPositions.map((position) => [position.uid, position]));
  const query = $("#memory-search").value.trim().toLowerCase();
  const matches = new Set(nodes.filter((node) => !query || `${node.uid} ${node.label} ${node.kind}`.toLowerCase().includes(query)).map((node) => node.uid));
  ctx.lineWidth = 1;
  edges.forEach((edge) => {
    const from = positions.get(edge.source); const to = positions.get(edge.target); if (!from || !to) return;
    const selected = state.memorySelectedUid && (edge.source === state.memorySelectedUid || edge.target === state.memorySelectedUid);
    const matched = !query || matches.has(edge.source) || matches.has(edge.target);
    ctx.strokeStyle = selected ? "rgba(18,97,216,.8)" : matched ? "rgba(70,76,73,.16)" : "rgba(70,76,73,.035)";
    ctx.lineWidth = selected ? 1.8 : 1; ctx.setLineDash(edge.kind === "role" ? [3, 3] : []);
    ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
  });
  ctx.setLineDash([]);
  nodes.forEach((node) => {
    const position = positions.get(node.uid); if (!position) return;
    const selected = node.uid === state.memorySelectedUid; const matched = matches.has(node.uid);
    ctx.globalAlpha = query && !matched ? .13 : 1;
    ctx.beginPath(); ctx.arc(position.x, position.y, selected ? position.radius + 3 : position.radius, 0, Math.PI * 2);
    ctx.fillStyle = colors[node.kind === "N" ? "N" : node.section]; ctx.fill();
    if (selected) { ctx.strokeStyle = "#17191a"; ctx.lineWidth = 2; ctx.stroke(); }
  });
  ctx.globalAlpha = 1;
  if (query) $("#memory-visible-count").textContent += ` · ${matches.size} совпадений`;
}

async function refreshMemoryView() {
  try { await loadMemoryMap(); updateMemoryStats(); renderMemoryGraph(); }
  catch (error) { $("#memory-visible-count").textContent = error.message; }
}

function downloadMemoryDump() {
  if (!state.memoryExport) return;
  const revision = state.memoryExport.dump?.revision ?? "unknown";
  const blob = new Blob([JSON.stringify(state.memoryExport, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob); const link = document.createElement("a");
  link.href = url; link.download = `ah-memory-rev-${revision}.json`; link.click(); URL.revokeObjectURL(url);
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
  const path = new Set(result.answer_path?.length ? result.answer_path : result.minimal_path || []);
  let hypernodes = [...state.memory.hypernodes.values()].filter((item) => path.has(item.uid));
  const signatures = new Set();
  hypernodes = hypernodes.filter((item) => {
    const signature = `${item.predicate}:${item.bindings.map((binding) => `${binding.role}:${binding.label.toLowerCase()}`).join("|")}`;
    if (signatures.has(signature)) return false;
    signatures.add(signature);
    return true;
  }).slice(0, 8);
  const nodes = [];
  const nodeSet = new Set();
  const edges = [];
  const addNode = (uid) => { if (uid && !nodeSet.has(uid)) { nodeSet.add(uid); nodes.push(uid); } };
  for (const hypernode of hypernodes) {
    addNode(hypernode.uid);
    for (const binding of hypernode.bindings) {
      addNode(binding.uid);
      edges.push({ source: hypernode.uid, target: binding.uid, type: binding.role, impulseType: "hypernode" });
    }
  }
  for (const link of state.memory.links.values()) {
    const source = link.source_ref?.target_uid;
    const target = link.target_ref?.target_uid;
    const representedByHypernode = hypernodes.some((item) => item.predicate === link.type_id && item.bindings.some((binding) => binding.uid === source) && item.bindings.some((binding) => binding.uid === target));
    if (!representedByHypernode && nodeSet.has(source) && nodeSet.has(target)) edges.push({ source, target, type: link.type_id, impulseType: "associative" });
  }
  return { nodes, edges };
}

function firstTraceTick(edge) {
  const matches = state.trace.filter((item) =>
    (item.source_uid === edge.source && item.target_uid === edge.target && item.impulse_type === edge.impulseType)
    || (edge.impulseType === "hypernode" && item.source_uid === edge.target && item.target_uid === edge.source && item.impulse_type === "role_to_hypernode")
  );
  return matches.length ? Math.min(...matches.map((item) => item.tick)) : 0;
}

function nodeFirstTick(uid) {
  const matches = state.trace.filter((item) => item.source_uid === uid || item.target_uid === uid);
  return matches.length ? Math.min(...matches.map((item) => item.tick)) : 0;
}

function edgeKey(edge) { return `${edge.source}>${edge.target}:${edge.impulseType}`; }

function buildAnimationSteps() {
  const graph = uniqueGraph(state.result || {});
  const seedUids = graph.nodes.filter((uid) => (state.result?.seed_uids || []).includes(uid));
  const steps = seedUids.length ? [{ label: `Запрос активировал: ${seedUids.map(compactLabel).join(", ")}`, nodeUids: seedUids, edgeKeys: [], focusUid: seedUids[0], traceTick: 0 }] : [];
  const events = graph.nodes.filter((uid) => nodeType(uid) === "N").map((uid) => {
    const edges = graph.edges.filter((edge) => edge.source === uid && edge.impulseType === "hypernode");
    return { label: `${state.memory.hypernodes.get(uid)?.predicate || "FACT"}: ${compactLabel(uid)}`, nodeUids: [uid, ...edges.map((edge) => edge.target)], edgeKeys: edges.map(edgeKey), focusUid: uid, traceTick: Math.min(...edges.map(firstTraceTick)) };
  });
  for (const edge of graph.edges.filter((item) => item.impulseType === "associative")) events.push({ label: `${edge.type}: ${compactLabel(edge.source)} → ${compactLabel(edge.target)}`, nodeUids: [edge.source, edge.target], edgeKeys: [edgeKey(edge)], focusUid: edge.target, traceTick: firstTraceTick(edge) });
  events.sort((a, b) => a.traceTick - b.traceTick || a.label.localeCompare(b.label));
  steps.push(...events.slice(0, 15));
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
  const roleEdges = edges.filter((item) => item.impulseType === "hypernode");
  if (width < 640) {
    for (const edge of roleEdges) {
      if (positions.has(edge.target)) continue;
      const order = edge.type === "SUBJECT" ? 0 : ["OBJECT", "RESULT"].includes(edge.type) ? 2 : 3;
      positions.set(edge.target, { x: width * .5, y: 66 + order * ((height - 132) / 3) });
    }
  } else {
    for (const side of ["left", "right"]) {
      const targets = [...new Set(roleEdges.filter((edge) => (["OBJECT", "RESULT"].includes(edge.type) ? "right" : "left") === side).map((edge) => edge.target))];
      targets.forEach((uid, index) => positions.set(uid, { x: width * (side === "right" ? (width < 800 ? .83 : .79) : (width < 800 ? .17 : .21)), y: height * ((index + 1) / (targets.length + 1)) }));
    }
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
      body: JSON.stringify({ question, profile: $("#profile").value, answer_model: $("#answer-model").value, ignition: ignitionConfig() }),
    });
    state.result = result;
    state.trace = result.trace || [];
    await loadMemoryMap();
    $("#answer-status").textContent = result.status === "answered" ? `${result.answer_mode === "llm_grounded" ? "LLM" : result.answer_warning ? "Локальный fallback" : "Локальный"} · ответ по доказательствам` : "Недостаточно доказательств";
    $("#answer-provenance-detail").textContent = result.answer_mode === "llm_grounded"
      ? `${result.answer_fact_count ?? result.grounded_fact_count} фактов ответа · citations проверены сервером${result.answer_provider?.model_id ? ` · ${result.answer_provider.model_id}` : ""}`
      : result.answer_warning || "Внешняя модель не использовалась";
    $("#answer-text").textContent = result.status === "answered" ? result.answer : "В AH-памяти нет достаточного подтверждённого пути. Ответ не был дополнен знаниями модели.";
    $("#trace-badge").textContent = result.trace_complete ? `полная трасса · ${state.trace.length} событий` : "трасса недоступна";
    $("#path-state").textContent = result.trace_complete ? "путь подтверждён" : "неполный путь";
    $("#run-state").textContent = `${result.profile || $("#profile").value} · ${result.run_uid ? result.run_uid.slice(0, 14) : "no run"}`;
    configureTimeline();
    const first = result.answer_path?.find((uid) => uid.startsWith("h_")) || result.answer_path?.[0] || result.working_memory?.[0] || result.minimal_path?.find((uid) => !uid.startsWith("l_"));
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

function candidateFromPreview(item) {
  const bindings = Object.fromEntries((item.bindings || []).map((binding) => [binding.role_id, binding.value]));
  const term = (value = "") => value
    .replace(/^OR\((.*)\)$/u, (_, operands) => operands.split(", ").join(" или "))
    .replace(/^AND\((.*)\)$/u, (_, operands) => operands.split(", ").join(" и "))
    .replace(/^VERY\((.*)\)$/u, "очень $1");
  const b = Object.fromEntries(Object.entries(bindings).map(([role, value]) => [role, term(value)]));
  const assertion = ({
    "IS-A": () => `${b.SUBJECT} — ${b.OBJECT}`,
    LOCATED_AT: () => `${b.SUBJECT} находится: ${b.LOCATION}`,
    LIVE: () => `${b.SUBJECT} обитает: ${b.LOCATION}`,
    HAS: () => `${b.SUBJECT} имеет: ${b.OBJECT}`,
    HAS_STATE: () => `${b.SUBJECT}: ${b.STATE}${b.TIME ? ` · ${b.TIME}` : ""}`,
    CAUSE: () => `${b.SUBJECT} вызывает: ${b.OBJECT}`,
    FOLLOW: () => `${b.SUBJECT} → затем: ${b.OBJECT}`,
    USES_TOOL: () => `${b.SUBJECT} использует ${b.TOOL}${b.OBJECT ? ` для: ${b.OBJECT}` : ""}`,
    RUN: () => `${b.SUBJECT} действует: ${b["HOW-TO"]}`,
    ACTION: () => `${b.SUBJECT} совершает: ${b.OBJECT}${b.TOOL ? ` · инструмент: ${b.TOOL}` : ""}`,
    PURPOSE: () => `${b.SUBJECT} предназначен для: ${b.PURPOSE}`,
  }[item.predicate] || (() => item.exact_text))();
  return { candidateUid: item.candidate_uid, predicate: item.predicate, assertion, sourceQuote: item.exact_text, confidence: item.confidence, span: `${item.source_start}:${item.source_end}`, groupUid: item.group_uid || item.span_uid, template: `${item.predicate}(${Object.keys(bindings).join(", ")})`, bindings, modelId: item.model_id, status: item.status || "pending", sectionHint: item.section_hint || "C", sectionConfidence: item.section_confidence ?? .5, sectionReason: item.section_reason || "" };
}

async function previewCandidates() {
  const text = $("#ingestion-text").value.trim(); if (!text) return;
  const button = $("#preview-candidates"); button.disabled = true; button.textContent = "Извлекаем…";
  $("#parser-status").textContent = "Серверный parser обрабатывает документ";
  try {
    const result = await api("/api/v1/ingestions/preview", { method: "POST", body: JSON.stringify({ text, source_name: "ingestion-review", parser_model: $("#parser-model").value }) });
    state.previewUid = result.preview_uid; state.selectedCandidate = 0;
    state.candidates = result.candidates.map(candidateFromPreview);
    state.rejectionLog = result.rejection_log || [];
    state.coverageWarnings = result.coverage_warnings || [];
    state.sourceGroups = new Map((result.source_groups || []).map((group) => [group.uid, group]));
    const provider = result.provider;
    const fallback = provider.fallback ? " · fallback" : "";
    const cached = provider.cached ? " · cache" : "";
    $("#parser-status").textContent = `${provider.active} · ${provider.model_id} · ${provider.prompt_version}${fallback}${cached}`;
    $("#document-label").textContent = result.document_uid;
    const rejected = result.rejection_log || []; const uncovered = result.coverage_warnings || [];
    $("#admission-result").textContent = rejected.length || uncovered.length
      ? `${result.count} candidates · ${rejected.length} отсечено · ${uncovered.length} фрагментов без фактов`
      : `${result.count} candidates · ожидают решения`;
    renderCandidates();
    if ($("#auto-admit").checked) await autoAdmitCandidates();
  } catch (error) {
    state.previewUid = null; state.candidates = []; state.rejectionLog = []; state.coverageWarnings = []; state.sourceGroups.clear(); renderCandidates();
    $("#parser-status").textContent = "parser unavailable";
    $("#admission-result").textContent = error.message;
  } finally { button.disabled = false; button.textContent = "Собрать кандидатов"; }
}

async function autoAdmitCandidates() {
  if (!state.previewUid) return;
  $("#admission-result").textContent = "Автоприём: компилятор проверяет кандидатов…";
  const result = await api("/api/v1/ingestions/auto-admit", { method: "POST", body: JSON.stringify({ preview_uid: state.previewUid }) });
  state.candidates = result.candidates.map(candidateFromPreview);
  $("#admission-result").textContent = `${result.admitted} admitted · ${result.rejected} rejected · ${result.pending} pending`;
  renderCandidates();
  if (result.admitted) await loadMemoryMap();
  toast(result.processed ? `Автоприём завершён · rev ${result.memory_revision}` : "Все кандидаты уже обработаны");
}

function renderCandidates() {
  const list = $("#candidate-list");
  const groupCounts = new Map(state.candidates.map((candidate) => [candidate.groupUid, state.candidates.filter((item) => item.groupUid === candidate.groupUid).length]));
  const children = [];
  let previousGroup = null;
  state.candidates.forEach((candidate, index) => {
    if (candidate.groupUid !== previousGroup) {
      const group = document.createElement("div"); group.className = "candidate-group-label";
      const source = state.sourceGroups.get(candidate.groupUid);
      const span = source ? `${source.start}:${source.end}` : candidate.span;
      const count = groupCounts.get(candidate.groupUid);
      const factLabel = count === 1 ? "1 атомарный факт" : `${count} атомарных ${count < 5 ? "факта" : "фактов"}`;
      group.textContent = `ИСХОДНЫЙ ФРАГМЕНТ · ${span} · ${factLabel}`;
      if (source) group.title = source.text;
      children.push(group); previousGroup = candidate.groupUid;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `candidate-card is-${candidate.status} ${index === state.selectedCandidate ? "is-selected" : ""}`;
    button.innerHTML = `<span class="predicate">${candidate.predicate}</span><span class="candidate-copy"><b></b><code></code></span><span class="confidence"></span>`;
    $(".predicate", button).textContent = `${candidate.sectionHint} · ${candidate.predicate}`;
    $("b", button).textContent = candidate.assertion;
    $("code", button).textContent = `Hyperedge template · ${candidate.template}`;
    $(".confidence", button).textContent = `${Math.round(candidate.confidence * 100)}% · span ${candidate.span} · ${candidate.status}`;
    button.addEventListener("click", () => { state.selectedCandidate = index; renderCandidates(); });
    children.push(button);
  });
  if (state.rejectionLog.length) {
    const heading = document.createElement("div"); heading.className = "candidate-group-label"; heading.textContent = `REJECTION LOG · ${state.rejectionLog.length}`; children.push(heading);
    state.rejectionLog.forEach((rejection) => {
      const row = document.createElement("div"); row.className = "candidate-rejection";
      const code = document.createElement("b"); code.textContent = rejection.reason;
      const message = document.createElement("span"); message.textContent = rejection.message || rejection.reason;
      row.append(code, message); children.push(row);
    });
  }
  if (state.coverageWarnings.length) {
    const heading = document.createElement("div"); heading.className = "candidate-group-label"; heading.textContent = `ТРЕБУЮТ ПРОВЕРКИ · ${state.coverageWarnings.length}`; children.push(heading);
    state.coverageWarnings.forEach((warning) => {
      const row = document.createElement("div"); row.className = "candidate-rejection is-warning";
      const code = document.createElement("b"); code.textContent = `${warning.source_start}:${warning.source_end}`;
      const message = document.createElement("span"); message.textContent = `Факты не извлечены: «${warning.exact_text}»`;
      row.append(code, message); children.push(row);
    });
  }
  list.replaceChildren(...children);
  $("#candidate-count").textContent = `${state.candidates.length} candidates`;
  const selected = state.candidates[state.selectedCandidate];
  if (!selected) {
    $("#candidate-title").textContent = "Нет выбранного кандидата";
    $("#candidate-span").textContent = "source span —";
    $("#candidate-quote").textContent = "Сначала запросите серверный preview документа.";
    $$(".check-list input").forEach((input) => { input.checked = false; });
    $("#binding-list").replaceChildren();
    $("#candidate-section").disabled = true; $("#section-reason").textContent = "";
    $("#reject-candidate").disabled = true; $("#admit-candidate").disabled = true;
    return;
  }
  $("#candidate-title").textContent = `${selected.predicate} · выбранный кандидат`;
  $$(".check-list input").forEach((input) => { input.checked = true; });
  $("#candidate-span").textContent = `source span ${selected.span} · confidence ${selected.confidence.toFixed(2)}`;
  $("#candidate-quote").textContent = selected.sourceQuote;
  const pending = selected.status === "pending";
  $("#candidate-section").value = selected.sectionHint;
  $("#candidate-section").disabled = !pending;
  $("#section-reason").textContent = `${Math.round(selected.sectionConfidence * 100)}% · ${selected.sectionReason}`;
  $("#reject-candidate").disabled = !pending; $("#admit-candidate").disabled = !pending;
  $("#binding-list").replaceChildren(...Object.entries(selected.bindings).map(([role, value]) => {
    const row = document.createElement("div"); row.className = "binding-row";
    const marker = document.createElement("span"); marker.textContent = role[0]; marker.title = role;
    const label = document.createElement("b"); label.textContent = `${role} · ${value}`;
    row.append(marker, label); return row;
  }));
}

async function decideSelectedCandidate(decision, event) {
  event?.preventDefault();
  const candidate = state.candidates[state.selectedCandidate];
  if (!state.previewUid || !candidate || candidate.status !== "pending") return;
  const admit = $("#admit-candidate"); const reject = $("#reject-candidate"); admit.disabled = true; reject.disabled = true;
  try {
    const result = await api("/api/v1/ingestions/decision", { method: "POST", body: JSON.stringify({ preview_uid: state.previewUid, candidate_uid: candidate.candidateUid, decision, section_override: candidate.sectionHint }) });
    candidate.status = result.status;
    const admitted = state.candidates.filter((item) => item.status === "admitted").length;
    const rejected = state.candidates.filter((item) => item.status === "rejected").length;
    const pending = state.candidates.length - admitted - rejected;
    $("#admission-result").textContent = `${admitted} admitted · ${rejected} rejected · ${pending} pending`;
    if (result.accepted.length) { await loadMemoryMap(); toast(`Факт записан в AH · rev ${result.memory_revision}`); }
    else toast("Кандидат отклонён");
    renderCandidates();
  } catch (error) {
    $("#admission-result").textContent = `validation error · ${error.message}`;
    renderCandidates();
  }
}

function percent(value) { return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—"; }

async function runEvaluation() {
  const button = $("#run-evaluation"); button.disabled = true; button.textContent = "Считаем…";
  try {
    const result = await api("/api/v1/evaluations", { method: "POST", body: "{}" });
    const metrics = result.metrics;
    $("#conformance-score").textContent = result.fixtures.rabbit.conformant ? `${result.fixtures.rabbit.accepted}/${result.fixtures.rabbit.facts}` : "FAIL";
    const values = [
      [percent(metrics.M1.normalized_required_roles_weighted_f1 ?? metrics.M1.normalized_weighted_f1), "M1 · SUBJECT / OBJECT / LOCATION"],
      [percent(metrics.M2.explain_score), metrics.M2.trace_pass_rate != null ? `trace pass ${percent(metrics.M2.trace_pass_rate)}` : (metrics.M2.trace_complete ? "trace complete" : "trace incomplete")],
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
  $("#preview-candidates").addEventListener("click", previewCandidates);
  $("#candidate-section").addEventListener("change", (event) => {
    const candidate = state.candidates[state.selectedCandidate];
    if (!candidate || candidate.status !== "pending") return;
    candidate.sectionHint = event.target.value; candidate.sectionConfidence = 1; candidate.sectionReason = "ручное исправление перед допуском";
    renderCandidates();
  });
  $("#ingestion-form").addEventListener("submit", (event) => decideSelectedCandidate("admit", event));
  $("#reject-candidate").addEventListener("click", (event) => decideSelectedCandidate("reject", event));
  $("#run-evaluation").addEventListener("click", runEvaluation);
  $("#memory-refresh").addEventListener("click", refreshMemoryView);
  $("#memory-download").addEventListener("click", downloadMemoryDump);
  $("#memory-search").addEventListener("input", renderMemoryGraph);
  $$("[data-memory-filter]").forEach((button) => button.addEventListener("click", () => {
    state.memoryFilter = button.dataset.memoryFilter;
    $$("[data-memory-filter]").forEach((item) => item.classList.toggle("is-active", item === button));
    renderMemoryGraph();
  }));
  $("#memory-canvas").addEventListener("click", (event) => {
    const rect = event.currentTarget.getBoundingClientRect(); const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    const nearest = state.memoryPositions.map((position) => ({ ...position, distance: Math.hypot(position.x - x, position.y - y) })).sort((a, b) => a.distance - b.distance)[0];
    if (nearest && nearest.distance <= Math.max(12, nearest.radius + 5)) renderMemoryInspector(nearest.uid);
  });
  $("#memory-canvas").addEventListener("mousemove", (event) => {
    const rect = event.currentTarget.getBoundingClientRect(); const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    event.currentTarget.style.cursor = state.memoryPositions.some((position) => Math.hypot(position.x - x, position.y - y) <= Math.max(10, position.radius + 4)) ? "pointer" : "default";
  });
  $("#gc-preview").addEventListener("click", async () => {
    try { const result = await api("/api/v1/gc/preview", { method: "POST", body: "{}" }); $("#gc-output").textContent = `${result.orphan_count_before} collectible · token ${result.preview_token}`; }
    catch (error) { $("#gc-output").textContent = error.message; }
  });
  window.addEventListener("hashchange", () => { const next = location.hash.slice(1); if (["observatory", "ingestion", "evaluation", "memory"].includes(next)) showView(next, false); });
  window.addEventListener("resize", () => { cancelAnimationFrame(state.resizeFrame); state.resizeFrame = requestAnimationFrame(() => { renderGraph(); renderMemoryGraph(); }); });
}

async function boot() {
  bindEvents();
  renderCandidates();
  const initialView = location.hash.slice(1);
  if (["observatory", "ingestion", "evaluation", "memory"].includes(initialView)) showView(initialView, false);
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
