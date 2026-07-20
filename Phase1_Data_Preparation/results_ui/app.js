const state = {
  runs: [],
  records: [],
  filtered: [],
  activeRun: "",
  activeKey: "",
  activeDetail: null,
  status: "all",
  search: "",
  refreshing: false,
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  runSelect: $("#runSelect"),
  questionSelect: $("#questionSelect"),
  questionSearch: $("#questionSearch"),
  questionList: $("#questionList"),
  visibleCount: $("#visibleCount"),
  detailContent: $("#detailContent"),
  emptyState: $("#emptyState"),
  toast: $("#toast"),
};

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined && text !== null) item.textContent = String(text);
  return item;
}

function valueOrDash(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Oui" : "Non";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function formatDate(value) {
  if (!value) return "Date inconnue";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("fr-CA", { dateStyle: "medium", timeStyle: "medium" }).format(date);
}

function prettyLabel(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function keyFor(record) {
  return `${record.status}:${record.scenario_id}`;
}

async function api(path) {
  const response = await fetch(path, { cache: "no-store" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Erreur HTTP ${response.status}`);
  return body;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.setTimeout(() => elements.toast.classList.remove("is-visible"), 2200);
}

function showError(error) {
  elements.questionList.replaceChildren(node("div", "error-box", error.message || String(error)));
}

function currentRun() {
  return state.runs.find((run) => run.run_id === state.activeRun);
}

function updateMetrics() {
  const run = currentRun();
  $("#acceptedMetric").textContent = run?.accepted ?? "—";
  $("#rejectedMetric").textContent = run?.rejected ?? "—";
  $("#rateMetric").textContent = run ? `${(run.acceptance_rate * 100).toFixed(1)} %` : "—";
  $("#costMetric").textContent = run ? `$${Number(run.cost_usd || 0).toFixed(4)}` : "—";
  $("#targetMetric").textContent = `objectif ${run?.target_accepted ?? "—"}`;
  $("#scenarioMetric").textContent = `${run?.scenarios ?? "—"} scénarios traités`;
  $("#modelMetric").textContent = `modèle ${run?.model || "—"}`;
  $("#callsMetric").textContent = `${run?.api_calls ?? "—"} appels API`;
  $("#runMeta").textContent = run ? `Mis à jour ${formatDate(run.updated_at)}` : "—";
}

async function loadRuns({ preserve = true } = {}) {
  const previous = preserve ? state.activeRun : "";
  const payload = await api("/api/runs");
  state.runs = payload.runs || [];
  state.activeRun = state.runs.some((run) => run.run_id === previous) ? previous : (state.runs[0]?.run_id || "");
  elements.runSelect.replaceChildren();
  if (!state.runs.length) {
    const option = node("option", "", "Aucun run détecté");
    option.value = "";
    elements.runSelect.append(option);
  } else {
    for (const run of state.runs) {
      const option = node("option", "", `${run.run_id} · ${run.accepted} ✓ / ${run.rejected} ×`);
      option.value = run.run_id;
      option.selected = run.run_id === state.activeRun;
      elements.runSelect.append(option);
    }
  }
  updateMetrics();
}

async function loadRecords({ preserve = true } = {}) {
  if (!state.activeRun) {
    state.records = [];
    applyFilters();
    return;
  }
  const previous = preserve ? state.activeKey : "";
  const payload = await api(`/api/records?run=${encodeURIComponent(state.activeRun)}`);
  state.records = payload.records || [];
  state.activeKey = state.records.some((record) => keyFor(record) === previous) ? previous : "";
  applyFilters();
  if (state.activeKey) await selectRecord(state.activeKey, { scroll: false });
}

function applyFilters() {
  const query = state.search.trim().toLocaleLowerCase("fr");
  state.filtered = state.records.filter((record) => {
    const statusMatch = state.status === "all" || record.status === state.status;
    const haystack = `${record.question} ${record.request_type || ""} ${record.stage || ""} ${(record.reasons || []).join(" ")}`.toLocaleLowerCase("fr");
    return statusMatch && (!query || haystack.includes(query));
  });
  renderQuestionControls();
}

function renderQuestionControls() {
  elements.visibleCount.textContent = state.filtered.length;
  elements.questionSelect.disabled = !state.filtered.length;
  elements.questionSelect.replaceChildren();
  const placeholder = node("option", "", state.filtered.length ? "Choisir une question…" : "Aucune question");
  placeholder.value = "";
  elements.questionSelect.append(placeholder);

  for (const [index, record] of state.filtered.entries()) {
    const option = node("option", "", `${index + 1}. ${record.status === "accepted" ? "✓" : "×"} ${record.question}`);
    option.value = keyFor(record);
    option.selected = option.value === state.activeKey;
    elements.questionSelect.append(option);
  }

  elements.questionList.replaceChildren();
  if (!state.filtered.length) {
    elements.questionList.append(node("div", "empty-list", state.records.length ? "Aucune question ne correspond aux filtres." : "Ce run ne contient encore aucun résultat."));
    return;
  }

  for (const record of state.filtered) {
    const button = node("button", `question-item${keyFor(record) === state.activeKey ? " is-active" : ""}`);
    button.type = "button";
    button.dataset.key = keyFor(record);
    button.append(node("p", "", record.question));
    const meta = node("div", "question-item-meta");
    meta.append(node("span", `status-dot is-${record.status}`, record.status === "accepted" ? "Acceptée" : "Rejetée"));
    meta.append(node("span", "", record.request_type ? prettyLabel(record.request_type) : `#${record.order + 1}`));
    button.append(meta);
    button.addEventListener("click", () => selectRecord(keyFor(record), { scroll: true }));
    elements.questionList.append(button);
  }
}

async function selectRecord(key, { scroll = false } = {}) {
  const record = state.records.find((item) => keyFor(item) === key);
  if (!record) return;
  state.activeKey = key;
  renderQuestionControls();
  elements.questionSelect.value = key;
  elements.emptyState.hidden = true;
  elements.detailContent.hidden = false;
  $("#detailQuestion").textContent = "Chargement…";
  try {
    const params = new URLSearchParams({ run: state.activeRun, status: record.status, id: record.scenario_id });
    state.activeDetail = await api(`/api/detail?${params}`);
    renderDetail(state.activeDetail);
    if (scroll && window.innerWidth < 760) $("#detailPanel").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    $("#detailQuestion").textContent = "Impossible de charger cette donnée";
    showToast(error.message);
  }
}

function badge(text, type = "neutral") {
  return node("span", `pill ${type}`, text);
}

function miniCard(label, value) {
  const card = node("div", "mini-card");
  card.append(node("span", "", label), node("strong", "", valueOrDash(value)));
  return card;
}

function card(title, index, countText = "") {
  const article = node("article", "data-card");
  const heading = node("div", "card-title-row");
  const h3 = node("h3");
  h3.append(node("i", "card-index", index), document.createTextNode(title));
  heading.append(h3);
  if (countText) heading.append(node("small", "", countText));
  article.append(heading);
  return article;
}

function renderKeyValues(target, object, omitted = []) {
  const grid = node("dl", "key-value-grid");
  const entries = Object.entries(object || {}).filter(([key, value]) => !omitted.includes(key) && value !== null && value !== undefined && value !== "");
  if (!entries.length) {
    target.append(node("div", "empty-list", "Aucune donnée disponible."));
    return;
  }
  for (const [key, value] of entries) {
    const item = node("div", "key-value");
    item.append(node("dt", "", prettyLabel(key)), node("dd", "", valueOrDash(value)));
    grid.append(item);
  }
  target.append(grid);
}

function renderScores(quality) {
  const article = card("Qualité et validation", "QA");
  const entries = Object.entries(quality || {});
  if (!entries.length) {
    article.append(node("div", "empty-list", "Aucun score disponible pour cette donnée."));
    return article;
  }
  for (const [key, value] of entries) {
    if (typeof value === "number" && value >= 0 && value <= 1) {
      const row = node("div", "score-row");
      row.append(node("span", "score-label", prettyLabel(key)));
      const track = node("div", "score-track");
      const fill = node("div", "score-fill");
      fill.style.width = `${Math.round(value * 100)}%`;
      track.append(fill);
      row.append(track, node("span", "score-value", value.toFixed(2)));
      article.append(row);
    } else {
      const row = node("div", "key-value");
      row.append(node("dt", "", prettyLabel(key)), node("dd", "", valueOrDash(value)));
      article.append(row);
    }
  }
  return article;
}

function renderDetail(detail) {
  const trajectory = detail.trajectory || {};
  $("#detailQuestion").textContent = detail.question;
  $("#detailId").textContent = `${detail.run_id} / ${detail.scenario_id}`;
  const badges = $("#detailBadges");
  badges.replaceChildren(
    badge(detail.status === "accepted" ? "Acceptée" : "Rejetée", detail.status),
    ...(detail.request_type ? [badge(prettyLabel(detail.request_type))] : []),
    ...(detail.stage ? [badge(`Étape · ${prettyLabel(detail.stage)}`)] : []),
  );

  const overview = $("#overviewGrid");
  overview.replaceChildren(
    miniCard("Juridiction attendue", trajectory.expected_jurisdiction || detail.scenario?.expected_jurisdiction),
    miniCard("Juridiction résolue", trajectory.resolved_jurisdiction),
    miniCard("Domaine", trajectory.legal_domain || detail.scenario?.legal_domain),
    miniCard("Horodatage", formatDate(detail.timestamp)),
  );

  const rejectionTarget = $("#rejectionSection");
  rejectionTarget.replaceChildren();
  if (detail.status === "rejected") {
    const rejection = card("Motifs du rejet", "ERR", `${detail.reasons.length} motif(s)`);
    rejection.classList.add("rejection-card");
    const list = node("ul", "reason-list");
    for (const reason of detail.reasons.length ? detail.reasons : ["Aucun motif enregistré."]) list.append(node("li", "", reason));
    rejection.append(list);
    rejectionTarget.append(rejection);
  }

  const scenarioTarget = $("#scenarioSection");
  scenarioTarget.replaceChildren();
  const scenario = card("Scénario généré", "SCN");
  renderKeyValues(scenario, detail.scenario || {}, ["user_query", "question"]);
  scenarioTarget.append(scenario);

  const messagesTarget = $("#messagesSection");
  messagesTarget.replaceChildren();
  const messages = Array.isArray(trajectory.messages) ? trajectory.messages : [];
  const messagesCard = card("Conversation", "MSG", `${messages.length} message(s)`);
  const messageStack = node("div", "stack");
  for (const [index, message] of messages.entries()) {
    const role = message.role || "inconnu";
    const item = node("div", `subcard message-${role}`);
    const header = node("div", "subcard-header");
    header.append(node("strong", "", prettyLabel(role)), node("span", "", `Message ${index + 1}${message.name ? ` · ${message.name}` : ""}`));
    item.append(header, node("p", "", valueOrDash(message.content)));
    messageStack.append(item);
  }
  if (!messages.length) messageStack.append(node("div", "empty-list", "La trajectoire n’a pas été produite avant ce rejet."));
  messagesCard.append(messageStack);
  messagesTarget.append(messagesCard);

  const toolsTarget = $("#toolsSection");
  toolsTarget.replaceChildren();
  const tools = Array.isArray(trajectory.tool_trace) ? trajectory.tool_trace : [];
  const toolsCard = card("Appels d’outils", "MCP", `${tools.length} appel(s)`);
  const toolStack = node("div", "stack");
  for (const [index, tool] of tools.entries()) toolStack.append(renderTool(tool, index));
  if (!tools.length) toolStack.append(node("div", "empty-list", "Aucun outil appelé dans cette trajectoire."));
  toolsCard.append(toolStack);
  toolsTarget.append(toolsCard);

  const groundingTarget = $("#groundingSection");
  groundingTarget.replaceChildren();
  const grounding = Array.isArray(trajectory.grounding) ? trajectory.grounding : [];
  const groundingCard = card("Ancrage et sources", "SRC", `${grounding.length} élément(s)`);
  const groundingStack = node("div", "stack");
  for (const item of grounding) {
    const source = node("div", "subcard");
    const header = node("div", "subcard-header");
    header.append(node("strong", "", item.tool_name || "Source"), node("span", "", item.content_hash ? `hash ${item.content_hash.slice(0, 12)}…` : ""));
    source.append(header);
    const links = renderLinks(item.source_urls || []);
    if (links) source.append(links);
    if (item.citations?.length) source.append(node("p", "", item.citations.join("\n")));
    groundingStack.append(source);
  }
  if (!grounding.length) groundingStack.append(node("div", "empty-list", "Aucune source d’ancrage enregistrée."));
  groundingCard.append(groundingStack);
  groundingTarget.append(groundingCard);

  const metadataTarget = $("#metadataSection");
  metadataTarget.replaceChildren();
  metadataTarget.append(renderScores(trajectory.quality));
  const metadata = card("Métadonnées de génération", "META");
  renderKeyValues(metadata, trajectory.generation_metadata || {});
  metadataTarget.append(metadata);

  $("#rawJson").textContent = JSON.stringify(detail, null, 2);
}

function renderTool(tool, index) {
  const item = node("div", "subcard tool-card");
  const header = node("div", "subcard-header");
  const name = node("strong", "tool-name", `${index + 1}. ${tool.tool_name || "outil inconnu"}`);
  const stats = node("div", "tool-stats");
  stats.append(
    node("span", "tool-stat", tool.ok === false ? "Échec" : "OK"),
    node("span", "tool-stat", `${tool.latency_ms ?? "—"} ms`),
    node("span", "tool-stat", tool.server || "serveur —"),
  );
  header.append(name, stats);
  item.append(header);

  const body = node("div", "tool-body");
  const args = node("div", "tool-block");
  args.append(node("h4", "", "Arguments"), node("pre", "json-block", JSON.stringify(tool.arguments || {}, null, 2)));
  const response = node("div", "tool-block");
  response.append(node("h4", "", "Réponse normalisée"), node("pre", "json-block", JSON.stringify(tool.normalized_response ?? tool.raw_response ?? null, null, 2)));
  const links = renderLinks(tool.source_urls || []);
  if (links) response.append(links);
  body.append(args, response);
  item.append(body);
  return item;
}

function renderLinks(urls) {
  const safeUrls = urls.filter((value) => typeof value === "string" && /^https?:\/\//i.test(value));
  if (!safeUrls.length) return null;
  const list = node("div", "source-list");
  for (const url of safeUrls) {
    const link = node("a", "", url);
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    list.append(link);
  }
  return list;
}

async function refresh({ quiet = false } = {}) {
  if (state.refreshing) return;
  state.refreshing = true;
  const button = $("#refreshButton");
  button.disabled = true;
  try {
    await loadRuns({ preserve: true });
    await loadRecords({ preserve: true });
    $("#lastRefresh").textContent = `Dernière lecture : ${formatDate(new Date().toISOString())}`;
    if (!quiet) showToast("Données actualisées");
  } catch (error) {
    showError(error);
    if (!quiet) showToast(error.message);
  } finally {
    state.refreshing = false;
    button.disabled = false;
  }
}

elements.runSelect.addEventListener("change", async (event) => {
  state.activeRun = event.target.value;
  state.activeKey = "";
  state.activeDetail = null;
  updateMetrics();
  elements.detailContent.hidden = true;
  elements.emptyState.hidden = false;
  await loadRecords({ preserve: false });
});

elements.questionSelect.addEventListener("change", (event) => {
  if (event.target.value) selectRecord(event.target.value, { scroll: true });
});

elements.questionSearch.addEventListener("input", (event) => {
  state.search = event.target.value;
  applyFilters();
});

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".segment").forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");
    state.status = button.dataset.status;
    applyFilters();
  });
});

$("#refreshButton").addEventListener("click", () => refresh({ quiet: false }));
$("#copyButton").addEventListener("click", async () => {
  if (!state.activeDetail) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(state.activeDetail, null, 2));
    showToast("JSON copié");
  } catch {
    showToast("Copie refusée par le navigateur");
  }
});

refresh({ quiet: true });
window.setInterval(() => refresh({ quiet: true }), 5000);
