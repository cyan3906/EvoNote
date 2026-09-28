const state = {
  token: localStorage.getItem("evorag_token") || "",
  lastPayload: null,
};

const dom = {
  authForm: document.querySelector("#auth-form"),
  password: document.querySelector("#password"),
  authState: document.querySelector("#auth-state"),
  knowledgeText: document.querySelector("#knowledge-text"),
  extractBtn: document.querySelector("#extract-btn"),
  clearBtn: document.querySelector("#clear-btn"),
  statusLine: document.querySelector("#status-line"),
  summaryStrip: document.querySelector("#summary-strip"),
  blockCount: document.querySelector("#block-count"),
  entityCount: document.querySelector("#entity-count"),
  blockOutput: document.querySelector("#block-output"),
  entityOutput: document.querySelector("#entity-output"),
  rawJson: document.querySelector("#raw-json"),
};

const ATTRIBUTE_ORDER = ["definition", "purpose", "core_idea", "mechanism", "components", "constraints", "related"];
const ATTRIBUTE_TITLES = {
  definition: "定义",
  purpose: "目的",
  core_idea: "核心思想",
  mechanism: "机制",
  components: "组成",
  constraints: "约束",
  related: "相关",
};

updateAuthState();
renderEmpty();

dom.authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const password = dom.password.value.trim();
  if (!password) {
    setStatus("请输入登录密码。", "error");
    return;
  }

  try {
    setBusy(dom.authForm.querySelector("button"), true, "登录中");
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    const data = await parseResponse(response);
    state.token = data.access_token;
    localStorage.setItem("evorag_token", state.token);
    dom.password.value = "";
    updateAuthState();
    setStatus("登录成功。", "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(dom.authForm.querySelector("button"), false, "登录");
  }
});

dom.extractBtn.addEventListener("click", runDebugExtract);
dom.clearBtn.addEventListener("click", () => {
  dom.knowledgeText.value = "";
  state.lastPayload = null;
  renderEmpty();
  setStatus("", "");
});

async function runDebugExtract() {
  const text = dom.knowledgeText.value.trim();
  if (!text) {
    setStatus("请输入要测试的文本。", "error");
    return;
  }

  const startedAt = performance.now();
  try {
    setBusy(dom.extractBtn, true, "运行中");
    setStatus("正在运行 Block 切分和实体抽取...", "");
    const payload = await apiPost("/api/evorag/debug/extract", { text });
    state.lastPayload = payload;
    renderPayload(payload);
    setStatus(`完成，耗时 ${formatDuration(startedAt)}。`, "ok");
  } catch (error) {
    setStatus(`${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.extractBtn, false, "运行抽取");
  }
}

async function apiPost(url, payload) {
  if (!state.token) {
    throw new Error("请先登录。");
  }

  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${state.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

async function parseResponse(response) {
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(text);
    }
  }
  if (!response.ok) {
    throw new Error(data.detail || data.message || `请求失败：${response.status}`);
  }
  return data;
}

function renderEmpty() {
  dom.summaryStrip.replaceChildren();
  dom.blockOutput.replaceChildren(emptyState("暂无 Block 切分结果"));
  dom.entityOutput.replaceChildren(emptyState("暂无实体抽取结果"));
  dom.blockCount.textContent = "0";
  dom.entityCount.textContent = "0";
  dom.rawJson.textContent = "";
}

function renderPayload(payload) {
  const blocks = payload.block_split?.blocks || [];
  const entityBlocks = payload.entity_extraction?.blocks || [];
  const entityTotal = entityBlocks.reduce((total, block) => total + (block.entities?.length || 0), 0);

  dom.blockCount.textContent = `${blocks.length} 个 Block`;
  dom.entityCount.textContent = `${entityTotal} 个实体`;
  dom.rawJson.textContent = JSON.stringify(payload, null, 2);

  renderSummary(payload, blocks.length, entityTotal);
  renderBlocks(blocks);
  renderEntityBlocks(entityBlocks);
}

function renderSummary(payload, blockCount, entityCount) {
  const timings = payload.timings || {};
  const items = [
    ["输入字符", textLength(payload.input_text)],
    ["Block", blockCount],
    ["实体", entityCount],
    ["总耗时", typeof timings.total_ms === "number" ? formatMilliseconds(timings.total_ms) : "-"],
  ];

  dom.summaryStrip.replaceChildren();
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "summary-item";
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = String(value);
    item.append(labelElement, valueElement);
    dom.summaryStrip.append(item);
  }
}

function renderBlocks(blocks) {
  dom.blockOutput.replaceChildren();
  if (!blocks.length) {
    dom.blockOutput.append(emptyState("暂无 Block 切分结果"));
    return;
  }

  for (const block of blocks) {
    dom.blockOutput.append(blockItem(block));
  }
}

function blockItem(block) {
  const item = document.createElement("article");
  item.className = "block-item";

  const head = document.createElement("div");
  head.className = "item-head";
  const title = document.createElement("strong");
  title.textContent = `#${Number(block.block_index ?? 0) + 1} ${block.heading || block.anchor_entity || "未命名 Block"}`;
  const size = document.createElement("span");
  size.textContent = `${textLength(block.l1_text)} 字`;
  head.append(title, size);
  item.append(head);

  const meta = metaRow([
    block.anchor_entity ? `锚点：${block.anchor_entity}` : "",
    block.split_reason ? `原因：${block.split_reason}` : "",
    typeof block.anchor_confidence === "number" ? `置信度：${formatScore(block.anchor_confidence)}` : "",
    block.candidate_entities?.length ? `候选：${block.candidate_entities.join("、")}` : "",
    Number(block.chunk_count || 1) > 1 ? `切片：${Number(block.chunk_index || 0) + 1}/${block.chunk_count}` : "",
  ]);
  if (meta) {
    item.append(meta);
  }

  const text = document.createElement("pre");
  text.className = "block-text";
  text.textContent = block.l1_text || "";
  item.append(text);
  return item;
}

function renderEntityBlocks(blocks) {
  dom.entityOutput.replaceChildren();
  if (!blocks.length) {
    dom.entityOutput.append(emptyState("暂无实体抽取结果"));
    return;
  }

  for (const block of blocks) {
    for (const entity of block.entities || []) {
      dom.entityOutput.append(entityItem(entity, block));
    }
    if (!(block.entities || []).length || block.warnings?.length) {
      dom.entityOutput.append(blockWarningItem(block));
    }
  }
}

function entityItem(entity, block) {
  const item = document.createElement("article");
  item.className = "entity-item";

  const head = document.createElement("div");
  head.className = "item-head";
  const title = document.createElement("strong");
  title.textContent = entity.name || "未命名实体";
  const meta = document.createElement("span");
  meta.textContent = [`Block #${Number(block.block_index ?? 0) + 1}`, entity.entity_type || "concept", scoreLabel(entity)].filter(Boolean).join(" · ");
  head.append(title, meta);
  item.append(head);

  const body = document.createElement("div");
  body.className = "entity-body";

  if (entity.identity_description) {
    const identity = document.createElement("p");
    identity.className = "identity";
    identity.textContent = entity.identity_description;
    body.append(identity);
  }

  const scoreGrid = scoreGridElement(entity.admission_score || {});
  if (scoreGrid) {
    body.append(scoreGrid);
  }

  for (const group of attributeGroups(entity.attributes || {})) {
    body.append(attributeGroup(group));
  }

  item.append(body);
  return item;
}

function blockWarningItem(block) {
  const item = document.createElement("article");
  item.className = "entity-item";
  const head = document.createElement("div");
  head.className = "item-head";
  const title = document.createElement("strong");
  title.textContent = `Block #${Number(block.block_index ?? 0) + 1}`;
  const meta = document.createElement("span");
  meta.textContent = block.entities?.length ? "warning" : "无实体";
  head.append(title, meta);
  item.append(head);

  const body = document.createElement("div");
  body.className = "entity-body";
  if (!(block.entities || []).length) {
    body.append(emptyState("这个 Block 没有抽到实体"));
  }
  if (block.warnings?.length) {
    const warnings = document.createElement("div");
    warnings.className = "warning-list";
    warnings.textContent = block.warnings.join("；");
    body.append(warnings);
  }
  item.append(body);
  return item;
}

function scoreGridElement(score) {
  const entries = [
    ["aggregate", score.aggregate_score],
    ["definable", score.definable],
    ["query", score.query_entry],
    ["scope", score.independent_scope],
    ["relations", score.stable_relations],
    ["key", score.key_sentence],
  ].filter(([, value]) => typeof value === "number");

  if (!entries.length) {
    return null;
  }

  const grid = document.createElement("div");
  grid.className = "score-grid";
  for (const [label, value] of entries) {
    const cell = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = label;
    const number = document.createElement("strong");
    number.textContent = formatScore(value);
    cell.append(name, number);
    grid.append(cell);
  }
  return grid;
}

function attributeGroups(attributes) {
  const groups = [];
  for (const type of ATTRIBUTE_ORDER) {
    const values = (attributes[type] || []).filter((item) => item?.value);
    if (values.length) {
      groups.push({ type, values });
    }
  }
  return groups;
}

function attributeGroup(group) {
  const section = document.createElement("section");
  section.className = "attribute-group";
  const title = document.createElement("h3");
  title.textContent = ATTRIBUTE_TITLES[group.type] || group.type;
  section.append(title);

  for (const value of group.values) {
    const row = document.createElement("div");
    row.className = "attribute-row";
    const text = document.createElement("p");
    text.textContent = value.value || "";
    row.append(text);
    const meta = [];
    if (value.evidence) {
      meta.push(`证据：${value.evidence}`);
    }
    if (typeof value.confidence === "number") {
      meta.push(`置信度 ${formatScore(value.confidence)}`);
    }
    if (meta.length) {
      const metaElement = document.createElement("span");
      metaElement.textContent = meta.join(" · ");
      row.append(metaElement);
    }
    section.append(row);
  }

  return section;
}

function metaRow(items) {
  const values = items.filter(Boolean);
  if (!values.length) {
    return null;
  }
  const row = document.createElement("div");
  row.className = "meta-row";
  for (const value of values) {
    const chip = document.createElement("span");
    chip.textContent = value;
    row.append(chip);
  }
  return row;
}

function emptyState(text) {
  const element = document.createElement("div");
  element.className = "empty-state";
  element.textContent = text;
  return element;
}

function setStatus(text, kind) {
  dom.statusLine.textContent = text;
  dom.statusLine.classList.toggle("ok", kind === "ok");
  dom.statusLine.classList.toggle("error", kind === "error");
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = label;
}

function updateAuthState() {
  dom.authState.textContent = state.token ? "已登录" : "未登录";
  dom.authState.classList.toggle("ready", Boolean(state.token));
}

function scoreLabel(entity) {
  const score = entity.admission_score?.aggregate_score;
  return typeof score === "number" ? `准入 ${formatScore(score)}` : "";
}

function textLength(text) {
  return String(text || "").length;
}

function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

function formatDuration(startedAt) {
  return formatMilliseconds(Math.max(0, performance.now() - startedAt));
}

function formatMilliseconds(value) {
  const number = Number(value || 0);
  if (number < 1000) {
    return `${Math.round(number)} ms`;
  }
  return `${(number / 1000).toFixed(2)} s`;
}
