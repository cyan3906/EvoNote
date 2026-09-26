const state = {
  token: localStorage.getItem("evorag_token") || "",
  lastAnswer: "",
};

const dom = {
  authForm: document.querySelector("#auth-form"),
  password: document.querySelector("#password"),
  authState: document.querySelector("#auth-state"),
  extractBtn: document.querySelector("#extract-btn"),
  ingestBtn: document.querySelector("#ingest-btn"),
  queryBtn: document.querySelector("#query-btn"),
  indexSearchBtn: document.querySelector("#index-search-btn"),
  copyAnswerBtn: document.querySelector("#copy-answer-btn"),
  knowledgeText: document.querySelector("#knowledge-text"),
  entityQuery: document.querySelector("#entity-query"),
  topK: document.querySelector("#top-k"),
  ingestStatus: document.querySelector("#ingest-status"),
  queryStatus: document.querySelector("#query-status"),
  indexSearchStatus: document.querySelector("#index-search-status"),
  extractedEntities: document.querySelector("#extracted-entities"),
  preprocessDetails: ensurePreprocessDetails(),
  answer: document.querySelector("#answer"),
  retrievedEntities: document.querySelector("#retrieved-entities"),
  graphEdges: document.querySelector("#graph-edges"),
  warnings: document.querySelector("#warnings"),
  esResults: document.querySelector("#es-results"),
  milvusResults: document.querySelector("#milvus-results"),
  fusedResults: document.querySelector("#fused-results"),
  backendStatus: document.querySelector("#backend-status"),
  indexTimings: document.querySelector("#index-timings"),
  indexWarnings: document.querySelector("#index-warnings"),
  workspaceId: document.querySelector("#workspace-id"),
  projectId: document.querySelector("#project-id"),
  collectionId: document.querySelector("#collection-id"),
  domain: document.querySelector("#domain"),
};

updateAuthState();

function ensurePreprocessDetails() {
  const existing = document.querySelector("#preprocess-details");
  if (existing) {
    return existing;
  }

  const element = document.createElement("div");
  element.id = "preprocess-details";
  element.className = "preprocess-details";
  document.querySelector("#extracted-entities")?.after(element);
  return element;
}

dom.authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const password = dom.password.value.trim();
  if (!password) {
    setStatus(dom.ingestStatus, "请输入登录密码。", "error");
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
    setStatus(dom.ingestStatus, "登录成功，可以提交文本。", "ok");
  } catch (error) {
    setStatus(dom.ingestStatus, error.message, "error");
  } finally {
    setBusy(dom.authForm.querySelector("button"), false, "登录");
  }
});

dom.extractBtn.addEventListener("click", async () => {
  const text = dom.knowledgeText.value.trim();
  if (!text) {
    setStatus(dom.ingestStatus, "请输入要测试抽取的知识文本。", "error");
    return;
  }

  const startedAt = performance.now();
  try {
    setBusy(dom.extractBtn, true, "抽取中");
    setStatus(dom.ingestStatus, "正在切分 Block 并抽取实体，不会入库...", "");
    const data = await apiFetch("/api/evorag/extract", {
      text,
      scope: readScope(),
    });
    renderExtractedEntities(data, []);
    const blockCount = data.blocks?.length || 0;
    const entityCount = data.blocks?.reduce((total, block) => total + (block.entities?.length || 0), 0) || 0;
    setStatus(dom.ingestStatus, `抽取完成：${blockCount} 个 Block，${entityCount} 个实体。耗时 ${formatDuration(startedAt)}。`, "ok");
  } catch (error) {
    setStatus(dom.ingestStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.extractBtn, false, "抽取预览");
  }
});

dom.ingestBtn.addEventListener("click", async () => {
  const text = dom.knowledgeText.value.trim();
  if (!text) {
    setStatus(dom.ingestStatus, "请输入要处理的知识文本。", "error");
    return;
  }

  const startedAt = performance.now();
  try {
    setBusy(dom.ingestBtn, true, "处理中");
    setStatus(dom.ingestStatus, "正在切分 Block、抽取实体并写入实体库...", "");
    const data = await apiFetch("/api/evorag/ingest", {
      text,
      scope: readScope(),
    });
    renderExtractedEntities(data.preprocess, data.ingest_results || []);
    const blockCount = data.preprocess.blocks?.length || 0;
    const entityCount = data.preprocess.blocks?.reduce((total, block) => total + (block.entities?.length || 0), 0) || 0;
    const upsertCount = data.ingest_results?.length || 0;
    setStatus(
      dom.ingestStatus,
      `完成：${blockCount} 个 Block，${entityCount} 个实体，${upsertCount} 个实体写入或合并。耗时 ${formatDuration(startedAt)}。`,
      "ok",
    );
  } catch (error) {
    setStatus(dom.ingestStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.ingestBtn, false, "提交并入库");
  }
});

dom.queryBtn.addEventListener("click", async () => {
  const entity = dom.entityQuery.value.trim();
  if (!entity) {
    setStatus(dom.queryStatus, "请输入要检索的实体。", "error");
    return;
  }

  const startedAt = performance.now();
  try {
    setBusy(dom.queryBtn, true, "生成中");
    setStatus(dom.queryStatus, "正在检索实体、构建依赖图并生成长文本...", "");
    const data = await apiFetch("/api/evorag/query", {
      entity,
      top_k: Number(dom.topK.value || 5),
      scope: readScope(),
    });
    renderQueryResult(data);
    setStatus(
      dom.queryStatus,
      data.root_entity
        ? `完成：根实体 ${data.root_entity.canonical_name}。耗时 ${formatDuration(startedAt)}。`
        : `没有找到匹配实体。耗时 ${formatDuration(startedAt)}。`,
      data.root_entity ? "ok" : "error",
    );
  } catch (error) {
    setStatus(dom.queryStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.queryBtn, false, "生成");
  }
});

dom.indexSearchBtn.addEventListener("click", async () => {
  const entity = dom.entityQuery.value.trim();
  if (!entity) {
    setStatus(dom.indexSearchStatus, "请输入要检索的实体。", "error");
    return;
  }

  const startedAt = performance.now();
  try {
    setBusy(dom.indexSearchBtn, true, "检索中");
    setStatus(dom.indexSearchStatus, "正在分别查询 ES、Milvus 并融合结果...", "");
    const data = await apiFetch("/api/evorag/index-search", {
      entity,
      top_k: Number(dom.topK.value || 5),
      scope: readScope(),
    });
    renderIndexSearchResult(data);
    const total = (data.fused_results || []).length;
    setStatus(dom.indexSearchStatus, `完成：融合结果 ${total} 个。耗时 ${formatDuration(startedAt)}。`, "ok");
  } catch (error) {
    setStatus(dom.indexSearchStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.indexSearchBtn, false, "检索诊断");
  }
});

dom.copyAnswerBtn.addEventListener("click", async () => {
  if (!state.lastAnswer) {
    setStatus(dom.queryStatus, "还没有可复制的生成内容。", "error");
    return;
  }
  await navigator.clipboard.writeText(state.lastAnswer);
  setStatus(dom.queryStatus, "已复制生成内容。", "ok");
});

async function apiFetch(url, payload) {
  if (!state.token) {
    throw new Error("请先登录。");
  }
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${state.token}`,
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

function readScope() {
  return {
    workspace_id: dom.workspaceId.value.trim() || "local",
    project_id: dom.projectId.value.trim() || "evorag",
    collection_id: dom.collectionId.value.trim() || "default",
    domain: dom.domain.value.trim() || "general",
  };
}

function renderExtractedEntities(preprocess, ingestResults = []) {
  const entities = [];
  for (const block of preprocess.blocks || []) {
    for (const entity of block.entities || []) {
      if (!entities.includes(entity.name)) {
        entities.push(entity.name);
      }
    }
  }

  dom.extractedEntities.replaceChildren();
  dom.preprocessDetails.replaceChildren();

  for (const name of entities.slice(0, 32)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "entity-chip";
    button.textContent = name;
    button.addEventListener("click", () => {
      dom.entityQuery.value = name;
      dom.entityQuery.focus();
    });
    dom.extractedEntities.append(button);
  }

  renderPreprocessDetails(preprocess, ingestResults);
}

function renderPreprocessDetails(preprocess, ingestResults = []) {
  const blocks = preprocess.blocks || [];
  if (!blocks.length) {
    dom.preprocessDetails.append(emptyState("暂无 Block 切分结果"));
    return;
  }

  const header = document.createElement("div");
  header.className = "preprocess-head";
  const title = document.createElement("h3");
  title.textContent = "处理明细";
  const summary = document.createElement("span");
  summary.textContent = `${blocks.length} 个 Block / ${countBlockEntities(blocks)} 个实体`;
  header.append(title, summary);
  dom.preprocessDetails.append(header);

  renderPreprocessTimings(preprocess.timings || {});

  for (const blockResult of blocks) {
    dom.preprocessDetails.append(blockCard(blockResult));
  }

  renderIngestResults(ingestResults);
}

function renderPreprocessTimings(timings) {
  const entries = [
    ["启动到首次切分前", timings.startup_to_first_block_split_ms],
    ["实体锚点预切分", timings.entity_anchor_split_ms],
    ["物理二切", timings.physical_chunk_split_ms],
    ["实体抽取", timings.entity_extraction_ms],
    ["预处理总耗时", timings.total_ms],
  ].filter(([, value]) => typeof value === "number");

  if (!entries.length) {
    return;
  }

  const grid = document.createElement("div");
  grid.className = "timing-grid preprocess-timing-grid";
  for (const [label, value] of entries) {
    const item = document.createElement("div");
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = formatMilliseconds(value);
    item.append(labelElement, valueElement);
    grid.append(item);
  }
  dom.preprocessDetails.append(grid);
}

function renderIngestResults(results) {
  if (!results.length) {
    return;
  }

  const section = document.createElement("section");
  section.className = "ingest-result-list";
  const title = document.createElement("h3");
  title.textContent = "入库结果";
  section.append(title);

  for (const result of results) {
    const item = document.createElement("div");
    item.className = "ingest-result-item";
    const name = document.createElement("strong");
    name.textContent = result.canonical_name || `Entity ${result.entity_id}`;
    const meta = document.createElement("span");
    meta.textContent = [
      `ID ${result.entity_id}`,
      result.created ? "新建" : "合并",
      `${result.attribute_count || 0} 条属性`,
      `${result.evidence_count || 0} 条证据`,
    ].join(" · ");
    item.append(name, meta);
    section.append(item);
  }

  dom.preprocessDetails.append(section);
}

function blockCard(blockResult) {
  const block = blockResult.block || {};
  const entities = blockResult.entities || [];
  const details = document.createElement("details");
  details.className = "block-detail";
  details.open = true;

  const summary = document.createElement("summary");
  const title = document.createElement("strong");
  title.textContent = `${formatBlockTitle(block)} · ${entities.length} 个实体`;
  const meta = document.createElement("span");
  meta.textContent = `${textLength(block.l1_text)} 字`;
  summary.append(title, meta);
  details.append(summary);

  const blockMeta = blockMetaPanel(block);
  if (blockMeta) {
    details.append(blockMeta);
  }

  const blockText = document.createElement("pre");
  blockText.className = "block-text";
  blockText.textContent = block.l1_text || "";
  details.append(blockText);

  if (blockResult.warnings?.length) {
    const warningBox = document.createElement("div");
    warningBox.className = "block-warnings";
    for (const warning of blockResult.warnings) {
      const line = document.createElement("div");
      line.textContent = warning;
      warningBox.append(line);
    }
    details.append(warningBox);
  }

  const entityList = document.createElement("div");
  entityList.className = "block-entities";
  if (!entities.length) {
    entityList.append(emptyState("这个 Block 没有抽到实体"));
  } else {
    for (const entity of entities) {
      entityList.append(entityPanel(entity));
    }
  }
  details.append(entityList);
  return details;
}

function blockMetaPanel(block) {
  const items = [
    block.anchor_entity ? `锚点实体：${block.anchor_entity}` : "",
    block.candidate_entities?.length ? `候选实体：${block.candidate_entities.slice(0, 6).join("、")}` : "",
    block.split_reason ? `切分原因：${block.split_reason}` : "",
    Number.isInteger(block.chunk_count) && block.chunk_count > 1
      ? `切片：${(block.chunk_index || 0) + 1}/${block.chunk_count}`
      : "",
    typeof block.anchor_confidence === "number" && block.anchor_confidence > 0
      ? `锚点置信度：${formatScore(block.anchor_confidence)}`
      : "",
  ].filter(Boolean);

  if (!items.length) {
    return null;
  }

  const panel = document.createElement("div");
  panel.className = "block-meta";
  for (const item of items) {
    const span = document.createElement("span");
    span.textContent = item;
    panel.append(span);
  }
  return panel;
}

function entityPanel(entity) {
  const section = document.createElement("section");
  section.className = "entity-detail";

  const head = document.createElement("div");
  head.className = "entity-detail-head";
  const nameButton = document.createElement("button");
  nameButton.type = "button";
  nameButton.className = "entity-name-button";
  nameButton.textContent = entity.name || "未命名实体";
  nameButton.addEventListener("click", () => {
    dom.entityQuery.value = entity.name || "";
    dom.entityQuery.focus();
  });
  const meta = document.createElement("span");
  meta.textContent = [
    entity.entity_type || "concept",
    admissionText(entity.admission_score),
    aliasesText(entity.aliases),
  ]
    .filter(Boolean)
    .join(" · ");
  head.append(nameButton, meta);
  section.append(head);

  if (entity.admission_score?.rationale) {
    const admission = document.createElement("p");
    admission.className = "entity-admission";
    admission.textContent = `准入理由：${entity.admission_score.rationale}`;
    section.append(admission);
  }

  if (entity.identity_description) {
    const description = document.createElement("p");
    description.className = "entity-description";
    description.textContent = entity.identity_description;
    section.append(description);
  }

  const attributes = attributeRows(entity.attributes || {});
  if (!attributes.length) {
    section.append(emptyState("暂无属性"));
    return section;
  }

  const attributeList = document.createElement("div");
  attributeList.className = "attribute-list";
  for (const row of attributes) {
    attributeList.append(attributeItem(row));
  }
  section.append(attributeList);
  return section;
}

function attributeItem(row) {
  const item = document.createElement("div");
  item.className = "attribute-item";

  const label = document.createElement("div");
  label.className = "attribute-label";
  label.textContent = attributeTitle(row.type);
  item.append(label);

  const body = document.createElement("div");
  body.className = "attribute-body";
  const value = document.createElement("p");
  value.textContent = row.value.value || "";
  body.append(value);

  const metaParts = [];
  if (row.value.evidence) {
    metaParts.push(`证据：${row.value.evidence}`);
  }
  if (typeof row.value.confidence === "number") {
    metaParts.push(`置信度 ${formatScore(row.value.confidence)}`);
  }
  if (metaParts.length) {
    const meta = document.createElement("span");
    meta.textContent = metaParts.join(" · ");
    body.append(meta);
  }

  item.append(body);
  return item;
}

function attributeRows(attributes) {
  const rows = [];
  for (const type of ATTRIBUTE_ORDER) {
    for (const value of attributes[type] || []) {
      if (value?.value) {
        rows.push({ type, value });
      }
    }
  }
  return rows;
}

function countBlockEntities(blocks) {
  return blocks.reduce((total, block) => total + (block.entities?.length || 0), 0);
}

function formatBlockTitle(block) {
  const index = Number.isInteger(block.block_index) ? block.block_index + 1 : "?";
  return `Block ${index}${block.heading ? `：${block.heading}` : ""}`;
}

function aliasesText(aliases) {
  const values = aliases || [];
  if (!values.length) {
    return "";
  }
  return `别名 ${values.slice(0, 4).join("、")}`;
}

function admissionText(score) {
  if (!score || typeof score.aggregate_score !== "number") {
    return "";
  }
  return `准入 ${formatScore(score.aggregate_score)}`;
}

function textLength(value) {
  return String(value || "").length;
}

function emptyState(text) {
  const item = document.createElement("div");
  item.className = "empty-state";
  item.textContent = text;
  return item;
}

function renderQueryResult(data) {
  state.lastAnswer = data.answer || "";
  dom.answer.classList.toggle("empty", !state.lastAnswer);
  dom.answer.replaceChildren(...markdownBlocks(state.lastAnswer || "生成结果会显示在这里。"));
  renderEntities(data.retrieved_entities || []);
  renderEdges(data.graph);
  renderWarnings(data.warnings || []);
}

function renderEntities(entities) {
  dom.retrievedEntities.replaceChildren();
  if (!entities.length) {
    dom.retrievedEntities.append(emptyItem("暂无实体"));
    return;
  }
  for (const entity of entities) {
    const item = document.createElement("div");
    item.className = "list-item";
    const title = document.createElement("strong");
    title.textContent = entity.canonical_name;
    const meta = document.createElement("span");
    meta.textContent = `${entity.entity_type || "concept"} · score ${formatScore(entity.score)} · ${entity.source || "unknown"}`;
    item.append(title, meta);
    dom.retrievedEntities.append(item);
  }
}

function renderEdges(graph) {
  dom.graphEdges.replaceChildren();
  if (!graph) {
    dom.graphEdges.append(emptyItem("暂无依赖图"));
    return;
  }
  const entityById = new Map((graph.entities || []).map((entity) => [entity.id, entity]));
  const edges = [...(graph.semantic_edges || []), ...(graph.conditional_edges || [])];
  if (!edges.length) {
    dom.graphEdges.append(emptyItem("暂无依赖边"));
    return;
  }
  for (const edge of edges) {
    const source = entityById.get(edge.source_id)?.canonical_name || edge.source_id;
    const target = entityById.get(edge.target_id)?.canonical_name || edge.target_id;
    const item = document.createElement("div");
    item.className = "list-item";
    const title = document.createElement("strong");
    title.textContent = `${source} -> ${target}`;
    const meta = document.createElement("span");
    meta.textContent = `${edge.graph_type} · ${edge.relation}`;
    item.append(title, meta);
    dom.graphEdges.append(item);
  }
}

function renderWarnings(warnings) {
  dom.warnings.replaceChildren();
  for (const warning of warnings) {
    const line = document.createElement("div");
    line.textContent = warning;
    dom.warnings.append(line);
  }
}

function renderIndexSearchResult(data) {
  renderBackendStatus(data.backend_status || {});
  renderIndexTimings(data.timings || {});
  renderIndexList(dom.esResults, data.elasticsearch_results || []);
  renderIndexList(dom.milvusResults, data.milvus_results || []);
  renderIndexList(dom.fusedResults, data.fused_results || []);
  renderIndexWarnings(data.warnings || []);
}

function renderBackendStatus(status) {
  dom.backendStatus.replaceChildren();
  if (!Object.keys(status).length) {
    return;
  }

  const core = status.core || {};
  const evorag = status.evorag || {};
  const items = [
    ["Core 初始化", core.initialized ? "是" : "否"],
    ["EvoRAG MySQL", availabilityText(evorag.mysql)],
    ["ES 健康", availabilityText(core.elasticsearch)],
    ["Milvus 健康", availabilityText(core.milvus)],
    ["EvoRAG ES 复用连接", evorag.elasticsearch?.uses_core_client ? "是" : "否"],
    ["EvoRAG Milvus 复用连接", evorag.milvus?.uses_core_client ? "是" : "否"],
  ];

  for (const [label, value] of items) {
    const item = document.createElement("div");
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = value;
    item.append(labelElement, valueElement);
    dom.backendStatus.append(item);
  }
}

function availabilityText(status) {
  if (!status) {
    return "未知";
  }
  if (status.available) {
    return "可用";
  }
  return status.error ? `不可用：${status.error}` : "不可用";
}

function renderIndexTimings(timings) {
  dom.indexTimings.replaceChildren();
  if (!Object.keys(timings).length) {
    return;
  }

  const items = [
    ["健康检查", timings.backend_status_ms],
    ["Embedding", timings.embedding_ms],
    ["ES 准备", timings.elasticsearch_ensure_ms],
    ["ES 检索", timings.elasticsearch_search_ms],
    ["Milvus 准备", timings.milvus_ensure_ms],
    ["Milvus 检索", timings.milvus_search_ms],
    ["融合", timings.fusion_ms],
    ["MySQL 回填", timings.mysql_hydration_ms],
    ["后端总耗时", timings.total_ms],
  ];

  for (const [label, value] of items) {
    if (typeof value !== "number") {
      continue;
    }
    const item = document.createElement("div");
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = formatMilliseconds(value);
    item.append(labelElement, valueElement);
    dom.indexTimings.append(item);
  }
}

function renderIndexList(container, entities) {
  container.replaceChildren();
  if (!entities.length) {
    container.append(emptyState("暂无结果"));
    return;
  }

  for (const entity of entities) {
    container.append(indexEntityCard(entity));
  }
}

function indexEntityCard(entity) {
  const card = document.createElement("article");
  card.className = "index-result-card";

  const head = document.createElement("div");
  head.className = "index-result-head";
  const title = document.createElement("strong");
  title.textContent = entity.canonical_name || `Entity ${entity.id}`;
  const score = document.createElement("span");
  score.textContent = `#${entity.rank || "-"} · ${formatScore(entity.score)} · ${entity.source || "unknown"}`;
  head.append(title, score);
  card.append(head);

  const description = entity.identity_description || entity.summary || entity.description_for_match;
  if (description) {
    const text = document.createElement("p");
    text.textContent = description;
    card.append(text);
  }

  const meta = document.createElement("div");
  meta.className = "index-result-meta";
  meta.textContent = [entity.entity_type || "concept", aliasesText(entity.aliases)].filter(Boolean).join(" · ");
  card.append(meta);

  const attributes = entity.attributes || [];
  if (attributes.length) {
    const list = document.createElement("div");
    list.className = "index-attributes";
    for (const attribute of attributes.slice(0, 5)) {
      const item = document.createElement("div");
      const label = document.createElement("b");
      label.textContent = attributeTitle(attribute.attr_type);
      const value = document.createElement("span");
      value.textContent = attribute.value_text || "";
      item.append(label, value);
      list.append(item);
    }
    card.append(list);
  }

  return card;
}

function renderIndexWarnings(warnings) {
  dom.indexWarnings.replaceChildren();
  for (const warning of warnings) {
    const line = document.createElement("div");
    line.textContent = warning;
    dom.indexWarnings.append(line);
  }
}

function markdownBlocks(text) {
  const nodes = [];
  let list = null;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trimEnd();
    if (!line) {
      list = null;
      continue;
    }
    if (line.startsWith("### ")) {
      nodes.push(heading("h3", line.slice(4)));
      list = null;
    } else if (line.startsWith("## ")) {
      nodes.push(heading("h2", line.slice(3)));
      list = null;
    } else if (line.startsWith("# ")) {
      nodes.push(heading("h1", line.slice(2)));
      list = null;
    } else if (line.startsWith("- ")) {
      if (!list) {
        list = document.createElement("ul");
        nodes.push(list);
      }
      const li = document.createElement("li");
      li.textContent = line.slice(2);
      list.append(li);
    } else {
      const p = document.createElement("p");
      p.textContent = line;
      nodes.push(p);
      list = null;
    }
  }
  return nodes;
}

function heading(level, text) {
  const element = document.createElement(level);
  element.textContent = text;
  return element;
}

function emptyItem(text) {
  const item = document.createElement("div");
  item.className = "list-item";
  const span = document.createElement("span");
  span.textContent = text;
  item.append(span);
  return item;
}

function setStatus(element, text, kind) {
  element.textContent = text;
  element.classList.toggle("ok", kind === "ok");
  element.classList.toggle("error", kind === "error");
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = label;
}

function updateAuthState() {
  dom.authState.textContent = state.token ? "已登录" : "未登录";
  dom.authState.classList.toggle("ready", Boolean(state.token));
}

function formatScore(value) {
  const number = Number(value || 0);
  return number.toFixed(3);
}

function formatDuration(startedAt) {
  const elapsedMs = Math.max(0, performance.now() - startedAt);
  return formatMilliseconds(elapsedMs);
}

function formatMilliseconds(elapsedMs) {
  if (elapsedMs < 1000) {
    return `${Math.round(elapsedMs)} ms`;
  }
  return `${(elapsedMs / 1000).toFixed(2)} s`;
}

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

function attributeTitle(type) {
  return ATTRIBUTE_TITLES[type] || type;
}
