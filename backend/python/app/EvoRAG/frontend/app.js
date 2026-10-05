const state = {
  token: localStorage.getItem("evorag_token") || "",
  lastAnswer: "",
  activeJobId: 0,
  jobPollTimer: 0,
  graphRootEntityId: 0,
  graphSelectedEntityIds: [],
};

const dom = {
  authForm: document.querySelector("#auth-form"),
  password: document.querySelector("#password"),
  authState: document.querySelector("#auth-state"),
  extractBtn: document.querySelector("#extract-btn"),
  ingestBtn: document.querySelector("#ingest-btn"),
  queryBtn: document.querySelector("#query-btn"),
  graphCandidatesBtn: document.querySelector("#graph-candidates-btn"),
  graphGenerateBtn: document.querySelector("#graph-generate-btn"),
  indexSearchBtn: document.querySelector("#index-search-btn"),
  jobsRefreshBtn: document.querySelector("#jobs-refresh-btn"),
  reviewRefreshBtn: document.querySelector("#review-refresh-btn"),
  copyAnswerBtn: document.querySelector("#copy-answer-btn"),
  knowledgeText: document.querySelector("#knowledge-text"),
  entityQuery: document.querySelector("#entity-query"),
  topK: document.querySelector("#top-k"),
  ingestStatus: document.querySelector("#ingest-status"),
  queryStatus: document.querySelector("#query-status"),
  graphCandidates: document.querySelector("#graph-candidates"),
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
  jobsStatus: document.querySelector("#jobs-status"),
  workerSummary: document.querySelector("#worker-summary"),
  jobProgress: document.querySelector("#job-progress"),
  jobList: document.querySelector("#job-list"),
  reviewStatus: document.querySelector("#review-status"),
  reviewTasks: document.querySelector("#review-tasks"),
  workspaceId: document.querySelector("#workspace-id"),
  projectId: document.querySelector("#project-id"),
  collectionId: document.querySelector("#collection-id"),
  domain: document.querySelector("#domain"),
};

updateAuthState();
if (state.token) {
  loadWorkerStatus();
}

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
    await loadWorkerStatus();
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
    renderExtractedEntities(data.preprocess, data.ingest_results || [], data);
    const blockCount = data.preprocess.blocks?.length || 0;
    const entityCount = data.preprocess.blocks?.reduce((total, block) => total + (block.entities?.length || 0), 0) || 0;
    setStatus(
      dom.ingestStatus,
      `完成：${blockCount} 个 Block，${entityCount} 个实体，已创建入库任务 #${data.job_id || "-"}，排队 ${data.queued_count || 0} 个实体。耗时 ${formatDuration(startedAt)}。`,
      "ok",
    );
    if (data.job_id) {
      state.activeJobId = data.job_id;
      await loadJobStatus(data.job_id);
      startJobPolling(data.job_id);
    }
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

dom.graphCandidatesBtn.addEventListener("click", async () => {
  const entity = dom.entityQuery.value.trim();
  if (!entity) {
    setStatus(dom.queryStatus, "请输入要检索的实体。", "error");
    return;
  }

  const startedAt = performance.now();
  try {
    setBusy(dom.graphCandidatesBtn, true, "加载中");
    setStatus(dom.queryStatus, "正在检索相关实体候选并构建初始图谱...", "");
    const data = await apiFetch("/api/evorag/graph/candidates", {
      entity,
      top_k: Number(dom.topK.value || 5),
      scope: readScope(),
    });
    renderGraphCandidates(data);
    setStatus(
      dom.queryStatus,
      data.root_entity
        ? `已加载 ${data.candidates?.length || 0} 个相关实体候选。耗时 ${formatDuration(startedAt)}。`
        : `没有找到匹配实体。耗时 ${formatDuration(startedAt)}。`,
      data.root_entity ? "ok" : "error",
    );
  } catch (error) {
    setStatus(dom.queryStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.graphCandidatesBtn, false, "选择相关实体");
  }
});

dom.graphGenerateBtn.addEventListener("click", async () => {
  const selectedIds = selectedGraphEntityIds();
  if (!state.graphRootEntityId || !selectedIds.length) {
    setStatus(dom.queryStatus, "请先加载并选择相关实体。", "error");
    return;
  }

  const startedAt = performance.now();
  try {
    setBusy(dom.graphGenerateBtn, true, "生成中");
    setStatus(dom.queryStatus, "正在根据选中实体生成知识图和长文...", "");
    const data = await apiFetch("/api/evorag/graph/generate", {
      root_entity_id: state.graphRootEntityId,
      selected_entity_ids: selectedIds,
      scope: readScope(),
    });
    renderGraphGeneratedResult(data);
    setStatus(dom.queryStatus, `完成：已生成 ${data.entities?.length || 0} 个实体的知识图长文。耗时 ${formatDuration(startedAt)}。`, "ok");
  } catch (error) {
    setStatus(dom.queryStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.graphGenerateBtn, false, "生成知识图长文");
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

dom.reviewRefreshBtn.addEventListener("click", () => {
  loadReviewTasks();
});

dom.jobsRefreshBtn.addEventListener("click", async () => {
  await loadWorkerStatus();
  if (state.activeJobId) {
    await loadJobStatus(state.activeJobId);
  } else {
    await loadIngestJobs();
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
  return apiRequest(url, { method: "POST", payload });
}

async function apiGet(url) {
  return apiRequest(url, { method: "GET" });
}

async function apiRequest(url, options = {}) {
  if (!state.token) {
    throw new Error("请先登录。");
  }
  const headers = {
    Authorization: `Bearer ${state.token}`,
  };
  if (options.payload !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(url, {
    method: options.method || "GET",
    headers,
    body: options.payload === undefined ? undefined : JSON.stringify(options.payload),
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

function renderExtractedEntities(preprocess, ingestResults = [], ingestJob = null) {
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

  renderPreprocessDetails(preprocess, ingestResults, ingestJob);
}

function renderPreprocessDetails(preprocess, ingestResults = [], ingestJob = null) {
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

  renderIngestJob(ingestJob);
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

function renderIngestJob(job) {
  if (!job?.job_id) {
    return;
  }

  const section = document.createElement("section");
  section.className = "ingest-result-list";
  const title = document.createElement("h3");
  title.textContent = "入库任务";
  section.append(title);

  const item = document.createElement("div");
  item.className = "ingest-result-item";
  const name = document.createElement("strong");
  name.textContent = `Job #${job.job_id}`;
  const meta = document.createElement("span");
  meta.textContent = [
    `状态 ${job.status || "queued"}`,
    `${job.queued_count || 0} 个实体排队`,
    job.incoming_entity_ids?.length ? `任务实体 ID ${job.incoming_entity_ids.slice(0, 8).join("、")}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  item.append(name, meta);
  section.append(item);
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

function renderGraphCandidates(data) {
  state.graphRootEntityId = data.root_entity?.id || 0;
  state.graphSelectedEntityIds = data.default_selected_entity_ids || [];
  renderEntities(data.candidates || []);
  renderEdges(data.graph);
  renderWarnings(data.warnings || []);
  renderGraphCandidateControls(data.candidates || [], state.graphSelectedEntityIds);
}

function renderGraphGeneratedResult(data) {
  state.lastAnswer = data.article || "";
  dom.answer.classList.toggle("empty", !state.lastAnswer);
  dom.answer.replaceChildren(...markdownBlocks(state.lastAnswer || "生成结果会显示在这里。"));
  renderEntities(data.entities || []);
  renderEdges(data.graph);
  renderWarnings(data.warnings || []);
}

function renderGraphCandidateControls(entities, selectedIds) {
  dom.graphCandidates.replaceChildren();
  if (!entities.length) {
    dom.graphCandidates.append(emptyItem("暂无可选相关实体"));
    return;
  }
  const selected = new Set((selectedIds || []).map((id) => Number(id)));
  for (const entity of entities) {
    const label = document.createElement("label");
    label.className = "graph-candidate";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = String(entity.id);
    checkbox.checked = selected.has(Number(entity.id));
    if (Number(entity.id) === Number(state.graphRootEntityId)) {
      checkbox.checked = true;
      checkbox.disabled = true;
    }

    const body = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = entity.canonical_name || `Entity ${entity.id}`;
    const meta = document.createElement("small");
    meta.textContent = [
      Number(entity.id) === Number(state.graphRootEntityId) ? "根实体" : "",
      entity.entity_type || "concept",
      `score ${formatScore(entity.score)}`,
      entity.source || "",
    ]
      .filter(Boolean)
      .join(" · ");
    body.append(title, meta);
    label.append(checkbox, body);
    dom.graphCandidates.append(label);
  }
}

function selectedGraphEntityIds() {
  return [...dom.graphCandidates.querySelectorAll("input[type='checkbox']:checked")]
    .map((input) => Number(input.value))
    .filter((value) => Number.isFinite(value) && value > 0);
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

async function loadReviewTasks() {
  const startedAt = performance.now();
  try {
    setBusy(dom.reviewRefreshBtn, true, "加载中");
    setStatus(dom.reviewStatus, "正在加载待审核实体...", "");
    const data = await apiGet("/api/evorag/review-tasks?status=pending&limit=50");
    renderReviewTasks(data.tasks || []);
    setStatus(dom.reviewStatus, `待审核 ${data.tasks?.length || 0} 个。耗时 ${formatDuration(startedAt)}。`, "ok");
  } catch (error) {
    setStatus(dom.reviewStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.reviewRefreshBtn, false, "刷新审核");
  }
}

function startJobPolling(jobId) {
  stopJobPolling();
  state.jobPollTimer = window.setInterval(() => {
    loadJobStatus(jobId, { silent: true });
  }, 2000);
}

function stopJobPolling() {
  if (state.jobPollTimer) {
    window.clearInterval(state.jobPollTimer);
    state.jobPollTimer = 0;
  }
}

async function loadWorkerStatus() {
  try {
    const data = await apiGet("/api/evorag/worker-status");
    renderWorkerSummary(data);
  } catch (error) {
    setStatus(dom.jobsStatus, `Worker 状态读取失败：${error.message}`, "error");
  }
}

async function loadIngestJobs() {
  const startedAt = performance.now();
  try {
    setBusy(dom.jobsRefreshBtn, true, "加载中");
    setStatus(dom.jobsStatus, "正在加载最近入库任务...", "");
    await loadWorkerStatus();
    const data = await apiGet("/api/evorag/ingest-jobs?limit=12");
    renderJobList(data.jobs || []);
    setStatus(dom.jobsStatus, `最近任务 ${data.jobs?.length || 0} 个。耗时 ${formatDuration(startedAt)}。`, "ok");
  } catch (error) {
    setStatus(dom.jobsStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.jobsRefreshBtn, false, "刷新任务");
  }
}

async function loadJobStatus(jobId, options = {}) {
  const startedAt = performance.now();
  try {
    if (!options.silent) {
      setBusy(dom.jobsRefreshBtn, true, "刷新中");
      setStatus(dom.jobsStatus, `正在刷新 Job #${jobId}...`, "");
    }
    await loadWorkerStatus();
    const data = await apiGet(`/api/evorag/ingest-jobs/${jobId}`);
    renderJobProgress(data);
    const progress = data.progress || {};
    const terminal = isJobTerminal(data);
    setStatus(
      dom.jobsStatus,
      `Job #${jobId}：${statusTitle(data.status)}，${progress.done || 0}/${progress.total || 0} 已完成或待处理，${formatPercent(progress.percent)}。耗时 ${formatDuration(startedAt)}。`,
      data.status === "failed" ? "error" : "ok",
    );
    if (terminal) {
      stopJobPolling();
      await loadReviewTasks();
    }
  } catch (error) {
    setStatus(dom.jobsStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
    stopJobPolling();
  } finally {
    if (!options.silent) {
      setBusy(dom.jobsRefreshBtn, false, "刷新任务");
    }
  }
}

function renderWorkerSummary(data) {
  const worker = data.worker || {};
  const queue = data.queue || {};
  const mysql = data.mysql || {};
  const items = [
    ["Worker 启用", worker.enabled ? "是" : "否"],
    ["Worker 已启动", worker.started ? "是" : "否"],
    ["Worker 数", worker.worker_count ?? "-"],
    ["Stream 长度", queue.stream_length ?? "-"],
    ["待 ACK", queue.pending_count ?? "-"],
    ["消费者", queue.consumer_count ?? "-"],
    ["平均完成耗时", formatMilliseconds(mysql.average_completed_ms || 0)],
    ["重试次数", mysql.retry_count ?? 0],
    ["失败数", mysql.failed_count ?? 0],
  ];
  dom.workerSummary.replaceChildren();
  for (const [label, value] of items) {
    const item = document.createElement("div");
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = String(value);
    item.append(labelElement, valueElement);
    dom.workerSummary.append(item);
  }
}

function renderJobList(jobs) {
  dom.jobProgress.replaceChildren();
  dom.jobList.replaceChildren();
  if (!jobs.length) {
    dom.jobList.append(emptyState("暂无入库任务"));
    return;
  }
  for (const job of jobs) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "job-list-item";
    item.textContent = `#${job.id} · ${statusTitle(job.status)} · ${job.entity_count || 0} 实体 · ${job.created_at || ""}`;
    item.addEventListener("click", async () => {
      state.activeJobId = job.id;
      await loadJobStatus(job.id);
      if (!isJobTerminal(job)) {
        startJobPolling(job.id);
      }
    });
    dom.jobList.append(item);
  }
}

function renderJobProgress(job) {
  dom.jobList.replaceChildren();
  dom.jobProgress.replaceChildren();
  const progress = job.progress || {};

  const head = document.createElement("div");
  head.className = "job-progress-head";
  const title = document.createElement("strong");
  title.textContent = `Job #${job.id} · ${statusTitle(job.status)}`;
  const meta = document.createElement("span");
  meta.textContent = [
    `${progress.done || 0}/${progress.total || 0}`,
    `${formatPercent(progress.percent)}`,
    `${job.block_count || 0} Block`,
    `${job.entity_count || 0} 实体`,
    job.updated_at ? `更新 ${job.updated_at}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  head.append(title, meta);
  dom.jobProgress.append(head);

  const bar = document.createElement("div");
  bar.className = "job-progress-bar";
  const fill = document.createElement("div");
  fill.style.width = `${Math.max(0, Math.min(100, Number(progress.percent || 0)))}%`;
  bar.append(fill);
  dom.jobProgress.append(bar);

  const counts = document.createElement("div");
  counts.className = "job-status-counts";
  const statusCounts = job.status_counts || {};
  for (const status of ["pending", "processing", "auto_merged", "manual_merged", "new_created", "needs_review", "failed", "dead_letter"]) {
    if (!statusCounts[status]) {
      continue;
    }
    const chip = document.createElement("span");
    chip.textContent = `${statusTitle(status)} ${statusCounts[status]}`;
    counts.append(chip);
  }
  dom.jobProgress.append(counts);

  const list = document.createElement("div");
  list.className = "incoming-status-list";
  for (const incoming of job.incoming_entities || []) {
    list.append(incomingStatusItem(incoming));
  }
  dom.jobProgress.append(list);
}

function incomingStatusItem(incoming) {
  const item = document.createElement("div");
  item.className = `incoming-status-item status-${incoming.status || "unknown"}`;
  const body = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = incoming.name || `Incoming #${incoming.id}`;
  const meta = document.createElement("span");
  meta.textContent = [
    `#${incoming.id}`,
    incoming.entity_type || "concept",
    statusTitle(incoming.status),
    `尝试 ${incoming.attempt_count || 0}`,
    incoming.matched_entity_id ? `匹配实体 ${incoming.matched_entity_id}` : "",
    incoming.review_task_id ? `审核 #${incoming.review_task_id}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  body.append(name, meta);
  if (incoming.decision_reason || incoming.last_error) {
    const reason = document.createElement("p");
    reason.textContent = incoming.last_error || incoming.decision_reason;
    body.append(reason);
  }
  item.append(body);
  if (incoming.review_task_id) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary-button";
    button.textContent = "看审核";
    button.addEventListener("click", loadReviewTasks);
    item.append(button);
  }
  return item;
}

function isJobTerminal(job) {
  return ["completed", "failed", "needs_review"].includes(String(job.status || ""));
}

function renderReviewTasks(tasks) {
  dom.reviewTasks.replaceChildren();
  if (!tasks.length) {
    dom.reviewTasks.append(emptyState("暂无待审核任务"));
    return;
  }

  for (const task of tasks) {
    dom.reviewTasks.append(reviewTaskCard(task));
  }
}

function reviewTaskCard(task) {
  const incoming = task.incoming_snapshot || {};
  const candidates = task.candidates || [];
  const card = document.createElement("article");
  card.className = "review-card";

  const head = document.createElement("div");
  head.className = "review-head";
  const title = document.createElement("strong");
  title.textContent = incoming.name || `Review #${task.id}`;
  const meta = document.createElement("span");
  meta.textContent = [
    `任务 #${task.id}`,
    `Incoming #${task.incoming_entity_id}`,
    incoming.entity_type || "concept",
    task.reason || "",
  ]
    .filter(Boolean)
    .join(" · ");
  head.append(title, meta);
  card.append(head);

  if (incoming.identity_description || incoming.description_for_match) {
    const description = document.createElement("p");
    description.className = "review-description";
    description.textContent = incoming.identity_description || incoming.description_for_match;
    card.append(description);
  }

  const actions = document.createElement("div");
  actions.className = "review-actions";
  const createButton = document.createElement("button");
  createButton.type = "button";
  createButton.className = "secondary-button";
  createButton.textContent = "新建实体";
  createButton.addEventListener("click", () => handleReviewAction(createButton, task.id, "new", {}));
  actions.append(createButton);
  card.append(actions);

  const list = document.createElement("div");
  list.className = "review-candidates";
  if (!candidates.length) {
    list.append(emptyState("暂无候选实体"));
  } else {
    for (const candidate of candidates) {
      list.append(reviewCandidateItem(task.id, candidate));
    }
  }
  card.append(list);
  return card;
}

function reviewCandidateItem(reviewTaskId, candidate) {
  const item = document.createElement("div");
  item.className = "review-candidate";

  const body = document.createElement("div");
  body.className = "review-candidate-body";
  const name = document.createElement("strong");
  name.textContent = candidate.canonical_name || `Entity ${candidate.id}`;
  const meta = document.createElement("span");
  meta.textContent = [
    `ID ${candidate.id}`,
    candidate.entity_type || "concept",
    `融合 ${formatScore(candidate.score)}`,
    `向量 ${formatScore(candidate.vector_score)}`,
    `ES ${formatScore(candidate.es_score)}`,
  ].join(" · ");
  body.append(name, meta);
  if (candidate.identity_description || candidate.summary || candidate.description_for_match) {
    const text = document.createElement("p");
    text.textContent = candidate.identity_description || candidate.summary || candidate.description_for_match;
    body.append(text);
  }
  item.append(body);

  const actions = document.createElement("div");
  actions.className = "review-candidate-actions";
  const mergeButton = document.createElement("button");
  mergeButton.type = "button";
  mergeButton.textContent = "合并";
  mergeButton.addEventListener("click", () =>
    handleReviewAction(mergeButton, reviewTaskId, "merge", {
      entity_id: candidate.id,
    }),
  );
  const rejectButton = document.createElement("button");
  rejectButton.type = "button";
  rejectButton.className = "secondary-button";
  rejectButton.textContent = "排除";
  rejectButton.addEventListener("click", () =>
    handleReviewAction(rejectButton, reviewTaskId, "reject", {
      candidate_entity_ids: [candidate.id],
    }),
  );
  actions.append(mergeButton, rejectButton);
  item.append(actions);
  return item;
}

async function handleReviewAction(button, reviewTaskId, action, payload) {
  const startedAt = performance.now();
  const endpoints = {
    merge: `/api/evorag/review-tasks/${reviewTaskId}/merge`,
    new: `/api/evorag/review-tasks/${reviewTaskId}/new`,
    reject: `/api/evorag/review-tasks/${reviewTaskId}/reject`,
  };
  try {
    setBusy(button, true, "处理中");
    setStatus(dom.reviewStatus, "正在提交审核决策...", "");
    await apiFetch(endpoints[action], {
      ...payload,
      decided_by: "manual",
      reason: "manual review from EvoRAG workbench",
    });
    setStatus(dom.reviewStatus, `审核已提交。耗时 ${formatDuration(startedAt)}。`, "ok");
    await loadReviewTasks();
  } catch (error) {
    setStatus(dom.reviewStatus, `${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(button, false, actionLabel(action));
  }
}

function actionLabel(action) {
  if (action === "merge") {
    return "合并";
  }
  if (action === "reject") {
    return "排除";
  }
  return "新建实体";
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

function formatPercent(value) {
  const number = Number(value || 0);
  return `${number.toFixed(number % 1 === 0 ? 0 : 1)}%`;
}

function statusTitle(status) {
  return STATUS_TITLES[status] || status || "未知";
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

const STATUS_TITLES = {
  queued: "已排队",
  pending: "排队中",
  processing: "处理中",
  auto_merged: "自动合并",
  manual_merged: "人工合并",
  new_created: "新建实体",
  completed: "已完成",
  needs_review: "待审核",
  failed: "失败",
  dead_letter: "死信",
  skipped: "跳过",
};

function attributeTitle(type) {
  return ATTRIBUTE_TITLES[type] || type;
}
