const state = {
  token: localStorage.getItem("evorag_token") || "",
  lastPayload: null,
};

const dom = {
  authForm: document.querySelector("#auth-form"),
  password: document.querySelector("#password"),
  authState: document.querySelector("#auth-state"),
  workspaceId: document.querySelector("#workspace-id"),
  projectId: document.querySelector("#project-id"),
  collectionId: document.querySelector("#collection-id"),
  domain: document.querySelector("#domain"),
  knowledgeText: document.querySelector("#knowledge-text"),
  pipelineBtn: document.querySelector("#pipeline-btn"),
  clearBtn: document.querySelector("#clear-btn"),
  statusLine: document.querySelector("#status-line"),
  summaryStrip: document.querySelector("#summary-strip"),
  stage1Count: document.querySelector("#stage1-count"),
  stage2Count: document.querySelector("#stage2-count"),
  stage3Count: document.querySelector("#stage3-count"),
  stage1Output: document.querySelector("#stage1-output"),
  stage2Output: document.querySelector("#stage2-output"),
  stage3Output: document.querySelector("#stage3-output"),
  rawJson: document.querySelector("#raw-json"),
};

dom.collectionId.value = `pipeline-${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}`;

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

dom.pipelineBtn.addEventListener("click", runPipeline);
dom.clearBtn.addEventListener("click", () => {
  dom.knowledgeText.value = "";
  state.lastPayload = null;
  renderEmpty();
  setStatus("", "");
});

async function runPipeline() {
  const text = dom.knowledgeText.value.trim();
  if (!text) {
    setStatus("请输入要测试的文本。", "error");
    return;
  }

  const startedAt = performance.now();
  try {
    setBusy(dom.pipelineBtn, true, "运行中");
    setStatus("正在运行完整 Pipeline...", "");
    const payload = await apiPost("/api/evorag/debug/pipeline", {
      text,
      note_id: "debug-pipeline",
      task_name: "pipeline-debug",
      scope: scopePayload(),
    });
    state.lastPayload = payload;
    renderPayload(payload);
    setStatus(`完成，耗时 ${formatDuration(startedAt)}。`, "ok");
  } catch (error) {
    setStatus(`${error.message}（耗时 ${formatDuration(startedAt)}）`, "error");
  } finally {
    setBusy(dom.pipelineBtn, false, "运行 Pipeline");
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

function scopePayload() {
  return {
    workspace_id: dom.workspaceId.value.trim() || "debug",
    project_id: dom.projectId.value.trim() || "pipeline-test",
    collection_id: dom.collectionId.value.trim() || "default",
    domain: dom.domain.value.trim() || "general",
  };
}

function renderEmpty() {
  dom.summaryStrip.replaceChildren();
  dom.stage1Count.textContent = "0";
  dom.stage2Count.textContent = "0";
  dom.stage3Count.textContent = "0";
  dom.stage1Output.replaceChildren(emptyState("暂无实体抽取结果"));
  dom.stage2Output.replaceChildren(emptyState("暂无 memory_guarded_hybrid 结果"));
  dom.stage3Output.replaceChildren(emptyState("暂无入库合并结果"));
  dom.rawJson.textContent = "";
}

function renderPayload(payload) {
  const stages = payload.stages || {};
  const stage1 = stages.stage1_extraction || {};
  const stage2 = stages.stage2_memory_guarded_hybrid || {};
  const stage3 = stages.stage3_merge_persistence || {};

  dom.rawJson.textContent = JSON.stringify(payload, null, 2);
  dom.stage1Count.textContent = `${stage1.entity_count || 0} 个实体`;
  dom.stage2Count.textContent = `${stage2.decision_count || 0} 个决策`;
  dom.stage3Count.textContent = `${stage3.processed?.length || 0} 条处理`;

  renderSummary(payload, stage1, stage2, stage3);
  renderStage1(stage1);
  renderStage2(stage2);
  renderStage3(stage3);
}

function renderSummary(payload, stage1, stage2, stage3) {
  const jobStatus = stage3.job_status?.status || stage3.status || "-";
  const scope = payload.scope || {};
  const items = [
    ["输入字符", textLength(payload.input_text)],
    ["Block", stage1.block_count || 0],
    ["Incoming", stage2.incoming_count || 0],
    ["Job", `${stage3.job_id || "-"} / ${jobStatus}`],
    ["Scope", [scope.workspace_id, scope.project_id, scope.collection_id].filter(Boolean).join(" / ")],
  ];

  dom.summaryStrip.replaceChildren();
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "summary-item";
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = String(value || "-");
    item.append(labelElement, valueElement);
    dom.summaryStrip.append(item);
  }
}

function renderStage1(stage) {
  const blocks = stage.entity_extraction?.blocks || [];
  dom.stage1Output.replaceChildren();
  if (!blocks.length) {
    dom.stage1Output.append(emptyState("暂无实体抽取结果"));
    return;
  }

  for (const block of blocks) {
    const item = panelItem(`Block #${Number(block.block_index ?? 0) + 1}`, block.heading || block.anchor_entity || "未命名");
    for (const entity of block.entities || []) {
      item.append(entityLine(entity));
    }
    if (!(block.entities || []).length) {
      item.append(emptyState("这个 Block 没有抽到实体"));
    }
    if (block.warnings?.length) {
      item.append(warningLine(block.warnings.join("；")));
    }
    dom.stage1Output.append(item);
  }
}

function renderStage2(stage) {
  const decisions = stage.decisions || [];
  dom.stage2Output.replaceChildren();
  if (!decisions.length) {
    dom.stage2Output.append(emptyState("暂无 memory_guarded_hybrid 决策"));
    return;
  }

  for (const decision of decisions) {
    const item = panelItem(decision.incoming?.name || "未命名 incoming", decision.decision || "-");
    item.append(metaRow([
      `分数：${formatScore(decision.score)}`,
      decision.reason ? `原因：${decision.reason}` : "",
      decision.matched_entity ? `命中：#${decision.matched_entity.id} ${decision.matched_entity.canonical_name}` : "",
    ]));
    const candidates = decision.candidates || [];
    if (candidates.length) {
      item.append(sectionTitle("候选"));
      for (const candidate of candidates) {
        item.append(metaRow([
          `#${candidate.rank || "-"}`,
          candidate.canonical_name || `Entity ${candidate.entity_id}`,
          candidate.source || "-",
          `score ${formatScore(candidate.score)}`,
          `vector ${formatScore(candidate.vector_score)}`,
          `es ${formatScore(candidate.es_score)}`,
        ]));
      }
    }
    dom.stage2Output.append(item);
  }
}

function renderStage3(stage) {
  dom.stage3Output.replaceChildren();
  const item = panelItem(`Job #${stage.job_id || "-"}`, stage.job_status?.status || stage.status || "-");
  item.append(metaRow([
    `排队：${stage.queued_count || 0}`,
    `Incoming IDs：${(stage.incoming_entity_ids || []).join(", ") || "-"}`,
  ]));

  for (const processed of stage.processed || []) {
    item.append(metaRow([
      `Incoming #${processed.incoming_entity_id}`,
      `状态：${processed.status}`,
    ]));
  }

  if (stage.job_status) {
    item.append(sectionTitle("Job 状态"));
    item.append(codeBlock(JSON.stringify(stage.job_status, null, 2)));
  }
  dom.stage3Output.append(item);
}

function entityLine(entity) {
  const row = document.createElement("div");
  row.className = "entity-line";
  const title = document.createElement("strong");
  title.textContent = entity.name || "未命名实体";
  const meta = document.createElement("span");
  meta.textContent = [entity.entity_type || "concept", entity.identity_description || ""].filter(Boolean).join(" · ");
  row.append(title, meta);
  return row;
}

function panelItem(title, meta) {
  const item = document.createElement("article");
  item.className = "result-item";
  const head = document.createElement("div");
  head.className = "item-head";
  const titleElement = document.createElement("strong");
  titleElement.textContent = title;
  const metaElement = document.createElement("span");
  metaElement.textContent = meta;
  head.append(titleElement, metaElement);
  item.append(head);
  return item;
}

function metaRow(items) {
  const values = items.filter(Boolean);
  const row = document.createElement("div");
  row.className = "meta-row";
  for (const value of values) {
    const chip = document.createElement("span");
    chip.textContent = value;
    row.append(chip);
  }
  return row;
}

function sectionTitle(text) {
  const title = document.createElement("h3");
  title.className = "section-title";
  title.textContent = text;
  return title;
}

function codeBlock(text) {
  const block = document.createElement("pre");
  block.className = "inline-json";
  block.textContent = text;
  return block;
}

function warningLine(text) {
  const line = document.createElement("div");
  line.className = "warning-line";
  line.textContent = text;
  return line;
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

function textLength(text) {
  return String(text || "").length;
}

function formatScore(value) {
  if (typeof value !== "number") {
    return "-";
  }
  return value.toFixed(3);
}

function formatDuration(startedAt) {
  const value = Math.max(0, performance.now() - startedAt);
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(2)} s`;
}
