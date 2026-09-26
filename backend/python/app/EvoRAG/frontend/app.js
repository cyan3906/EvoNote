const state = {
  token: localStorage.getItem("evorag_token") || "",
  lastAnswer: "",
};

const dom = {
  authForm: document.querySelector("#auth-form"),
  password: document.querySelector("#password"),
  authState: document.querySelector("#auth-state"),
  ingestBtn: document.querySelector("#ingest-btn"),
  queryBtn: document.querySelector("#query-btn"),
  copyAnswerBtn: document.querySelector("#copy-answer-btn"),
  knowledgeText: document.querySelector("#knowledge-text"),
  entityQuery: document.querySelector("#entity-query"),
  topK: document.querySelector("#top-k"),
  ingestStatus: document.querySelector("#ingest-status"),
  queryStatus: document.querySelector("#query-status"),
  extractedEntities: document.querySelector("#extracted-entities"),
  answer: document.querySelector("#answer"),
  retrievedEntities: document.querySelector("#retrieved-entities"),
  graphEdges: document.querySelector("#graph-edges"),
  warnings: document.querySelector("#warnings"),
  workspaceId: document.querySelector("#workspace-id"),
  projectId: document.querySelector("#project-id"),
  collectionId: document.querySelector("#collection-id"),
  domain: document.querySelector("#domain"),
};

updateAuthState();

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

dom.ingestBtn.addEventListener("click", async () => {
  const text = dom.knowledgeText.value.trim();
  if (!text) {
    setStatus(dom.ingestStatus, "请输入要处理的知识文本。", "error");
    return;
  }

  try {
    setBusy(dom.ingestBtn, true, "处理中");
    setStatus(dom.ingestStatus, "正在切分 Block、抽取实体并写入实体库...", "");
    const data = await apiFetch("/api/evorag/ingest", {
      text,
      scope: readScope(),
    });
    renderExtractedEntities(data.preprocess);
    const blockCount = data.preprocess.blocks?.length || 0;
    const entityCount = data.preprocess.blocks?.reduce((total, block) => total + (block.entities?.length || 0), 0) || 0;
    const upsertCount = data.ingest_results?.length || 0;
    setStatus(dom.ingestStatus, `完成：${blockCount} 个 Block，${entityCount} 个实体，${upsertCount} 个实体写入或合并。`, "ok");
  } catch (error) {
    setStatus(dom.ingestStatus, error.message, "error");
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

  try {
    setBusy(dom.queryBtn, true, "生成中");
    setStatus(dom.queryStatus, "正在检索实体、构建依赖图并生成长文本...", "");
    const data = await apiFetch("/api/evorag/query", {
      entity,
      top_k: Number(dom.topK.value || 5),
      scope: readScope(),
    });
    renderQueryResult(data);
    setStatus(dom.queryStatus, data.root_entity ? `完成：根实体 ${data.root_entity.canonical_name}` : "没有找到匹配实体。", data.root_entity ? "ok" : "error");
  } catch (error) {
    setStatus(dom.queryStatus, error.message, "error");
  } finally {
    setBusy(dom.queryBtn, false, "生成");
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

function renderExtractedEntities(preprocess) {
  const entities = [];
  for (const block of preprocess.blocks || []) {
    for (const entity of block.entities || []) {
      if (!entities.includes(entity.name)) {
        entities.push(entity.name);
      }
    }
  }

  dom.extractedEntities.replaceChildren();
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
