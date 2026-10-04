import { createEmptyMergeTask, createSampleNotePayload, getInitialAppView, MERGE_STAGES } from "./app-state.mjs";
import { createApiClient } from "./api-client.mjs";
import { markdownToHtml } from "./markdown.mjs";

const page = document.querySelector(".login-page");
const loginPanel = document.querySelector(".login-panel");
const noteShell = document.querySelector("#note-shell");
const loginForm = document.querySelector("#login-form");
const passwordInput = document.querySelector("#password");
const loginButton = document.querySelector("#login-button");
const loginMessage = document.querySelector("#login-message");
const togglePasswordButton = document.querySelector("#toggle-password");
const logoutButton = document.querySelector("#logout-button");
const saveStatus = document.querySelector("#save-status");
const noteList = document.querySelector("#note-list");
const noteSearch = document.querySelector("#note-search");
const newNoteButton = document.querySelector("#new-note-button");
const noteTitle = document.querySelector("#note-title");
const noteTags = document.querySelector("#note-tags");
const noteBody = document.querySelector("#note-body");
const noteUpdated = document.querySelector("#note-updated");
const manualSaveButton = document.querySelector("#manual-save-button");
const mergeNoteButton = document.querySelector("#merge-note-button");
const deleteNoteButton = document.querySelector("#delete-note-button");
const editorPane = document.querySelector(".editor-pane");
const evolutionPane = document.querySelector("#evolution-pane");
const editorWorkbench = document.querySelector("#editor-workbench");
const markdownPreview = document.querySelector("#markdown-preview");
const exportFormat = document.querySelector("#export-format");
const exportButton = document.querySelector("#export-button");
const backToEditorButton = document.querySelector("#back-to-editor-button");
const scanNoteButton = document.querySelector("#scan-note-button");
const evolutionStatus = document.querySelector("#evolution-status");
const l2Summary = document.querySelector("#l2-summary");
const l3Keywords = document.querySelector("#l3-keywords");
const blockList = document.querySelector("#block-list");
const suggestionList = document.querySelector("#suggestion-list");
const formatButtons = document.querySelectorAll("[data-format]");
const modeButtons = document.querySelectorAll("[data-mode]");
const appViews = document.querySelectorAll("[data-app-view]");
const appTargetButtons = document.querySelectorAll("[data-app-target]");
const dashboardNewNoteButton = document.querySelector("#dashboard-new-note-button");
const dashboardTaskList = document.querySelector("#dashboard-task-list");
const uploadPlaceholderButton = document.querySelector("#upload-placeholder-button");
const mergeTaskTitle = document.querySelector("#merge-task-title");
const mergeTaskStatus = document.querySelector("#merge-task-status");
const mergeTaskId = document.querySelector("#merge-task-id");
const mergeTaskStarted = document.querySelector("#merge-task-started");
const mergeTaskElapsed = document.querySelector("#merge-task-elapsed");
const mergeStageTrack = document.querySelector("#merge-stage-track");
const mergeDetailList = document.querySelector("#merge-detail-list");
const mergeBlockCount = document.querySelector("#merge-block-count");
const mergeBlockList = document.querySelector("#merge-block-list");
const mergeLiveStatus = document.querySelector("#merge-live-status");
const downloadMergeResultButton = document.querySelector("#download-merge-result-button");
const generateLongTextButton = document.querySelector("#generate-long-text-button");
const reviewTotalCount = document.querySelector("#review-total-count");
const reviewSubtitle = document.querySelector("#review-subtitle");
const reviewTaskList = document.querySelector("#review-task-list");
const reviewDetail = document.querySelector("#review-detail");

let authToken = "";
const apiClient = createApiClient({ getToken: () => authToken });
let notes = [];
let activeNoteId = null;
let saveTimer = null;
let mergingNoteIds = new Set();
let mergePendingNoteIds = new Set();
let evolutionPollTimers = new Map();
let activeWorkspace = "editor";
let activeAppView = getInitialAppView();
let initialWorkspace = window.location.pathname === "/evolution" ? "evolution" : "editor";
let mergePollTimer = null;
let currentMergeTask = createEmptyMergeTask();
let expandedMergeDetailKeys = new Set(["entity"]);
let mergeJobs = [];
let reviewTasks = [];
let activeReviewJobId = 0;
let activeReviewTaskId = 0;

function showApp(isLoggedIn) {
  loginPanel.hidden = isLoggedIn;
  noteShell.hidden = !isLoggedIn;
  page.classList.toggle("is-authenticated", isLoggedIn);

  if (isLoggedIn) {
    setAppView(activeAppView);
    if (activeAppView === "notes") {
      setWorkspaceView(initialWorkspace);
    } else {
      activeWorkspace = initialWorkspace;
      editorPane.hidden = activeWorkspace !== "editor";
      evolutionPane.hidden = activeWorkspace !== "evolution";
    }
    loadNotes()
      .then(() => {
        if (activeAppView === "notes" && activeWorkspace === "editor") {
          noteTitle.focus();
        }
      })
      .catch(showError);
    return;
  }

  passwordInput.focus();
}

function setMessage(text, isError = false) {
  loginMessage.textContent = text;
  loginMessage.classList.toggle("error", isError);
}

function setSaveStatus(text) {
  saveStatus.textContent = text;
}

function showError(error) {
  setSaveStatus(error.message || "操作失败");
}

function setAppView(view, shouldPushUrl = false) {
  activeAppView = view;

  for (const section of appViews) {
    section.hidden = section.dataset.appView !== view;
  }

  for (const button of appTargetButtons) {
    button.classList.toggle("is-active", button.dataset.appTarget === view);
  }

  if (shouldPushUrl) {
    const nextUrl = view === "home" ? "/" : view === "notes" ? "/notes" : `/#${view}`;
    window.history.pushState({ appView: view }, "", nextUrl);
  }
}

function setEvolutionStatus(text) {
  evolutionStatus.textContent = text;
}

function setNoteMerging(noteId, isMerging) {
  if (!noteId) {
    return;
  }

  if (isMerging) {
    mergingNoteIds.add(noteId);
    mergePendingNoteIds.delete(noteId);
  } else {
    mergingNoteIds.delete(noteId);
  }

  renderNotes();
  renderDashboard();
}

function setNoteMergePending(noteId, isPending) {
  if (!noteId) {
    return;
  }

  if (isPending) {
    mergePendingNoteIds.add(noteId);
  } else {
    mergePendingNoteIds.delete(noteId);
  }

  renderNotes();
  renderDashboard();
}

function hasPendingMergeSuggestions(state) {
  return (state?.suggestions || []).some((suggestion) => suggestion.status === "pending");
}

function countPendingMergeSuggestions(state) {
  return (state?.suggestions || []).filter((suggestion) => suggestion.status === "pending").length;
}

function syncMergePendingFromState(noteId, state) {
  setNoteMergePending(noteId, hasPendingMergeSuggestions(state));
}

async function refreshPendingMergeNotes() {
  try {
    const suggestions = await apiRequest("/evolution/suggestions");
    mergePendingNoteIds = new Set(
      suggestions
        .filter((suggestion) => suggestion.status === "pending")
        .map((suggestion) => suggestion.source_note_id)
    );
    renderNotes();
    renderDashboard();
  } catch {
    mergePendingNoteIds = new Set();
    renderDashboard();
  }
}

function clearEvolutionPoll(noteId) {
  const timer = evolutionPollTimers.get(noteId);

  if (timer) {
    window.clearTimeout(timer);
    evolutionPollTimers.delete(noteId);
  }
}

function clearEvolutionPolls() {
  for (const timer of evolutionPollTimers.values()) {
    window.clearTimeout(timer);
  }

  evolutionPollTimers = new Map();
}

function setMergeButtonsDisabled(isDisabled) {
  mergeNoteButton.disabled = isDisabled;
}

function setWorkspaceView(view, shouldPushUrl = false) {
  setAppView("notes");
  activeWorkspace = view;
  editorPane.hidden = view !== "editor";
  evolutionPane.hidden = view !== "evolution";

  if (shouldPushUrl) {
    window.history.pushState({ workspace: view }, "", view === "evolution" ? "/evolution" : "/notes");
  }
}

async function apiRequest(path, options = {}) {
  return apiClient.request(path, options);
}

async function login(password) {
  return apiClient.login(password);
}

async function loadNotes() {
  setSaveStatus("正在加载");
  mergingNoteIds = new Set();
  mergePendingNoteIds = new Set();
  const [loadedNotes] = await Promise.all([
    apiRequest("/notes"),
    loadMergeJobs(),
  ]);
  notes = loadedNotes;

  if (notes.length === 0) {
    const firstNote = await apiRequest("/notes", {
      method: "POST",
      body: JSON.stringify(createSampleNotePayload()),
    });
    notes = [firstNote];
  }

  activeNoteId = notes[0]?.id || null;
  renderNotes();
  openNote(activeNoteId);
  await refreshPendingMergeNotes();
  renderDashboard();
  setSaveStatus("已保存");
}

async function loadMergeJobs() {
  try {
    const result = await apiRequest("/evorag/ingest-jobs?limit=20");
    mergeJobs = result.jobs || [];
    renderDashboard();
  } catch {
    mergeJobs = [];
    renderDashboard();
  }
}

function getActiveNote() {
  return notes.find((note) => note.id === activeNoteId) || null;
}

function hasActiveNoteChanges(note) {
  return noteTitle.value !== note.title || noteTags.value !== note.tags || noteBody.value !== note.body;
}

function formatDate(value) {
  const date = new Date(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function getNoteTitle(note) {
  return note.title.trim() || "无标题笔记";
}

function getNotePreview(note) {
  return note.body
    .replace(/```[\s\S]*?```/g, "代码块")
    .replace(/[#>*_`[\]-]/g, "")
    .trim() || note.tags.trim() || "还没有内容";
}

function updatePreview() {
  markdownPreview.innerHTML = markdownToHtml(noteBody.value);
}

function renderNotes() {
  const keyword = noteSearch.value.trim().toLowerCase();
  const filteredNotes = notes.filter((note) => {
    const haystack = `${note.title} ${note.tags} ${note.body}`.toLowerCase();
    return haystack.includes(keyword);
  });

  noteList.innerHTML = "";

  if (filteredNotes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "没有找到笔记";
    noteList.append(empty);
    return;
  }

  for (const note of filteredNotes) {
    const button = document.createElement("button");
    button.className = `note-item${note.id === activeNoteId ? " is-active" : ""}`;
    button.type = "button";
    button.addEventListener("click", async () => {
      await saveActiveNote();
      setWorkspaceView("editor", true);
      openNote(note.id);
    });

    const titleRow = document.createElement("span");
    titleRow.className = "note-item-title-row";

    const title = document.createElement("span");
    title.className = "note-item-title";
    title.textContent = getNoteTitle(note);

    titleRow.append(title);

    if (mergingNoteIds.has(note.id)) {
      const badge = document.createElement("span");
      badge.className = "note-item-status";
      badge.textContent = "正在合并";
      titleRow.append(badge);
    } else if (mergePendingNoteIds.has(note.id)) {
      const badge = document.createElement("span");
      badge.className = "note-item-status is-actionable";
      badge.role = "button";
      badge.tabIndex = 0;
      badge.title = "打开合并确认";
      badge.textContent = "合并待确认";
      badge.addEventListener("click", (event) => {
        event.stopPropagation();
        openPendingMergeReview(note.id).catch(showError);
      });
      badge.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") {
          return;
        }

        event.preventDefault();
        event.stopPropagation();
        openPendingMergeReview(note.id).catch(showError);
      });
      titleRow.append(badge);
    }

    const preview = document.createElement("span");
    preview.className = "note-item-preview";
    preview.textContent = getNotePreview(note);

    const meta = document.createElement("span");
    meta.className = "note-item-meta";
    meta.textContent = formatDate(note.updated_at);

    button.append(titleRow, preview, meta);
    noteList.append(button);
  }
}

function renderDashboard() {
  if (!dashboardTaskList) {
    return;
  }

  dashboardTaskList.innerHTML = "";

  if (mergeJobs.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.className = "task-empty";
    cell.colSpan = 7;
    cell.textContent = "暂无任务";
    row.append(cell);
    dashboardTaskList.append(row);
    return;
  }

  for (const job of mergeJobs.slice(0, 5)) {
    dashboardTaskList.append(createDashboardTaskRow(job));
  }
}

function createDashboardTaskRow(job) {
  const row = document.createElement("tr");
  const progress = job.progress || {};
  const reviewCount = progress.needs_review || 0;
  const status = mergeStatusLabel(job.status, job.status);
  const progressValue = Math.round(progress.percent || 0);
  const progressText = `${progress.done || 0}/${progress.total || job.queued_count || 0}`;
  const taskName = job.task_name || `合并任务 ${job.id}`;
  const createdAt = job.created_at ? formatDate(job.created_at) : "-";

  row.innerHTML = `
    <td>
      <strong>${escapeText(taskName)}</strong>
      <span>${job.source_note_id ? `笔记ID: ${escapeText(job.source_note_id)}` : "合并生成"} · ${createdAt}</span>
    </td>
    <td><span class="task-status ${statusClass(status)}">${status}</span></td>
    <td>
      <div class="task-progress">
        <span><i style="width: ${progressValue}%"></i></span>
        <em>${progressText}</em>
      </div>
    </td>
    <td><strong>--</strong><span>${createdAt}</span></td>
    <td>${job.entity_count || 0}</td>
    <td>${reviewCount ? `<button class="review-count-button" type="button">${reviewCount}</button>` : "-"}</td>
    <td><button class="table-action" type="button">${job.status === "processing" || job.status === "queued" ? "查看进度" : "查看结果"}</button></td>
  `;

  row.querySelector(".table-action").addEventListener("click", () => {
    openMergeJob(job.id).catch(showError);
  });

  row.querySelector(".review-count-button")?.addEventListener("click", () => {
    openReviewForJob(job.id).catch(showError);
  });

  return row;
}

function escapeText(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}

function statusClass(status) {
  if (status === "处理中") {
    return "is-running";
  }

  if (status === "待确认" || status === "需人工确认") {
    return "is-review";
  }

  return "is-done";
}

function renderMergeTask() {
  if (!mergeStageTrack) {
    return;
  }

  const task = currentMergeTask;
  const job = task.job;
  const preprocess = task.preprocess;
  const progress = job?.progress || {};
  const statusLabel = mergeStatusLabel(task.status, job?.status);

  mergeTaskTitle.textContent = task.title || "合并生成";
  mergeTaskStatus.textContent = statusLabel;
  mergeTaskStatus.className = `merge-status ${mergeStatusClass(task.status, job?.status)}`;
  mergeTaskId.textContent = `任务ID: ${task.jobId ? `task_${String(task.jobId).padStart(8, "0")}` : "-"}`;
  mergeTaskStarted.textContent = `开始时间: ${task.startedAt ? formatDateTime(task.startedAt) : "-"}`;
  mergeTaskElapsed.textContent = `已用时: ${task.startedAt ? elapsedText(task.startedAt) : "-"}`;
  mergeLiveStatus.textContent = task.status === "idle" ? "实时状态：等待任务" : `实时状态：${statusLabel}`;
  downloadMergeResultButton.disabled = true;
  generateLongTextButton.disabled = !isMergeTerminal(task);

  renderMergeStages(task);
  renderMergeDetails(task, progress);
  renderMergeBlocks(preprocess, task);
}

function renderMergeStages(task) {
  mergeStageTrack.innerHTML = "";
  const stageStates = getMergeStageStates(task);

  for (const stage of MERGE_STAGES) {
    const item = document.createElement("div");
    item.className = `merge-stage is-${stageStates[stage.key]?.state || "pending"}`;

    const dot = document.createElement("span");
    dot.className = "merge-stage-dot";

    const label = document.createElement("strong");
    label.textContent = stage.label;

    const time = document.createElement("em");
    time.textContent = stageStates[stage.key]?.time || "-";

    item.append(dot, label, time);
    mergeStageTrack.append(item);
  }
}

function getMergeStageStates(task) {
  const preprocess = task.preprocess;
  const job = task.job;
  const status = job?.status || task.status;
  const terminal = isMergeTerminal(task);
  const processing = status === "queued" || status === "processing" || task.status === "submitting";
  const progress = job?.progress || {};
  const hasWritten = (progress.finished || 0) + (progress.needs_review || 0) > 0;
  const timings = preprocess?.timings || {};

  return {
    resource: {
      state: task.status === "idle" ? "pending" : task.workerStatus || preprocess ? "done" : "active",
      time: task.workerStatus || preprocess ? "ok" : task.status === "idle" ? "-" : "检查中",
    },
    split: {
      state: preprocess ? "done" : task.status === "submitting" ? "active" : "pending",
      time: timingText((timings.entity_anchor_split_ms || 0) + (timings.physical_chunk_split_ms || 0)),
    },
    entity: {
      state: preprocess ? "done" : task.status === "submitting" ? "active" : "pending",
      time: timingText(timings.entity_extraction_ms),
    },
    normalize: {
      state: job ? "done" : preprocess ? "active" : "pending",
      time: job ? `${job.queued_count || 0} 项` : "-",
    },
    resolve: {
      state: terminal ? "done" : processing && job ? "active" : "pending",
      time: job ? `${progress.done || 0}/${progress.total || job.queued_count || 0}` : "-",
    },
    mysql: {
      state: terminal ? "done" : hasWritten ? "active" : "pending",
      time: hasWritten ? `${progress.finished || 0} 写入` : "-",
    },
    index: {
      state: terminal ? "done" : hasWritten ? "active" : "pending",
      time: hasWritten ? "同步中" : "-",
    },
    done: {
      state: status === "failed" ? "error" : terminal ? "done" : "pending",
      time: terminal ? "完成" : "-",
    },
  };
}

function renderMergeDetails(task, progress) {
  mergeDetailList.innerHTML = "";
  for (const row of buildMergeDetailRows(task, progress)) {
    mergeDetailList.append(createMergeDetailRow(row));
  }
}

function buildMergeDetailRows(task, progress) {
  const preprocess = task.preprocess;
  const job = task.job;
  const timings = preprocess?.timings || {};
  const blocks = preprocess?.blocks || [];
  const entityCount = countPreprocessEntities(preprocess);
  const statusCounts = job?.status_counts || {};
  const worker = task.workerStatus || {};
  const resourceWarnings = worker.warnings || [];

  return [
    {
      key: "resource",
      index: 1,
      title: "资源准备",
      state: task.workerStatus || preprocess ? "done" : task.status === "idle" ? "pending" : "active",
      summary: resourceWarnings.length ? `资源检查有 ${resourceWarnings.length} 条警告` : "MySQL / Redis / Worker 状态已检查",
      time: task.workerStatus || preprocess ? "ok" : "-",
      details: [
        `Worker: ${worker.worker?.started ? "已启动" : worker.worker?.enabled === false ? "未启用" : "检查中"}`,
        `Redis 队列: ${worker.queue?.available ? "可用" : "等待检查"}`,
        `MySQL: ${worker.mysql ? "可访问" : "等待检查"}`,
        ...resourceWarnings,
      ],
    },
    {
      key: "split",
      index: 2,
      title: "Block 切分",
      state: preprocess ? "done" : task.status === "submitting" ? "active" : "pending",
      summary: preprocess ? `语义切分 ${blocks.length} 个 block，物理二切 ${countPhysicalChunks(blocks)} 个长 block` : "等待文本切分",
      time: timingText((timings.entity_anchor_split_ms || 0) + (timings.physical_chunk_split_ms || 0)),
      details: [
        `启动到切分: ${timingText(timings.startup_to_first_block_split_ms)}`,
        `实体锚点预切分: ${timingText(timings.entity_anchor_split_ms)}`,
        `物理二切: ${timingText(timings.physical_chunk_split_ms)}`,
      ],
    },
    {
      key: "entity",
      index: 3,
      title: "实体抽取",
      state: preprocess ? "done" : task.status === "submitting" ? "active" : "pending",
      summary: preprocess ? `抽取实体 ${entityCount} 个，属性与 evidence 同步抽取` : "并发处理 block 中",
      time: timingText(timings.entity_extraction_ms),
      progress: preprocess ? 100 : task.status === "submitting" ? 42 : 0,
      details: [
        `Block 数: ${blocks.length || "-"}`,
        `抽取实体: ${entityCount || "-"}`,
        "属性与 evidence: 同步抽取",
        `失败重试: ${countBlockWarnings(blocks)}`,
      ],
    },
    {
      key: "normalize",
      index: 4,
      title: "规范去重",
      state: job ? "done" : preprocess ? "active" : "pending",
      summary: job ? `同批实体去重后入队 ${job.queued_count || 0} 项` : "等待实体抽取完成",
      time: job ? `${job.queued_count || 0} 项` : "-",
      details: [`原始实体: ${entityCount || "-"}`, `入队实体: ${job?.queued_count ?? "-"}`],
    },
    {
      key: "resolve",
      index: 5,
      title: "候选消歧",
      state: isMergeTerminal(task) ? "done" : job ? "active" : "pending",
      summary: job ? `别名缓存、ES、Milvus 与 RRF 融合，已处理 ${progress.done || 0}/${progress.total || job.queued_count || 0}` : "等待规范去重",
      time: job ? `${Math.round(progress.percent || 0)}%` : "-",
      progress: progress.percent || 0,
      details: [
        `待处理: ${statusCounts.pending || 0}`,
        `处理中: ${statusCounts.processing || 0}`,
        `自动合并: ${statusCounts.auto_merged || 0}`,
        `新建实体: ${statusCounts.new_created || 0}`,
        `需人工确认: ${statusCounts.needs_review || 0}`,
      ],
    },
    {
      key: "mysql",
      index: 6,
      title: "写入 MySQL",
      state: isMergeTerminal(task) ? "done" : progress.finished ? "active" : "pending",
      summary: job ? `已完成写入/合并 ${progress.finished || 0} 项` : "等待候选消歧",
      time: job ? `${progress.finished || 0}` : "-",
      details: ["实体、属性、evidence、resolution audit 会在这一阶段落库"],
    },
    {
      key: "index",
      index: 7,
      title: "索引更新",
      state: isMergeTerminal(task) ? "done" : progress.finished ? "active" : "pending",
      summary: job ? "实体写入后同步更新 ES / Milvus / alias cache" : "等待 MySQL 写入",
      time: isMergeTerminal(task) ? "完成" : "-",
      details: ["当前后端按实体处理结果逐项更新索引"],
    },
  ];
}

function createMergeDetailRow(row) {
  const item = document.createElement("article");
  item.className = `merge-detail-row is-${row.state}`;

  const button = document.createElement("button");
  button.className = "merge-detail-toggle";
  button.type = "button";
  button.setAttribute("aria-expanded", String(expandedMergeDetailKeys.has(row.key)));

  const icon = document.createElement("span");
  icon.className = "merge-row-icon";

  const body = document.createElement("span");
  body.className = "merge-row-body";

  const title = document.createElement("strong");
  title.textContent = `${row.index}. ${row.title}`;

  const summary = document.createElement("span");
  summary.textContent = row.summary;

  body.append(title, summary);

  const time = document.createElement("em");
  time.textContent = row.time || "-";

  const arrow = document.createElement("span");
  arrow.className = "merge-row-arrow";
  arrow.textContent = "⌄";

  button.append(icon, body, time, arrow);

  const details = document.createElement("div");
  details.className = "merge-detail-extra";
  details.hidden = !expandedMergeDetailKeys.has(row.key);

  if (row.progress !== undefined) {
    const progressBar = document.createElement("div");
    progressBar.className = "merge-inline-progress";
    progressBar.innerHTML = `<span><i style="width: ${Math.max(0, Math.min(100, row.progress))}%"></i></span><em>${Math.round(row.progress)}%</em>`;
    details.append(progressBar);
  }

  const list = document.createElement("div");
  list.className = "merge-detail-metrics";
  for (const detail of row.details || []) {
    const metric = document.createElement("span");
    metric.textContent = detail;
    list.append(metric);
  }
  details.append(list);

  button.addEventListener("click", () => {
    if (expandedMergeDetailKeys.has(row.key)) {
      expandedMergeDetailKeys.delete(row.key);
    } else {
      expandedMergeDetailKeys.add(row.key);
    }
    renderMergeTask();
  });

  item.append(button, details);
  return item;
}

function renderMergeBlocks(preprocess, task) {
  const blocks = preprocess?.blocks || [];
  mergeBlockCount.textContent = `共 ${blocks.length} 个 block`;
  mergeBlockList.innerHTML = "";

  if (!blocks.length) {
    const empty = document.createElement("p");
    empty.className = "merge-empty";
    empty.textContent = task.status === "submitting" ? "正在切分和抽取 block..." : "暂无 block 任务";
    mergeBlockList.append(empty);
    return;
  }

  blocks.forEach((blockResult, index) => {
    const item = document.createElement("div");
    const warningCount = (blockResult.warnings || []).length;
    item.className = `merge-block-row ${warningCount ? "is-warning" : "is-done"}`;

    const label = document.createElement("span");
    label.textContent = `Block ${index + 1}`;

    const entityCount = document.createElement("strong");
    entityCount.textContent = `${(blockResult.entities || []).length} 实体`;

    const time = document.createElement("em");
    time.textContent = warningCount ? `${warningCount} 警告` : "已完成";

    item.append(label, entityCount, time);
    mergeBlockList.append(item);
  });
}

async function startMergePolling(jobId) {
  clearMergePoll();
  await refreshMergeTask(jobId);
}

async function refreshMergeTask(jobId) {
  const [job, workerStatus] = await Promise.all([
    apiRequest(`/evorag/ingest-jobs/${jobId}`),
    apiRequest("/evorag/worker-status").catch((error) => ({ warnings: [error.message] })),
  ]);
  currentMergeTask.job = job;
  currentMergeTask.workerStatus = workerStatus;
  currentMergeTask.status = job.status || "processing";
  currentMergeTask.jobId = job.id || jobId;
  currentMergeTask.noteId = job.source_note_id || currentMergeTask.noteId;
  currentMergeTask.title = job.task_name || currentMergeTask.title || `合并任务 ${jobId}`;
  currentMergeTask.startedAt = job.created_at ? new Date(job.created_at) : currentMergeTask.startedAt;
  mergeJobs = mergeJobs.map((item) => (Number(item.id) === Number(job.id) ? job : item));
  if (!mergeJobs.some((item) => Number(item.id) === Number(job.id))) {
    mergeJobs.unshift(job);
  }
  renderMergeTask();
  renderDashboard();

  if (!isMergeTerminal(currentMergeTask)) {
    mergePollTimer = window.setTimeout(() => {
      refreshMergeTask(jobId).catch((error) => {
        currentMergeTask.error = error.message;
        currentMergeTask.status = "failed";
        if (currentMergeTask.noteId) {
          setNoteMerging(currentMergeTask.noteId, false);
        }
        renderMergeTask();
      });
    }, 1500);
  } else {
    if (currentMergeTask.noteId) {
      setNoteMerging(currentMergeTask.noteId, false);
    }
    setSaveStatus(mergeStatusLabel(currentMergeTask.status, job.status));
    await refreshPendingMergeNotes();
    await loadMergeJobs();
  }
}

function clearMergePoll() {
  if (mergePollTimer) {
    window.clearTimeout(mergePollTimer);
    mergePollTimer = null;
  }
}

function isMergeTerminal(task) {
  const status = task.job?.status || task.status;
  return status === "completed" || status === "needs_review" || status === "failed";
}

function mergeStatusLabel(taskStatus, jobStatus) {
  const status = jobStatus || taskStatus;
  if (status === "submitting") return "处理中";
  if (status === "queued") return "排队中";
  if (status === "processing") return "处理中";
  if (status === "completed") return "已完成";
  if (status === "needs_review") return "需人工确认";
  if (status === "failed") return "失败";
  return "等待任务";
}

function mergeStatusClass(taskStatus, jobStatus) {
  const status = jobStatus || taskStatus;
  if (status === "completed") return "is-done";
  if (status === "needs_review") return "is-review";
  if (status === "failed") return "is-error";
  if (status === "queued" || status === "processing" || status === "submitting") return "is-running";
  return "is-idle";
}

function timingText(value) {
  const ms = Number(value || 0);
  if (!ms) return "-";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)}s`;
}

function formatDateTime(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function elapsedText(startedAt) {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${minutes} 分 ${rest} 秒`;
}

function countPreprocessEntities(preprocess) {
  return (preprocess?.blocks || []).reduce((total, block) => total + (block.entities || []).length, 0);
}

function countPhysicalChunks(blocks) {
  return blocks.filter((item) => (item.block?.chunk_count || 1) > 1).length;
}

function countBlockWarnings(blocks) {
  return blocks.reduce((total, block) => total + (block.warnings || []).length, 0);
}

async function openReviewForJob(jobId) {
  activeReviewJobId = Number(jobId);
  activeReviewTaskId = 0;
  setAppView("entities", true);
  await loadReviewTasksForJob(activeReviewJobId);
}

async function loadReviewTasksForJob(jobId) {
  reviewSubtitle.textContent = jobId
    ? `任务 task_${String(jobId).padStart(8, "0")} 的待确认实体。`
    : "全部待确认实体。";
  reviewTaskList.innerHTML = "";
  reviewDetail.innerHTML = '<p class="review-empty">正在加载待确认内容...</p>';

  const result = await apiRequest("/evorag/review-tasks?status=pending&limit=100");
  reviewTasks = (result.tasks || []).filter((task) => !jobId || Number(task.job_id) === Number(jobId));
  if (!activeReviewTaskId && reviewTasks.length) {
    activeReviewTaskId = Number(reviewTasks[0].id);
  }
  renderReviewTasks();
}

function renderReviewTasks() {
  reviewTotalCount.textContent = `(${reviewTasks.length})`;
  reviewTaskList.innerHTML = "";

  if (!reviewTasks.length) {
    reviewTaskList.innerHTML = '<p class="review-empty">这个任务暂无需要人工确认的实体。</p>';
    reviewDetail.innerHTML = '<p class="review-empty">暂无需要确认的实体。</p>';
    return;
  }

  if (!reviewTasks.some((task) => Number(task.id) === Number(activeReviewTaskId))) {
    activeReviewTaskId = Number(reviewTasks[0].id);
  }

  for (const task of reviewTasks) {
    const incoming = task.incoming_snapshot || {};
    const candidate = (task.candidates || [])[0] || {};
    const button = document.createElement("button");
    button.className = `review-task-item${Number(task.id) === Number(activeReviewTaskId) ? " is-active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <strong>${escapeText(incoming.name || "未命名实体")}</strong>
      <span>置信度: ${formatScore(candidate.score)}</span>
      <em>待确认</em>
    `;
    button.addEventListener("click", () => {
      activeReviewTaskId = Number(task.id);
      renderReviewTasks();
    });
    reviewTaskList.append(button);
  }

  const activeTask = reviewTasks.find((task) => Number(task.id) === Number(activeReviewTaskId)) || reviewTasks[0];
  renderReviewDetail(activeTask);
}

function renderReviewDetail(task) {
  const incoming = task.incoming_snapshot || {};
  const candidates = task.candidates || [];
  const candidate = candidates[0] || {};

  const defaultAction = candidate.id ? "merge" : "new";
  reviewDetail.innerHTML = `
    <div class="review-detail-head">
      <div>
        <p class="section-kicker">实体合并判断</p>
        <h2>${escapeText(incoming.name || "未命名实体")} ${candidate.canonical_name ? `与 ${escapeText(candidate.canonical_name)}` : ""}</h2>
      </div>
      <span>置信度 ${formatScore(candidate.score)}</span>
    </div>

    <div class="entity-compare-grid">
      <article class="entity-summary-card">
        <span>系统提取的内容</span>
        <h3>${escapeText(incoming.name || "未命名实体")}</h3>
        <p>${escapeText(summarizeIncomingEntity(incoming))}</p>
      </article>
      <article class="entity-summary-card">
        <span>候选已有实体</span>
        <h3>${escapeText(candidate.canonical_name || "暂无候选实体")}</h3>
        <p>${escapeText(summarizeCandidateEntity(candidate))}</p>
      </article>
    </div>

    <div class="review-actions-panel">
      <span>可选操作</span>
      <button class="review-choice ${defaultAction === "merge" ? "is-primary" : ""}" type="button" data-review-action="merge" ${candidate.id ? "" : "disabled"}>
        合并到：${escapeText(candidate.canonical_name || "候选实体")}
      </button>
      <button class="review-choice ${defaultAction === "new" ? "is-primary" : ""}" type="button" data-review-action="new">创建为新实体</button>
      <button class="review-choice" type="button" data-review-action="reject" ${candidate.id ? "" : "disabled"}>拒绝当前候选</button>
      <textarea id="review-reason" placeholder="补充说明，可选"></textarea>
      <div class="review-action-row">
        <button class="ghost-button" type="button" data-review-action="refresh">刷新</button>
        <button class="primary-button" type="button" data-review-action="confirm">确认</button>
      </div>
    </div>
  `;

  let selectedAction = defaultAction;
  const reasonInput = reviewDetail.querySelector("#review-reason");

  for (const button of reviewDetail.querySelectorAll("[data-review-action]")) {
    button.addEventListener("click", async () => {
      const action = button.dataset.reviewAction;
      if (action === "refresh") {
        await loadReviewTasksForJob(activeReviewJobId);
        return;
      }
      if (action === "confirm") {
        await submitReviewDecision(task, selectedAction, reasonInput.value, candidate);
        return;
      }
      selectedAction = action;
      for (const choice of reviewDetail.querySelectorAll(".review-choice")) {
        choice.classList.toggle("is-primary", choice.dataset.reviewAction === selectedAction);
      }
    });
  }
}

async function submitReviewDecision(task, action, reason, candidate) {
  const reviewTaskId = Number(task.id);
  if (action === "merge") {
    await apiRequest(`/evorag/review-tasks/${reviewTaskId}/merge`, {
      method: "POST",
      body: JSON.stringify({ entity_id: Number(candidate.id), reason, decided_by: "manual" }),
    });
  } else if (action === "new") {
    await apiRequest(`/evorag/review-tasks/${reviewTaskId}/new`, {
      method: "POST",
      body: JSON.stringify({ reason, decided_by: "manual" }),
    });
  } else {
    await apiRequest(`/evorag/review-tasks/${reviewTaskId}/reject`, {
      method: "POST",
      body: JSON.stringify({ candidate_entity_ids: [Number(candidate.id)], reason, decided_by: "manual" }),
    });
  }

  setSaveStatus("已提交人工确认");
  await loadReviewTasksForJob(activeReviewJobId);
  if (currentMergeTask.jobId === activeReviewJobId) {
    await refreshMergeTask(activeReviewJobId);
  }
  await loadMergeJobs();
}

function summarizeIncomingEntity(entity) {
  const attrs = entity.attributes || [];
  const definition = attrs.find((item) => item.attr_type === "definition")?.value_text;
  return entity.identity_description || definition || entity.description_for_match || "暂无概述";
}

function summarizeCandidateEntity(entity) {
  return entity.identity_description || entity.summary || entity.description_for_match || "暂无概述";
}

function formatScore(value) {
  const score = Number(value || 0);
  return score ? score.toFixed(2) : "-";
}

function openNote(noteId) {
  window.clearTimeout(saveTimer);
  const note = notes.find((item) => item.id === noteId) || notes[0];

  if (!note) {
    return;
  }

  activeNoteId = note.id;
  noteTitle.value = note.title;
  noteTags.value = note.tags;
  noteBody.value = note.body;
  noteUpdated.textContent = `更新于 ${formatDate(note.updated_at)}`;
  setSaveStatus("已保存");
  updatePreview();
  renderNotes();
  renderDashboard();

  if (activeWorkspace === "evolution") {
    renderEvolutionState(null);
    loadEvolutionState(note.id).catch(showError);
  }
}

async function saveActiveNote() {
  const note = getActiveNote();

  if (!note) {
    return null;
  }

  window.clearTimeout(saveTimer);

  if (!hasActiveNoteChanges(note)) {
    setSaveStatus("已保存");
    return note;
  }

  setSaveStatus("正在保存");

  const updated = await apiRequest(`/notes/${note.id}`, {
    method: "PUT",
    body: JSON.stringify({
      title: noteTitle.value,
      tags: noteTags.value,
      body: noteBody.value,
    }),
  });

  const noteIndex = notes.findIndex((item) => item.id === updated.id);

  if (noteIndex === -1) {
    notes.unshift(updated);
  } else {
    notes = notes.map((item) => (item.id === updated.id ? updated : item));
  }

  activeNoteId = updated.id;
  noteUpdated.textContent = `更新于 ${formatDate(updated.updated_at)}`;
  setSaveStatus("已保存");
  updatePreview();
  renderNotes();
  renderDashboard();

  if (activeWorkspace === "evolution") {
    loadEvolutionState(updated.id).catch(showError);
  }

  return updated;
}

function queueSave() {
  window.clearTimeout(saveTimer);
  setSaveStatus("正在输入");
  updatePreview();
  saveTimer = window.setTimeout(() => {
    saveActiveNote().catch(showError);
  }, 450);
}

async function addNewNote() {
  await saveActiveNote();
  const note = await apiRequest("/notes", {
    method: "POST",
    body: JSON.stringify({ title: "", tags: "", body: "" }),
  });
  notes.unshift(note);
  noteSearch.value = "";
  renderNotes();
  renderDashboard();
  openNote(note.id);
  setWorkspaceView("editor", true);
}

async function deleteActiveNote() {
  const note = getActiveNote();

  if (!note) {
    return;
  }

  const shouldDelete = window.confirm(`删除「${getNoteTitle(note)}」吗？`);

  if (!shouldDelete) {
    return;
  }

  await apiRequest(`/notes/${note.id}`, { method: "DELETE" });
  notes = notes.filter((item) => item.id !== note.id);

  if (notes.length === 0) {
    const next = await apiRequest("/notes", {
      method: "POST",
      body: JSON.stringify({ title: "", tags: "", body: "" }),
    });
    notes.push(next);
  }

  activeNoteId = notes[0].id;
  renderNotes();
  renderDashboard();
  openNote(activeNoteId);
}

async function loadEvolutionState(noteId, options = {}) {
  if (!noteId) {
    renderEvolutionState(null);
    return null;
  }

  setEvolutionStatus("正在加载");
  const state = await apiRequest(`/evolution/notes/${noteId}`);

  if (noteId !== activeNoteId) {
    return state;
  }

  renderEvolutionState(state);
  syncMergePendingFromState(noteId, state);

  if (state.analysis_status === "queued" || state.analysis_status === "running") {
    setNoteMerging(noteId, true);
    setEvolutionStatus("后台分析中");
    pollEvolutionScan(noteId, options);
    return state;
  }

  setNoteMerging(noteId, false);
  syncMergePendingFromState(noteId, state);

  if (state.stale && options.startIfStale) {
    await startEvolutionScan(noteId, options);
    return state;
  }

  setEvolutionStatus(state.stale ? "内容已更新，等待整理" : "已分析");
  return state;
}

async function startEvolutionScan(noteId, options = {}) {
  if (!noteId) {
    return null;
  }

  const job = await apiRequest(`/evolution/notes/${noteId}/scan`, {
    method: "POST",
  });

  if (noteId === activeNoteId) {
    setEvolutionStatus(job.status === "failed" ? `分析失败：${job.error || "请稍后重试"}` : "后台分析中");

    if (options.updateSaveStatus) {
      setSaveStatus("已保存，后台整理中");
    }
  }

  if (job.status === "queued" || job.status === "running") {
    setNoteMerging(noteId, true);
    pollEvolutionScan(noteId, options);
  } else {
    setNoteMerging(noteId, false);
  }

  return job;
}

function pollEvolutionScan(noteId, options = {}) {
  clearEvolutionPoll(noteId);

  const tick = async () => {
    const job = await apiRequest(`/evolution/notes/${noteId}/scan/status`);

    if (job.status === "queued" || job.status === "running") {
      setNoteMerging(noteId, true);

      if (noteId === activeNoteId) {
        setEvolutionStatus("后台分析中");
      }

      evolutionPollTimers.set(noteId, window.setTimeout(tick, 1500));
      return;
    }

    clearEvolutionPoll(noteId);
    setNoteMerging(noteId, false);

    if (job.status === "succeeded") {
      const state = await apiRequest(`/evolution/notes/${noteId}`);
      const pendingCount = countPendingMergeSuggestions(state);
      syncMergePendingFromState(noteId, state);
      renderDashboard();

      if (noteId === activeNoteId) {
        if (pendingCount > 0 && options.openReviewWhenPending) {
          setWorkspaceView("evolution", true);
        }

        renderEvolutionState(state);
        setEvolutionStatus(pendingCount > 0 ? `扫描完成，发现 ${pendingCount} 条合并建议` : "扫描完成，没有发现需要合并的内容");

        if (options.updateSaveStatus) {
          setSaveStatus(pendingCount > 0 ? `已保存，${pendingCount} 条合并待确认` : "已保存，无需合并");
        }
      }
      return;
    }

    if (job.status === "failed") {
      if (noteId === activeNoteId) {
        setEvolutionStatus(`分析失败：${job.error || "请稍后重试"}`);
      }

      if (options.updateSaveStatus) {
        setSaveStatus("已保存，整理失败");
      }
    }
  };

  evolutionPollTimers.set(noteId, window.setTimeout(() => {
    tick().catch(showError);
  }, 1500));
}

async function scanActiveNote() {
  const note = await saveActiveNote();

  if (!note) {
    return;
  }

  setEvolutionStatus("正在提交后台扫描");
  await startEvolutionScan(note.id);
}

async function submitActiveNote() {
  const note = await saveActiveNote();

  if (!note) {
    return;
  }

  setSaveStatus("已保存，后台整理中");
  await startEvolutionScan(note.id, { updateSaveStatus: true });
}

async function mergeActiveNote() {
  setMergeButtonsDisabled(true);
  let mergeNoteId = "";

  try {
    const note = await saveActiveNote();

    if (!note) {
      return;
    }
    mergeNoteId = note.id;

    if (!note.body.trim()) {
      setSaveStatus("当前笔记为空，无法合并");
      return;
    }

    clearMergePoll();
    setSaveStatus("已保存，正在提交合并任务");
    setNoteMerging(note.id, true);
    currentMergeTask = {
      ...createEmptyMergeTask(),
      noteId: note.id,
      title: `${getNoteTitle(note)} 知识合并`,
      status: "submitting",
      startedAt: new Date(),
    };
    setAppView("merge", true);
    renderMergeTask();

    const response = await apiRequest("/evorag/ingest", {
      method: "POST",
      body: JSON.stringify({
        text: note.body,
        note_id: note.id,
        task_name: `${getNoteTitle(note)} 知识合并`,
      }),
    });

    currentMergeTask.preprocess = response.preprocess;
    currentMergeTask.jobId = response.job_id;
    currentMergeTask.status = response.status || "queued";
    currentMergeTask.job = {
      id: response.job_id,
      status: response.status || "queued",
      block_count: response.preprocess?.blocks?.length || 0,
      entity_count: countPreprocessEntities(response.preprocess),
      queued_count: response.queued_count,
      progress: {
        total: response.queued_count,
        active: response.queued_count,
        finished: 0,
        needs_review: 0,
        failed: 0,
        done: 0,
        percent: response.queued_count ? 0 : 100,
      },
      status_counts: response.queued_count ? { pending: response.queued_count } : {},
    };
    renderMergeTask();

    if (response.job_id) {
      await startMergePolling(response.job_id);
      await loadMergeJobs();
    }
  } catch (error) {
    if (mergeNoteId) {
      setNoteMerging(mergeNoteId, false);
    }
    currentMergeTask.status = "failed";
    currentMergeTask.error = error.message || "合并任务失败";
    renderMergeTask();
    throw error;
  } finally {
    setMergeButtonsDisabled(false);
  }
}

async function openMergeJob(jobId) {
  clearMergePoll();
  setAppView("merge", true);
  currentMergeTask = {
    ...createEmptyMergeTask(),
    status: "processing",
    jobId: Number(jobId),
    title: `合并任务 ${jobId}`,
    startedAt: new Date(),
  };
  renderMergeTask();
  await refreshMergeTask(Number(jobId));
}

async function openPendingMergeReview(noteId) {
  await saveActiveNote();
  openNote(noteId);
  setWorkspaceView("evolution", true);
  renderEvolutionState(null);
  await loadEvolutionState(noteId);
}

async function openEvolutionView() {
  setWorkspaceView("evolution", true);
  renderEvolutionState(null);

  try {
    const note = await saveActiveNote();
    await loadEvolutionState(note?.id || activeNoteId);
  } catch (error) {
    setEvolutionStatus("加载失败");
    throw error;
  }
}

async function openEditorView() {
  setWorkspaceView("editor", true);
  noteTitle.focus();
}

function renderEvolutionState(state) {
  blockList.innerHTML = "";
  suggestionList.innerHTML = "";
  l3Keywords.innerHTML = "";

  if (!state || !state.representation) {
    l2Summary.textContent = "保存后生成摘要";
    setEvolutionStatus(state?.analysis_status === "running" ? "后台分析中" : "等待分析");
    renderEmpty(blockList, "暂无 block");
    renderEmpty(suggestionList, "暂无建议");
    return;
  }

  l2Summary.textContent = state.representation.l2_summary || "暂无摘要";

  for (const keyword of state.representation.keywords || []) {
    const badge = document.createElement("span");
    badge.className = "keyword-badge";
    badge.textContent = keyword;
    l3Keywords.append(badge);
  }

  if (!l3Keywords.children.length) {
    renderEmpty(l3Keywords, "暂无关键词");
  }

  const blocks = state.blocks || [];
  const claims = state.claims || [];
  const suggestions = state.suggestions || [];
  const claimsByBlock = new Map();

  for (const claim of claims) {
    const blockClaims = claimsByBlock.get(claim.block_id) || [];
    blockClaims.push(claim);
    claimsByBlock.set(claim.block_id, blockClaims);
  }

  if (!blocks.length) {
    renderEmpty(blockList, "没有提取到 block");
  }

  blocks.forEach((block, index) => {
    blockList.append(createBlockItem(block, claimsByBlock.get(block.id) || [], index === 0));
    claimsByBlock.delete(block.id);
  });

  const orphanClaims = [...claimsByBlock.values()].flat();

  if (orphanClaims.length) {
    blockList.append(createBlockItem({ id: "orphan", heading: "未归属 Block", block_index: blocks.length }, orphanClaims, false));
  }

  if (!suggestions.length) {
    renderEmpty(suggestionList, "没有发现需要合并或去重的内容");
  }

  for (const suggestion of suggestions) {
    suggestionList.append(createSuggestionItem(suggestion));
  }
}

function createBlockItem(block, claims, expanded) {
  const item = document.createElement("article");
  item.className = "block-item";

  const toggle = document.createElement("button");
  toggle.className = "block-toggle";
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", String(expanded));

  const heading = document.createElement("span");
  heading.className = "block-heading";
  heading.textContent = block.heading || `Block ${Number(block.block_index || 0) + 1}`;

  const meta = document.createElement("span");
  meta.className = "block-meta";
  meta.textContent = `Block ${Number(block.block_index || 0) + 1} · ${claims.length} 条 claim`;

  const indicator = document.createElement("span");
  indicator.className = "block-indicator";
  indicator.setAttribute("aria-hidden", "true");
  indicator.textContent = expanded ? "−" : "+";

  const title = document.createElement("span");
  title.className = "block-title";
  title.append(heading, meta);
  toggle.append(title, indicator);

  const summary = document.createElement("p");
  summary.className = "block-summary";
  summary.textContent = block.l2_summary || "暂无 block 摘要";

  const details = document.createElement("div");
  details.className = "block-details";
  details.id = `block-details-${block.id}`;
  details.hidden = !expanded;
  toggle.setAttribute("aria-controls", details.id);

  const source = document.createElement("section");
  source.className = "block-source";

  const sourceLabel = document.createElement("span");
  sourceLabel.className = "block-detail-label";
  sourceLabel.textContent = "Block 原文";

  const sourceText = document.createElement("p");
  sourceText.textContent = block.l1_text || "暂无原文";
  source.append(sourceLabel, sourceText);

  const claimLabel = document.createElement("span");
  claimLabel.className = "block-detail-label";
  claimLabel.textContent = `Claims（${claims.length}）`;

  const claimContainer = document.createElement("div");
  claimContainer.className = "block-claims";

  if (!claims.length) {
    renderEmpty(claimContainer, "这个 block 暂未提取到 claim");
  }

  for (const claim of claims) {
    claimContainer.append(createClaimItem(claim));
  }

  details.append(source, claimLabel, claimContainer);

  toggle.addEventListener("click", () => {
    const nextExpanded = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(nextExpanded));
    details.hidden = !nextExpanded;
    indicator.textContent = nextExpanded ? "−" : "+";
  });

  item.append(toggle, summary, details);
  return item;
}

function createClaimItem(claim) {
  const item = document.createElement("article");
  item.className = "claim-item";

  const text = document.createElement("p");
  text.textContent = claim.claim_text;

  const meta = document.createElement("span");
  meta.textContent = `${claim.subject || "未知主题"} · ${claim.predicate || "知识点"}`;

  item.append(text, meta);
  return item;
}

function renderEmpty(container, text) {
  const empty = document.createElement("p");
  empty.className = "mini-empty";
  empty.textContent = text;
  container.append(empty);
}

function createSuggestionItem(suggestion) {
  const item = document.createElement("article");
  item.className = `suggestion-item is-${suggestion.relation}`;

  const header = document.createElement("div");
  header.className = "suggestion-title";

  const relation = document.createElement("strong");
  relation.textContent = relationLabel(suggestion.relation);

  const score = document.createElement("span");
  score.textContent = `${Math.round(suggestion.confidence * 100)}% · ${riskLabel(suggestion.risk_level)}`;

  header.append(relation, score);

  const reason = document.createElement("p");
  reason.textContent = suggestion.reason;

  const patch = document.createElement("div");
  patch.className = "patch-preview";
  patch.textContent = suggestion.patch.content || suggestion.patch.source_claim || "等待确认";

  const actions = document.createElement("div");
  actions.className = "suggestion-actions";

  const applyButton = document.createElement("button");
  applyButton.className = "secondary-button";
  applyButton.type = "button";
  applyButton.textContent = suggestion.risk_level === "high" ? "确认记录" : "接受";
  applyButton.addEventListener("click", () => applySuggestion(suggestion.id).catch(showError));

  const rejectButton = document.createElement("button");
  rejectButton.className = "ghost-button";
  rejectButton.type = "button";
  rejectButton.textContent = "忽略";
  rejectButton.addEventListener("click", () => rejectSuggestion(suggestion.id).catch(showError));

  actions.append(applyButton, rejectButton);
  item.append(header, reason, patch, actions);
  return item;
}

async function applySuggestion(suggestionId) {
  setEvolutionStatus("正在应用");
  await apiRequest(`/evolution/suggestions/${suggestionId}/apply`, { method: "POST" });
  notes = await apiRequest("/notes");
  const current = notes.find((note) => note.id === activeNoteId);

  if (current) {
    openNote(current.id);
  } else {
    renderNotes();
    await loadEvolutionState(activeNoteId);
  }

  await refreshPendingMergeNotes();
  setSaveStatus("已应用建议");
}

async function rejectSuggestion(suggestionId) {
  await apiRequest(`/evolution/suggestions/${suggestionId}/reject`, { method: "POST" });
  await loadEvolutionState(activeNoteId);
  await refreshPendingMergeNotes();
  setEvolutionStatus("已忽略");
}

function relationLabel(relation) {
  if (relation === "duplicate") {
    return "可能重复";
  }

  if (relation === "conflict") {
    return "可能冲突";
  }

  return "可补充";
}

function riskLabel(risk) {
  if (risk === "high") {
    return "高风险";
  }

  if (risk === "medium") {
    return "需确认";
  }

  return "低风险";
}

async function exportActiveNote() {
  const note = await saveActiveNote();

  if (!note) {
    return;
  }

  const format = exportFormat.value;
  const { blob, filename } = await apiClient.exportNote(note.id, format);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setSaveStatus("已导出");
}

function replaceSelection(nextValue, cursorOffset = nextValue.length) {
  const start = noteBody.selectionStart;
  const end = noteBody.selectionEnd;
  const before = noteBody.value.slice(0, start);
  const after = noteBody.value.slice(end);

  noteBody.value = `${before}${nextValue}${after}`;
  noteBody.focus();
  noteBody.setSelectionRange(start + cursorOffset, start + cursorOffset);
  queueSave();
}

function prefixSelectedLines(prefixFactory) {
  const start = noteBody.selectionStart;
  const end = noteBody.selectionEnd;
  const value = noteBody.value;
  const lineStart = value.lastIndexOf("\n", start - 1) + 1;
  const lineEndIndex = value.indexOf("\n", end);
  const lineEnd = lineEndIndex === -1 ? value.length : lineEndIndex;
  const block = value.slice(lineStart, lineEnd);
  const lines = block.split("\n");
  const nextBlock = lines.map((line, index) => `${prefixFactory(index)}${line}`).join("\n");

  noteBody.value = `${value.slice(0, lineStart)}${nextBlock}${value.slice(lineEnd)}`;
  noteBody.focus();
  noteBody.setSelectionRange(lineStart, lineStart + nextBlock.length);
  queueSave();
}

function insertMarkdown(format) {
  const selected = noteBody.value.slice(noteBody.selectionStart, noteBody.selectionEnd);

  if (format === "heading") {
    prefixSelectedLines(() => "# ");
    return;
  }

  if (format === "bullet") {
    prefixSelectedLines(() => "- ");
    return;
  }

  if (format === "numbered") {
    prefixSelectedLines((index) => `${index + 1}. `);
    return;
  }

  if (format === "nested") {
    prefixSelectedLines(() => "  - ");
    return;
  }

  if (format === "check") {
    prefixSelectedLines(() => "- [ ] ");
    return;
  }

  if (format === "quote") {
    prefixSelectedLines(() => "> ");
    return;
  }

  if (format === "inline-code") {
    replaceSelection(`\`${selected || "code"}\``, selected ? selected.length + 2 : 5);
    return;
  }

  if (format === "code-block") {
    const content = selected || "写代码";
    replaceSelection(`\`\`\`js\n${content}\n\`\`\``, selected ? content.length + 7 : 9);
  }
}

function setViewMode(mode) {
  editorWorkbench.dataset.viewMode = mode;

  for (const button of modeButtons) {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  }
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const password = passwordInput.value.trim();

  if (!password) {
    setMessage("请输入访问密码", true);
    return;
  }

  loginButton.disabled = true;
  setMessage("正在登录...");

  try {
    const result = await login(password);
    authToken = result.access_token;
    noteShell.dataset.token = result.access_token;
    passwordInput.value = "";
    setMessage("");
    showApp(true);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    loginButton.disabled = false;
  }
});

togglePasswordButton.addEventListener("click", () => {
  const shouldShow = passwordInput.type === "password";
  passwordInput.type = shouldShow ? "text" : "password";
  togglePasswordButton.setAttribute("aria-label", shouldShow ? "隐藏密码" : "显示密码");
});

logoutButton.addEventListener("click", () => {
  authToken = "";
  delete noteShell.dataset.token;
  notes = [];
  activeNoteId = null;
  mergingNoteIds = new Set();
  mergePendingNoteIds = new Set();
  clearEvolutionPolls();
  clearMergePoll();
  currentMergeTask = createEmptyMergeTask();
  renderMergeTask();
  showApp(false);
});

for (const button of appTargetButtons) {
  button.addEventListener("click", () => {
    const target = button.dataset.appTarget;

    if (target === "notes") {
      setWorkspaceView("editor", true);
      return;
    }

    setAppView(target, true);
    if (target === "merge") {
      renderMergeTask();
    } else if (target === "entities") {
      openReviewForJob(0).catch(showError);
    }
  });
}

dashboardNewNoteButton.addEventListener("click", () => addNewNote().catch(showError));
uploadPlaceholderButton.addEventListener("click", () => {
  setSaveStatus("文件上传功能稍后接入");
});

newNoteButton.addEventListener("click", () => addNewNote().catch(showError));
manualSaveButton.addEventListener("click", () => saveActiveNote().catch(showError));
mergeNoteButton.addEventListener("click", () => mergeActiveNote().catch(showError));
deleteNoteButton.addEventListener("click", () => deleteActiveNote().catch(showError));
exportButton.addEventListener("click", () => exportActiveNote().catch(showError));
backToEditorButton.addEventListener("click", () => openEditorView().catch(showError));
scanNoteButton.addEventListener("click", () => scanActiveNote().catch(showError));
noteSearch.addEventListener("input", renderNotes);
noteBody.addEventListener("keydown", (event) => {
  if (event.key !== "Tab") {
    return;
  }

  event.preventDefault();

  if (event.shiftKey) {
    const start = noteBody.selectionStart;
    const lineStart = noteBody.value.lastIndexOf("\n", start - 1) + 1;
    const beforeLine = noteBody.value.slice(0, lineStart);
    const line = noteBody.value.slice(lineStart);
    const nextLine = line.replace(/^ {1,2}/, "");
    noteBody.value = `${beforeLine}${nextLine}`;
    noteBody.setSelectionRange(Math.max(lineStart, start - 2), Math.max(lineStart, start - 2));
    queueSave();
    return;
  }

  replaceSelection("  ");
});

for (const field of [noteTitle, noteTags, noteBody]) {
  field.addEventListener("input", queueSave);
}

for (const button of formatButtons) {
  button.addEventListener("click", () => insertMarkdown(button.dataset.format));
}

for (const button of modeButtons) {
  button.addEventListener("click", () => setViewMode(button.dataset.mode));
}

window.addEventListener("popstate", () => {
  activeAppView = getInitialAppView();
  initialWorkspace = window.location.pathname === "/evolution" ? "evolution" : "editor";
  setAppView(activeAppView);

  if (activeAppView === "notes") {
    setWorkspaceView(initialWorkspace);
  }

  if (authToken && activeAppView === "notes" && activeWorkspace === "evolution") {
    loadEvolutionState(activeNoteId).catch(showError);
  }
});

showApp(false);
