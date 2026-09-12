import { markdownToHtml } from "./markdown.mjs";

const API_BASE_URL = "http://127.0.0.1:8000/api";

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
const deleteNoteButton = document.querySelector("#delete-note-button");
const editorWorkbench = document.querySelector("#editor-workbench");
const markdownPreview = document.querySelector("#markdown-preview");
const exportFormat = document.querySelector("#export-format");
const exportButton = document.querySelector("#export-button");
const formatButtons = document.querySelectorAll("[data-format]");
const modeButtons = document.querySelectorAll("[data-mode]");

let authToken = "";
let notes = [];
let activeNoteId = null;
let saveTimer = null;

function createSampleNotePayload() {
  return {
    title: "Markdown 示例",
    tags: "markdown, evonote",
    body: [
      "# 今天的笔记",
      "",
      "- 先写一个想法",
      "  - 再补一个子想法",
      "  - 子列表可以继续展开",
      "- 把结论放在最后",
      "",
      "```js",
      "console.log('Evonote');",
      "```",
    ].join("\n"),
  };
}

function showApp(isLoggedIn) {
  loginPanel.hidden = isLoggedIn;
  noteShell.hidden = !isLoggedIn;
  page.classList.toggle("is-authenticated", isLoggedIn);

  if (isLoggedIn) {
    loadNotes().then(() => noteTitle.focus()).catch(showError);
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

async function apiRequest(path, options = {}) {
  const headers = new Headers(options.headers || {});

  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  if (authToken) {
    headers.set("Authorization", `Bearer ${authToken}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let message = "请求失败";

    try {
      const result = await response.json();
      message = result.detail || message;
    } catch {
      message = response.statusText || message;
    }

    throw new Error(message);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

async function login(password) {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ password }),
  });

  if (!response.ok) {
    throw new Error(response.status === 401 ? "密码不正确" : "登录失败，请稍后再试");
  }

  return response.json();
}

async function loadNotes() {
  setSaveStatus("正在加载");
  notes = await apiRequest("/notes");

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
  setSaveStatus("已保存");
}

function getActiveNote() {
  return notes.find((note) => note.id === activeNoteId) || null;
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
      openNote(note.id);
    });

    const title = document.createElement("span");
    title.className = "note-item-title";
    title.textContent = getNoteTitle(note);

    const preview = document.createElement("span");
    preview.className = "note-item-preview";
    preview.textContent = getNotePreview(note);

    const meta = document.createElement("span");
    meta.className = "note-item-meta";
    meta.textContent = formatDate(note.updated_at);

    button.append(title, preview, meta);
    noteList.append(button);
  }
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
}

async function saveActiveNote() {
  const note = getActiveNote();

  if (!note) {
    return null;
  }

  window.clearTimeout(saveTimer);
  setSaveStatus("正在保存");

  const updated = await apiRequest(`/notes/${note.id}`, {
    method: "PUT",
    body: JSON.stringify({
      title: noteTitle.value,
      tags: noteTags.value,
      body: noteBody.value,
    }),
  });

  notes = notes.filter((item) => item.id !== updated.id);
  notes.unshift(updated);
  activeNoteId = updated.id;
  noteUpdated.textContent = `更新于 ${formatDate(updated.updated_at)}`;
  setSaveStatus("已保存");
  updatePreview();
  renderNotes();
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
  openNote(note.id);
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
  openNote(activeNoteId);
}

async function exportActiveNote() {
  const note = await saveActiveNote();

  if (!note) {
    return;
  }

  const format = exportFormat.value;
  const response = await fetch(`${API_BASE_URL}/notes/${note.id}/export?format=${format}`, {
    headers: {
      Authorization: `Bearer ${authToken}`,
    },
  });

  if (!response.ok) {
    throw new Error("导出失败");
  }

  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") || "";
  const filenameMatch = disposition.match(/filename="([^"]+)"/);
  const filename = filenameMatch ? filenameMatch[1] : `note.${format}`;
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
  showApp(false);
});

newNoteButton.addEventListener("click", () => addNewNote().catch(showError));
manualSaveButton.addEventListener("click", () => saveActiveNote().catch(showError));
deleteNoteButton.addEventListener("click", () => deleteActiveNote().catch(showError));
exportButton.addEventListener("click", () => exportActiveNote().catch(showError));
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

showApp(false);
