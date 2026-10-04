export const MERGE_STAGES = [
  { key: "resource", label: "资源准备" },
  { key: "split", label: "Block 切分" },
  { key: "entity", label: "实体抽取" },
  { key: "normalize", label: "规范去重" },
  { key: "resolve", label: "候选消歧" },
  { key: "mysql", label: "写入 MySQL" },
  { key: "index", label: "索引更新" },
  { key: "done", label: "完成" },
];


export function getInitialAppView(location = window.location) {
  if (location.pathname === "/notes" || location.pathname === "/evolution") {
    return "notes";
  }

  const hashView = location.hash.replace("#", "");
  const knownViews = new Set(["home", "merge", "entities", "graph", "history", "settings"]);
  return knownViews.has(hashView) ? hashView : "home";
}


export function createEmptyMergeTask() {
  return {
    noteId: "",
    title: "合并生成",
    status: "idle",
    jobId: 0,
    startedAt: null,
    preprocess: null,
    job: null,
    workerStatus: null,
    error: "",
  };
}


export function createSampleNotePayload() {
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
