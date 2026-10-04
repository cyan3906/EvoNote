import assert from "node:assert/strict";
import { createEmptyMergeTask, createSampleNotePayload, getInitialAppView, MERGE_STAGES } from "../src/app-state.mjs";

assert.equal(getInitialAppView({ pathname: "/notes", hash: "" }), "notes");
assert.equal(getInitialAppView({ pathname: "/", hash: "#merge" }), "merge");
assert.equal(getInitialAppView({ pathname: "/", hash: "#unknown" }), "home");
console.log("ok - initial app view follows route and known hash");

const task = createEmptyMergeTask();
assert.equal(task.status, "idle");
assert.equal(task.jobId, 0);
assert.equal(task.error, "");
console.log("ok - empty merge task keeps idle defaults");

const sample = createSampleNotePayload();
assert.equal(sample.title, "Markdown 示例");
assert.match(sample.body, /console\.log\('Evonote'\);/);
console.log("ok - sample note payload keeps markdown example");

assert.deepEqual(MERGE_STAGES.map((stage) => stage.key), [
  "resource",
  "split",
  "entity",
  "normalize",
  "resolve",
  "mysql",
  "index",
  "done",
]);
console.log("ok - merge stage order is stable");
