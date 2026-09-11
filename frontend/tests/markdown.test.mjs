import assert from "node:assert/strict";
import { markdownToHtml, parseListLine } from "../src/markdown.mjs";

const tests = [
  {
    name: "renders h1 and h3 headings",
    run() {
      assert.equal(markdownToHtml("# 标题\n### 小标题"), "<h1>标题</h1><h3>小标题</h3>");
    },
  },
  {
    name: "joins paragraph lines",
    run() {
      assert.equal(markdownToHtml("第一行\n第二行"), "<p>第一行 第二行</p>");
    },
  },
  {
    name: "renders unordered list",
    run() {
      assert.equal(markdownToHtml("- A\n- B"), "<ul><li>A</li><li>B</li></ul>");
    },
  },
  {
    name: "renders nested unordered list",
    run() {
      assert.equal(
        markdownToHtml("- A\n  - A1\n  - A2\n- B"),
        "<ul><li>A<ul><li>A1</li><li>A2</li></ul></li><li>B</li></ul>",
      );
    },
  },
  {
    name: "renders ordered list",
    run() {
      assert.equal(markdownToHtml("1. A\n2. B"), "<ol><li>A</li><li>B</li></ol>");
    },
  },
  {
    name: "renders task list",
    run() {
      assert.equal(
        markdownToHtml("- [ ] Todo\n- [x] Done"),
        '<ul><li><input type="checkbox" disabled> Todo</li><li><input type="checkbox" disabled checked> Done</li></ul>',
      );
    },
  },
  {
    name: "renders fenced code block and escapes code",
    run() {
      assert.equal(
        markdownToHtml("```js\nif (a < b) {\n  run();\n}\n```"),
        "<pre><code>if (a &lt; b) {\n  run();\n}</code></pre>",
      );
    },
  },
  {
    name: "renders inline code, bold, italic, and link",
    run() {
      assert.equal(
        markdownToHtml("这是 `code`、**重点**、*补充*、[链接](https://example.com)"),
        '<p>这是 <code>code</code>、<strong>重点</strong>、<em>补充</em>、<a href="https://example.com" target="_blank" rel="noreferrer">链接</a></p>',
      );
    },
  },
  {
    name: "renders blockquote",
    run() {
      assert.equal(markdownToHtml("> 第一行\n> 第二行"), "<blockquote>第一行<br>第二行</blockquote>");
    },
  },
  {
    name: "escapes raw html",
    run() {
      assert.equal(markdownToHtml("<script>alert(1)</script>"), "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>");
    },
  },
];

for (const test of tests) {
  test.run();
  console.log(`ok - ${test.name}`);
}

assert.deepEqual(parseListLine("    - deep"), {
  level: 2,
  type: "ul",
  checked: null,
  content: "deep",
});

console.log(`\n${tests.length} markdown rendering tests passed`);
