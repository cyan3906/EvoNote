import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const indexHtml = readFileSync(join(root, "index.html"), "utf8");
const appSource = readFileSync(join(root, "src", "app.mjs"), "utf8");

assert.match(indexHtml, /id="graph-query"/);
assert.match(indexHtml, /id="graph-load-candidates"/);
assert.match(indexHtml, /id="graph-generate-article"/);
assert.match(indexHtml, /id="graph-candidate-list"/);

assert.match(appSource, /\/evorag\/graph\/candidates/);
assert.match(appSource, /\/evorag\/graph\/generate/);

console.log("ok - graph page exposes candidate selection and generation controls");
