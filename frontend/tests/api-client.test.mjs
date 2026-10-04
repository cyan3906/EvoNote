import assert from "node:assert/strict";
import { createApiClient, parseExportFilename } from "../src/api-client.mjs";

function jsonResponse(body, init = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText || "",
    headers: new Map(Object.entries(init.headers || {})),
    json: async () => body,
    blob: async () => body,
  };
}

{
  const calls = [];
  const client = createApiClient({
    origin: "http://example.test",
    getToken: () => "token-123",
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return jsonResponse({ ok: true });
    },
  });

  const result = await client.request("/notes", { method: "POST", body: "{}" });

  assert.deepEqual(result, { ok: true });
  assert.equal(calls[0].url, "http://example.test/api/notes");
  assert.equal(calls[0].options.headers.get("Authorization"), "Bearer token-123");
  assert.equal(calls[0].options.headers.get("Content-Type"), "application/json");
  console.log("ok - request adds api base, auth, and json content type");
}

{
  const client = createApiClient({
    origin: "http://example.test",
    fetchImpl: async () => jsonResponse({ detail: "密码不正确" }, { ok: false, status: 401 }),
  });

  await assert.rejects(() => client.request("/notes"), /密码不正确/);
  console.log("ok - request surfaces server detail errors");
}

{
  const client = createApiClient({
    origin: "http://example.test",
    fetchImpl: async () => jsonResponse({}, { ok: false, status: 401 }),
  });

  await assert.rejects(() => client.login("bad-password"), /密码不正确/);
  console.log("ok - login maps unauthorized responses");
}

{
  const filename = parseExportFilename('attachment; filename="note.docx"', "pdf");

  assert.equal(filename, "note.docx");
  assert.equal(parseExportFilename("", "txt"), "note.txt");
  console.log("ok - export filename parser uses header with fallback");
}
