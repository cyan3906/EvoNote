export function createApiClient({
  origin = window.location.origin,
  getToken = () => "",
  fetchImpl = fetch,
} = {}) {
  const apiBaseUrl = `${origin}/api`;

  async function request(path, options = {}) {
    const headers = new Headers(options.headers || {});

    if (options.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    const token = getToken();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }

    const response = await fetchImpl(`${apiBaseUrl}${path}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    if (response.status === 204) {
      return null;
    }

    return response.json();
  }

  async function login(password) {
    const response = await fetchImpl(`${apiBaseUrl}/auth/login`, {
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

  async function exportNote(noteId, format) {
    const token = getToken();
    const response = await fetchImpl(`${apiBaseUrl}/notes/${noteId}/export?format=${format}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error("导出失败");
    }

    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    return {
      blob,
      filename: parseExportFilename(disposition, format),
    };
  }

  return { request, login, exportNote };
}


async function readErrorMessage(response) {
  let message = "请求失败";

  try {
    const result = await response.json();
    message = result.detail || message;
  } catch {
    message = response.statusText || message;
  }

  return message;
}


export function parseExportFilename(disposition, format) {
  const filenameMatch = String(disposition || "").match(/filename="([^"]+)"/);
  return filenameMatch ? filenameMatch[1] : `note.${format}`;
}
