export function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function renderInline(value) {
  const codeSnippets = [];
  let html = escapeHtml(value);

  html = html.replace(/`([^`]+)`/g, (_, code) => {
    const index = codeSnippets.push(`<code>${code}</code>`) - 1;
    return `@@CODE${index}@@`;
  });

  html = html
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  return html.replace(/@@CODE(\d+)@@/g, (_, index) => codeSnippets[Number(index)]);
}

export function parseListLine(line) {
  const match = line.match(/^(\s*)(?:([-+*])\s+\[([ xX])\]\s+|([-+*])\s+|(\d+)[.)]\s+)(.*)$/);

  if (!match) {
    return null;
  }

  const indent = match[1].replace(/\t/g, "    ").length;

  return {
    level: Math.floor(indent / 2),
    type: match[5] ? "ol" : "ul",
    checked: match[3] ? match[3].toLowerCase() === "x" : null,
    content: match[6],
  };
}

function buildListTree(lines) {
  const root = { children: [] };
  const stack = [{ level: -1, node: root }];

  for (const line of lines) {
    const item = parseListLine(line);

    while (stack.length > 1 && stack[stack.length - 1].level >= item.level) {
      stack.pop();
    }

    const node = { ...item, children: [] };
    stack[stack.length - 1].node.children.push(node);
    stack.push({ level: item.level, node });
  }

  return root.children;
}

function renderListItem(item) {
  const checkbox = item.checked === null
    ? ""
    : `<input type="checkbox" disabled${item.checked ? " checked" : ""}> `;
  const children = item.children.length > 0 ? renderListGroups(item.children) : "";

  return `<li>${checkbox}${renderInline(item.content)}${children}</li>`;
}

function renderListGroups(nodes) {
  let html = "";
  let index = 0;

  while (index < nodes.length) {
    const type = nodes[index].type;
    const group = [];

    while (index < nodes.length && nodes[index].type === type) {
      group.push(nodes[index]);
      index += 1;
    }

    html += `<${type}>${group.map(renderListItem).join("")}</${type}>`;
  }

  return html;
}

function renderListBlock(lines) {
  return renderListGroups(buildListTree(lines));
}

export function markdownToHtml(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];

  function flushParagraph() {
    if (paragraph.length === 0) {
      return;
    }

    html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushParagraph();
      const code = [];
      index += 1;

      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        code.push(lines[index]);
        index += 1;
      }

      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);

    if (heading) {
      flushParagraph();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    if (/^---+$/.test(trimmed)) {
      flushParagraph();
      html.push("<hr>");
      continue;
    }

    if (trimmed.startsWith(">")) {
      flushParagraph();
      const quoteLines = [];

      while (index < lines.length && lines[index].trim().startsWith(">")) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }

      index -= 1;
      html.push(`<blockquote>${quoteLines.map(renderInline).join("<br>")}</blockquote>`);
      continue;
    }

    if (parseListLine(line)) {
      flushParagraph();
      const listLines = [];

      while (index < lines.length && parseListLine(lines[index])) {
        listLines.push(lines[index]);
        index += 1;
      }

      index -= 1;
      html.push(renderListBlock(listLines));
      continue;
    }

    paragraph.push(trimmed);
  }

  flushParagraph();
  return html.join("");
}
