from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

_PDF_PAGE_WIDTH = 595.28
_PDF_PAGE_HEIGHT = 841.89
_PDF_MARGIN_X = 62.36
_PDF_MARGIN_TOP = 62.36
_PDF_MARGIN_BOTTOM = 62.36


def note_filename(note: dict[str, str], extension: str) -> str:
    title = note["title"].strip() or "untitled-note"
    safe = "".join(
        char.lower() if char.isascii() and (char.isalnum() or char in ("-", "_")) else "-"
        for char in title
    )
    safe = "-".join(part for part in safe.split("-") if part)
    return f"{safe or 'untitled-note'}.{extension}"


def note_to_text(note: dict[str, str]) -> str:
    lines = [
        note["title"].strip() or "无标题笔记",
        "",
    ]

    if note["tags"].strip():
        lines.extend([f"标签: {note['tags'].strip()}", ""])

    lines.append(note["body"].rstrip())
    return "\n".join(lines).rstrip() + "\n"


def note_to_pdf(note: dict[str, str]) -> bytes:
    max_width = _PDF_PAGE_WIDTH - (_PDF_MARGIN_X * 2)
    pages: list[list[str]] = [[]]
    y = _PDF_PAGE_HEIGHT - _PDF_MARGIN_TOP

    def new_page() -> None:
        nonlocal y

        pages.append([])
        y = _PDF_PAGE_HEIGHT - _PDF_MARGIN_TOP

    def ensure_space(line_height: float) -> None:
        nonlocal y

        if y - line_height < _PDF_MARGIN_BOTTOM:
            new_page()

    def draw_line(
        text: str,
        size: int = 11,
        leading: int = 17,
        indent: int = 0,
        gray: float = 0,
    ) -> None:
        nonlocal y
        available_width = max_width - indent
        wrapped_lines = _wrap_pdf_text(text or " ", size, available_width)

        for wrapped in wrapped_lines:
            ensure_space(leading)
            pages[-1].extend(_pdf_text_commands(wrapped, _PDF_MARGIN_X + indent, y, size, gray))
            y -= leading

    draw_line(note["title"].strip() or "无标题笔记", size=21, leading=28)
    y -= 4

    if note["tags"].strip():
        draw_line(f"标签: {note['tags'].strip()}", size=9, leading=14, gray=0.38)
        y -= 8

    in_code_block = False

    for raw_line in note["body"].splitlines() or [""]:
        line = raw_line.expandtabs(2)
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if not stripped:
            y -= 8
            continue

        if in_code_block:
            draw_line(line, size=9, leading=14, indent=10, gray=0.15)
            continue

        if stripped.startswith("#"):
            hashes, _, heading = stripped.partition(" ")
            if heading and set(hashes) == {"#"}:
                size = max(12, 18 - min(len(hashes), 6))
                y -= 4
                draw_line(heading, size=size, leading=size + 7)
                y -= 2
                continue

        list_info = _parse_markdown_list(line)

        if list_info:
            marker = "1." if list_info["ordered"] else "-"
            indent = int(list_info["level"]) * 12
            draw_line(f"{marker} {list_info['content']}", size=11, leading=17, indent=indent)
            continue

        draw_line(line, size=11, leading=17)

    return _build_pdf(["\n".join(page).encode("ascii") for page in pages])


def _wrap_pdf_text(text: str, font_size: int, max_width: float) -> list[str]:
    normalized = text.replace("\r", "")
    lines: list[str] = []
    current = ""
    current_width = 0.0

    for char in normalized:
        char_width = _estimate_pdf_text_width(char, font_size)
        candidate_width = current_width + char_width

        if current and candidate_width > max_width:
            lines.append(current)
            current = char.lstrip() or char
            current_width = _estimate_pdf_text_width(current, font_size)
        else:
            current = f"{current}{char}"
            current_width = candidate_width

    lines.append(current)
    return lines or [" "]


def _estimate_pdf_text_width(text: str, font_size: int) -> float:
    width = 0.0

    for char in text:
        if char == " ":
            width += font_size * 0.32
        elif char.isascii():
            width += font_size * 0.56
        else:
            width += font_size

    return width


def _pdf_text_commands(text: str, x: float, y: float, size: int, gray: float = 0) -> list[str]:
    color = f"{gray:.2f} g " if gray else "0 g "
    commands: list[str] = []
    cursor_x = x

    for segment, is_ascii in _pdf_text_segments(text):
        font_name = "F1" if is_ascii else "F2"
        payload = _pdf_literal_text(segment) if is_ascii else f"<{_pdf_hex_text(segment)}>"
        commands.append(
            f"{color}BT /{font_name} {size} Tf 1 0 0 1 {cursor_x:.2f} {y:.2f} Tm {payload} Tj ET"
        )
        cursor_x += _estimate_pdf_text_width(segment, size)

    return commands or [f"{color}BT /F1 {size} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ( ) Tj ET"]


def _pdf_text_segments(text: str) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    current = ""
    current_is_ascii: bool | None = None

    for char in text:
        is_ascii = char.isascii()

        if current and is_ascii != current_is_ascii:
            segments.append((current, bool(current_is_ascii)))
            current = char
            current_is_ascii = is_ascii
        else:
            current = f"{current}{char}"
            current_is_ascii = is_ascii

    if current:
        segments.append((current, bool(current_is_ascii)))

    return segments


def _pdf_literal_text(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    escaped = escaped.replace("\n", " ").replace("\r", "")
    return f"({escaped})"


def _pdf_hex_text(text: str) -> str:
    return (b"\xfe\xff" + text.encode("utf-16-be", errors="replace")).hex().upper()


def _build_pdf(page_streams: list[bytes]) -> bytes:
    page_count = max(1, len(page_streams))
    page_ids = list(range(6, 6 + page_count))
    content_ids = list(range(6 + page_count, 6 + (page_count * 2)))
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Count {page_count} /Kids "
            f"[{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] >>"
        ).encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: (
            b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light "
            b"/Encoding /UniGB-UCS2-H /DescendantFonts [5 0 R] >>"
        ),
        5: (
            b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> /DW 1000 >>"
        ),
    }

    for index, page_id in enumerate(page_ids):
        content_id = content_ids[index]
        stream = page_streams[index] if index < len(page_streams) else b""
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PDF_PAGE_WIDTH:.2f} {_PDF_PAGE_HEIGHT:.2f}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )

    return _write_pdf_objects(objects)


def _write_pdf_objects(objects: dict[int, bytes]) -> bytes:
    highest_id = max(objects)
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (highest_id + 1)

    for object_id in range(1, highest_id + 1):
        offsets[object_id] = len(output)
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {highest_id + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")

    for object_id in range(1, highest_id + 1):
        output.extend(f"{offsets[object_id]:010d} 00000 n \n".encode("ascii"))

    output.extend(
        (
            f"trailer\n<< /Size {highest_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    return bytes(output)


def note_to_docx(note: dict[str, str]) -> bytes:
    paragraphs = [_docx_paragraph(note["title"].strip() or "无标题笔记", style="Title")]

    if note["tags"].strip():
        paragraphs.append(_docx_paragraph(f"标签: {note['tags'].strip()}"))

    in_code_block = False

    for line in note["body"].splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            paragraphs.append(_docx_paragraph(line, font="Consolas"))
            continue

        if stripped.startswith("#"):
            hashes, _, heading = stripped.partition(" ")
            if heading and set(hashes) == {"#"}:
                paragraphs.append(_docx_paragraph(heading, style=f"Heading{min(len(hashes), 4)}"))
                continue

        list_info = _parse_markdown_list(line)

        if list_info:
            marker = "1." if list_info["ordered"] else "-"
            prefix = "  " * int(list_info["level"])
            paragraphs.append(_docx_paragraph(f"{prefix}{marker} {list_info['content']}"))
            continue

        paragraphs.append(_docx_paragraph(line))

    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", _docx_content_types())
        package.writestr("_rels/.rels", _docx_root_relationships())
        package.writestr("word/document.xml", _docx_document("".join(paragraphs)))

    return buffer.getvalue()


def _parse_markdown_list(line: str) -> dict[str, object] | None:
    import re

    match = re.match(r"^(\s*)(?:([-+*])\s+\[([ xX])\]\s+|([-+*])\s+|(\d+)[.)]\s+)(.*)$", line)

    if not match:
        return None

    content = match.group(6)
    checked = match.group(3)

    if checked:
        content = f"[{'x' if checked.lower() == 'x' else ' '}] {content}"

    return {
        "level": len(match.group(1).replace("\t", "    ")) // 2,
        "ordered": bool(match.group(5)),
        "content": content,
    }


def _docx_paragraph(text: str, style: str | None = None, font: str | None = None) -> str:
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    font_xml = f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}"/>' if font else ""
    run_props = f"<w:rPr>{font_xml}</w:rPr>" if font_xml else ""
    paragraph_props = f"<w:pPr>{style_xml}</w:pPr>" if style_xml else ""

    return (
        "<w:p>"
        f"{paragraph_props}"
        "<w:r>"
        f"{run_props}"
        f'<w:t xml:space="preserve">{escape(text)}</w:t>'
        "</w:r>"
        "</w:p>"
    )


def _docx_document(body_xml: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body_xml}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""


def _docx_content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""


def _docx_root_relationships() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
