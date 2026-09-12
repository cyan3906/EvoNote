import hashlib
import math
import re
from uuid import uuid4

from app.core import database

VECTOR_SIZE = 64
MAX_CANDIDATE_CLAIMS = 80
MAX_SUGGESTIONS = 8

STOP_WORDS = {
    "一个",
    "一种",
    "这个",
    "那个",
    "以及",
    "可以",
    "如果",
    "因为",
    "所以",
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
}

PREDICATES = (
    "不是",
    "不能",
    "不会",
    "没有",
    "支持",
    "属于",
    "包括",
    "包含",
    "适合",
    "用于",
    "导致",
    "依赖",
    "需要",
    "可以",
    "能够",
    "会",
    "是",
)

CONFLICT_PAIRS = (
    ("单线程", "多线程"),
    ("同步", "异步"),
    ("阻塞", "非阻塞"),
    ("有状态", "无状态"),
    ("强一致", "最终一致"),
    ("安全", "不安全"),
    ("可靠", "不可靠"),
    ("快", "慢"),
)


def scan_note(note_id: str) -> dict[str, object] | None:
    note = database.get_note(note_id)

    if not note:
        return None

    candidate_claims = database.list_claims(exclude_note_id=note_id)
    candidate_notes = {item["id"]: item for item in database.list_notes() if item["id"] != note_id}
    analysis = analyze_note(note)
    database.replace_note_analysis(
        note_id=note_id,
        representation=analysis["representation"],
        blocks=analysis["blocks"],
        claims=analysis["claims"],
    )
    database.supersede_pending_suggestions(note_id)
    suggestions = build_merge_suggestions(
        source_note=note,
        source_claims=analysis["claims"],
        candidate_claims=candidate_claims,
        candidate_notes=candidate_notes,
    )
    stored_suggestions = database.create_merge_suggestions(suggestions)

    return {
        "note": note,
        "representation": analysis["representation"],
        "blocks": analysis["blocks"],
        "claims": analysis["claims"],
        "suggestions": stored_suggestions,
    }


def get_note_evolution_state(note_id: str) -> dict[str, object] | None:
    note = database.get_note(note_id)

    if not note:
        return None

    representation = database.get_note_representation(note_id)

    if representation is None:
        return scan_note(note_id)

    return {
        "note": note,
        "representation": representation,
        "blocks": database.list_note_blocks(note_id),
        "claims": database.list_claims(note_id=note_id),
        "suggestions": database.list_merge_suggestions(source_note_id=note_id, status="pending"),
    }


def analyze_note(note: dict[str, str]) -> dict[str, object]:
    raw_blocks = split_note_blocks(note)
    blocks: list[dict[str, object]] = []
    claims: list[dict[str, object]] = []

    for block_index, raw_block in enumerate(raw_blocks):
        block_id = str(uuid4())
        l1_text = raw_block["text"].strip()
        keywords = extract_keywords(f"{raw_block['heading']} {l1_text}", limit=10)
        block = {
            "id": block_id,
            "note_id": note["id"],
            "block_index": block_index,
            "heading": raw_block["heading"],
            "l1_text": l1_text,
            "l2_summary": summarize_text(l1_text),
            "l3_text": " ".join(keywords),
            "keywords": keywords,
        }
        blocks.append(block)

        for claim_index, claim in enumerate(extract_claims(l1_text, keywords)):
            claim["id"] = str(uuid4())
            claim["note_id"] = note["id"]
            claim["block_id"] = block_id
            claim["claim_index"] = claim_index
            claims.append(claim)

    full_text = "\n".join(block["l1_text"] for block in blocks)
    note_keywords = extract_keywords(f"{note['title']} {note['tags']} {full_text}", limit=14)
    representation = {
        "note_id": note["id"],
        "l2_summary": summarize_text(full_text, limit=180),
        "l3_text": " ".join(note_keywords),
        "keywords": note_keywords,
        "vector": text_vector(" ".join(note_keywords)),
    }

    return {
        "representation": representation,
        "blocks": blocks,
        "claims": claims,
    }


def split_note_blocks(note: dict[str, str]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current_heading = note["title"].strip() or "无标题笔记"
    current_lines: list[str] = []

    for line in note["body"].splitlines():
        stripped = line.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

        if heading_match and current_lines:
            blocks.append({"heading": current_heading, "text": "\n".join(current_lines)})
            current_lines = []

        if heading_match:
            current_heading = heading_match.group(2).strip()
            current_lines.append(stripped)
            continue

        current_lines.append(line)

        if sum(len(item) for item in current_lines) > 900 and stripped == "":
            blocks.append({"heading": current_heading, "text": "\n".join(current_lines)})
            current_lines = []

    if current_lines or not blocks:
        blocks.append({"heading": current_heading, "text": "\n".join(current_lines)})

    return [block for block in blocks if block["text"].strip()] or [{"heading": current_heading, "text": ""}]


def summarize_text(text: str, limit: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", strip_markdown(text)).strip()

    if not cleaned:
        return "空白笔记，等待继续补充。"

    sentence_match = re.split(r"(?<=[。！？.!?])\s+", cleaned, maxsplit=1)
    summary = sentence_match[0] if sentence_match else cleaned

    if len(summary) < 24 and len(cleaned) > len(summary):
        summary = cleaned

    return summary[:limit].rstrip()


def extract_claims(text: str, fallback_keywords: list[str]) -> list[dict[str, object]]:
    claims: list[dict[str, object]] = []
    clean_lines = []
    in_code_block = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = re.sub(r"^[-+*]\s+(\[[ xX]\]\s+)?", "", stripped)
        stripped = re.sub(r"^\d+[.)]\s+", "", stripped)

        if stripped:
            clean_lines.append(stripped)

    fragments = re.split(r"[。！？!?；;]\s*|\n+", "\n".join(clean_lines))

    for fragment in fragments:
        claim_text = re.sub(r"\s+", " ", fragment).strip(" -")

        if len(claim_text) < 6:
            continue

        subject, predicate, object_text = parse_claim_parts(claim_text, fallback_keywords)
        keywords = extract_keywords(claim_text, limit=8) or fallback_keywords[:4]
        claims.append(
            {
                "claim_text": claim_text,
                "subject": subject,
                "predicate": predicate,
                "object_text": object_text,
                "source_text": claim_text,
                "keywords": keywords,
                "vector": text_vector(" ".join(keywords + [claim_text])),
                "confidence": 0.72,
            }
        )

    return dedupe_claims(claims)


def parse_claim_parts(text: str, fallback_keywords: list[str]) -> tuple[str, str, str]:
    for predicate in PREDICATES:
        if predicate not in text:
            continue

        left, right = text.split(predicate, 1)
        subject = left.strip(" ，,：:") or (fallback_keywords[0] if fallback_keywords else text[:12])
        object_text = right.strip(" ，,：:") or text
        return subject[:48], predicate, object_text[:120]

    subject = fallback_keywords[0] if fallback_keywords else text[:12]
    return subject[:48], "related_to", text[:120]


def dedupe_claims(claims: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    seen: set[str] = set()

    for claim in claims:
        fingerprint = normalize_text(str(claim["claim_text"]))

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        unique.append(claim)

    return unique[:24]


def build_merge_suggestions(
    source_note: dict[str, str],
    source_claims: list[dict[str, object]],
    candidate_claims: list[dict[str, object]],
    candidate_notes: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    scored_candidates = rank_candidate_claims(source_claims, candidate_claims)
    suggestions: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for source_claim, target_claim, score in scored_candidates:
        target_note = candidate_notes.get(str(target_claim["note_id"]))

        if not target_note:
            continue

        relation = classify_relation(source_claim, target_claim, score)

        if relation == "unrelated":
            continue

        key = (relation, str(source_claim["claim_text"]), str(target_claim["note_id"]))

        if key in seen_keys:
            continue

        seen_keys.add(key)
        patch = build_patch(source_note, target_note, source_claim, target_claim, relation)
        suggestions.append(
            {
                "id": str(uuid4()),
                "source_note_id": source_note["id"],
                "target_note_id": target_note["id"],
                "source_claim_id": source_claim["id"],
                "target_claim_id": target_claim["id"],
                "relation": relation,
                "confidence": round(score, 3),
                "risk_level": patch["risk_level"],
                "reason": patch["reason"],
                "patch": patch,
            }
        )

    return sorted(suggestions, key=lambda item: item["confidence"], reverse=True)[:MAX_SUGGESTIONS]


def rank_candidate_claims(
    source_claims: list[dict[str, object]],
    candidate_claims: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object], float]]:
    ranked: list[tuple[dict[str, object], dict[str, object], float]] = []

    for source_claim in source_claims:
        for target_claim in candidate_claims[:MAX_CANDIDATE_CLAIMS]:
            score = claim_similarity(source_claim, target_claim)

            if score >= 0.38:
                ranked.append((source_claim, target_claim, score))

    return sorted(ranked, key=lambda item: item[2], reverse=True)


def claim_similarity(left: dict[str, object], right: dict[str, object]) -> float:
    left_keywords = set(left.get("keywords", []))
    right_keywords = set(right.get("keywords", []))
    keyword_score = jaccard(left_keywords, right_keywords)
    vector_score = cosine(left.get("vector", []), right.get("vector", []))
    subject_score = 0.2 if shared_entity(str(left.get("subject", "")), str(right.get("subject", ""))) else 0
    text_score = 0.15 if normalize_text(str(left["claim_text"])) == normalize_text(str(right["claim_text"])) else 0

    return min(1.0, (vector_score * 0.52) + (keyword_score * 0.38) + subject_score + text_score)


def classify_relation(source_claim: dict[str, object], target_claim: dict[str, object], score: float) -> str:
    if has_conflict_signal(str(source_claim["claim_text"]), str(target_claim["claim_text"])):
        return "conflict"

    if score >= 0.88:
        return "duplicate"

    if score >= 0.46:
        return "supplement"

    return "unrelated"


def build_patch(
    source_note: dict[str, str],
    target_note: dict[str, str],
    source_claim: dict[str, object],
    target_claim: dict[str, object],
    relation: str,
) -> dict[str, object]:
    if relation == "duplicate":
        return {
            "operation": "mark_duplicate",
            "risk_level": "low",
            "requires_user_review": False,
            "target_note_id": target_note["id"],
            "content": source_claim["claim_text"],
            "reason": f"新知识点与《{target_note['title'] or '无标题笔记'}》中的已有 claim 高度重复。",
            "source_claim": source_claim["claim_text"],
            "target_claim": target_claim["claim_text"],
        }

    if relation == "conflict":
        return {
            "operation": "create_conflict_review",
            "risk_level": "high",
            "requires_user_review": True,
            "target_note_id": target_note["id"],
            "content": source_claim["claim_text"],
            "reason": f"新 claim 与《{target_note['title'] or '无标题笔记'}》中的 claim 可能存在条件、版本或表述冲突。",
            "source_claim": source_claim["claim_text"],
            "target_claim": target_claim["claim_text"],
        }

    return {
        "operation": "append_claim",
        "risk_level": "medium",
        "requires_user_review": True,
        "target_note_id": target_note["id"],
        "content": source_claim["claim_text"],
        "reason": f"新笔记可以补充到《{target_note['title'] or '无标题笔记'}》，旧 claim 没有覆盖这个新增信息。",
        "source_claim": source_claim["claim_text"],
        "target_claim": target_claim["claim_text"],
    }


def apply_suggestion(suggestion_id: str) -> dict[str, object] | None:
    suggestion = database.get_merge_suggestion(suggestion_id)

    if not suggestion or suggestion["status"] != "pending":
        return suggestion

    patch = suggestion["patch"]
    operation = patch.get("operation")
    target_note_id = str(patch.get("target_note_id") or suggestion["target_note_id"])
    target_note = database.get_note(target_note_id)

    if not target_note:
        database.update_merge_suggestion_status(suggestion_id, "rejected")
        return database.get_merge_suggestion(suggestion_id)

    if operation == "append_claim":
        database.save_note_version(target_note_id, reason=f"apply merge suggestion {suggestion_id}")
        content = str(patch.get("content", "")).strip()
        body = append_section(target_note["body"], "智能补充", f"- {content}")
        database.update_note(target_note_id, target_note["title"], target_note["tags"], body)
        database.update_merge_suggestion_status(suggestion_id, "applied")
        scan_note(target_note_id)
    elif operation == "create_conflict_review":
        database.save_note_version(target_note_id, reason=f"apply conflict suggestion {suggestion_id}")
        body = append_section(
            target_note["body"],
            "待确认冲突",
            "\n".join(
                [
                    f"- 新内容：{patch.get('source_claim', '')}",
                    f"- 旧内容：{patch.get('target_claim', '')}",
                    f"- 原因：{patch.get('reason', '')}",
                ]
            ),
        )
        database.update_note(target_note_id, target_note["title"], target_note["tags"], body)
        database.update_merge_suggestion_status(suggestion_id, "applied")
        scan_note(target_note_id)
    else:
        database.update_merge_suggestion_status(suggestion_id, "applied")

    return database.get_merge_suggestion(suggestion_id)


def reject_suggestion(suggestion_id: str) -> dict[str, object] | None:
    return database.update_merge_suggestion_status(suggestion_id, "rejected")


def append_section(body: str, heading: str, content: str) -> str:
    body = body.rstrip()
    section_title = f"## {heading}"

    if section_title in body:
        return f"{body}\n{content}\n"

    return f"{body}\n\n{section_title}\n\n{content}\n".lstrip()


def strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-+*]\s+(\[[ xX]\]\s+)?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_>\[\]()]|https?://\S+", " ", text)
    return text


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    tokens = tokenize(text)
    scores: dict[str, float] = {}

    for index, token in enumerate(tokens):
        if token in STOP_WORDS:
            continue

        scores[token] = scores.get(token, 0.0) + 1.0 + max(0.0, 0.4 - (index * 0.01))

    return [token for token, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def tokenize(text: str) -> list[str]:
    normalized = strip_markdown(text).lower()
    raw_tokens = re.findall(r"[a-z0-9_+#.-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    tokens: list[str] = []

    for token in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 4:
            tokens.extend(token[index : index + 4] for index in range(0, len(token) - 1, 2))
        else:
            tokens.append(token)

    return [token for token in tokens if token and token not in STOP_WORDS]


def text_vector(text: str) -> list[float]:
    vector = [0.0] * VECTOR_SIZE

    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % VECTOR_SIZE
        vector[index] += 1.0

    length = math.sqrt(sum(value * value for value in vector))

    if not length:
        return vector

    return [round(value / length, 6) for value in vector]


def jaccard(left: set[object], right: set[object]) -> float:
    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def cosine(left: object, right: object) -> float:
    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        return 0.0

    return sum(float(a) * float(b) for a, b in zip(left, right))


def shared_entity(left: str, right: str) -> bool:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    return bool(left_tokens & right_tokens) or bool(left and right and (left in right or right in left))


def has_conflict_signal(left: str, right: str) -> bool:
    joined = f"{left} {right}"

    for first, second in CONFLICT_PAIRS:
        if first in joined and second in joined:
            return True

    negation_count = sum(1 for word in ("不是", "不能", "不会", "没有", "不需要") if word in joined)
    return negation_count > 0 and claim_similarity_for_text(left, right) > 0.42


def claim_similarity_for_text(left: str, right: str) -> float:
    return jaccard(set(tokenize(left)), set(tokenize(right)))


def normalize_text(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())
