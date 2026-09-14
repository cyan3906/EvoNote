import hashlib
import json
import math
import re
from typing import Any
from uuid import uuid4

from openai import OpenAI

from app.core import database
from app.core.config import settings
from app.core.retrieval import embed_text, hybrid_retrieve_notes, sync_note_indexes
from app.core.retry import retry_call

VECTOR_SIZE = 64
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

ANALYSIS_SYSTEM_PROMPT = """
你是 Evonote 的知识结构化智能体。你的任务是从用户的 Markdown 笔记中生成 L2 摘要、L3 检索表达、关键词和 Claims。

要求：
- 只根据输入原文生成，不要补充原文没有的事实。
- L2 summary 用中文，概括核心内容，不能超过 120 字。
- L3 text 是用于语义检索的精练表达，包含主题、实体、概念、技术名词和问题场景。
- keywords 输出 4-12 个关键词，优先保留专有名词、英文缩写、版本号、技术概念。
- claims 是可比较的知识点，不要抽取寒暄、标题本身、纯格式说明。
- 每个 claim 必须能在 source_text 中找到证据。
- 不确定的 claim 不要输出。
- 严格输出 JSON，不要 Markdown，不要解释。

JSON 格式：
{
  "note": {
    "l2_summary": "整篇笔记摘要",
    "l3_text": "整篇笔记检索表达",
    "keywords": ["关键词"]
  },
  "blocks": [
    {
      "block_index": 0,
      "l2_summary": "该 block 摘要",
      "l3_text": "该 block 检索表达",
      "keywords": ["关键词"],
      "claims": [
        {
          "claim_text": "完整知识点",
          "subject": "主体",
          "predicate": "关系/动作",
          "object_text": "客体/结论",
          "source_text": "原文证据",
          "keywords": ["关键词"],
          "confidence": 0.0
        }
      ]
    }
  ]
}
""".strip()


class EvolutionModelError(RuntimeError):
    pass


def scan_note(note_id: str) -> dict[str, object] | None:
    note = database.get_note(note_id)

    if not note:
        return None

    candidate_notes = {item["id"]: item for item in database.list_notes() if item["id"] != note_id}
    analysis = analyze_note(note)
    database.replace_note_analysis(
        note_id=note_id,
        representation=analysis["representation"],
        blocks=analysis["blocks"],
        claims=analysis["claims"],
    )
    sync_note_indexes(note, analysis["representation"], analysis["claims"])
    database.supersede_pending_suggestions(note_id)
    candidate_claims = retrieve_candidate_claims(
        note_id=note_id,
        representation=analysis["representation"],
    )

    if not candidate_claims:
        candidate_claims = database.list_claims(exclude_note_id=note_id)

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

    if representation is None or is_representation_stale(note, representation):
        return scan_note(note_id)

    return {
        "note": note,
        "representation": representation,
        "blocks": database.list_note_blocks(note_id),
        "claims": database.list_claims(note_id=note_id),
        "suggestions": database.list_merge_suggestions(source_note_id=note_id, status="pending"),
    }




def get_note_evolution_snapshot(note_id: str) -> dict[str, object] | None:
    note = database.get_note(note_id)

    if not note:
        return None

    representation = database.get_note_representation(note_id)
    stale = representation is None or is_representation_stale(note, representation)

    return {
        "note": note,
        "representation": representation,
        "blocks": database.list_note_blocks(note_id) if representation else [],
        "claims": database.list_claims(note_id=note_id) if representation else [],
        "suggestions": database.list_merge_suggestions(source_note_id=note_id, status="pending") if representation else [],
        "stale": stale,
    }
def is_representation_stale(note: dict[str, str], representation: dict[str, object]) -> bool:
    return str(note.get("updated_at", "")) > str(representation.get("updated_at", ""))


def analyze_note(note: dict[str, str]) -> dict[str, object]:
    raw_blocks = split_note_blocks(note)
    model_analysis = analyze_note_with_model(note, raw_blocks)
    blocks: list[dict[str, object]] = []
    claims: list[dict[str, object]] = []
    model_blocks = normalize_model_blocks(model_analysis.get("blocks", []), expected_count=len(raw_blocks))

    for block_index, raw_block in enumerate(raw_blocks):
        block_id = str(uuid4())
        l1_text = raw_block["text"].strip()
        model_block = model_blocks[block_index]
        keywords = clean_keywords(model_block.get("keywords", []), limit=12)
        l3_text = clean_text(model_block.get("l3_text", "")) or " ".join(keywords)
        block = {
            "id": block_id,
            "note_id": note["id"],
            "block_index": block_index,
            "heading": raw_block["heading"],
            "l1_text": l1_text,
            "l2_summary": clean_text(model_block.get("l2_summary", "")),
            "l3_text": l3_text,
            "keywords": keywords,
        }
        blocks.append(block)

        for claim_index, claim in enumerate(normalize_model_claims(model_block.get("claims", []), l1_text)):
            claim["id"] = str(uuid4())
            claim["note_id"] = note["id"]
            claim["block_id"] = block_id
            claim["claim_index"] = claim_index
            claim["vector"] = embedding_or_local_vector(" ".join(claim["keywords"] + [claim["claim_text"]]))
            claims.append(claim)

    model_note = model_analysis.get("note", {})
    note_keywords = clean_keywords(model_note.get("keywords", []), limit=14)
    note_l2_summary = clean_text(model_note.get("l2_summary", ""))
    note_l3_text = clean_text(model_note.get("l3_text", "")) or " ".join(note_keywords)
    representation = {
        "note_id": note["id"],
        "l2_summary": note_l2_summary,
        "l3_text": note_l3_text,
        "keywords": note_keywords,
        "vector": embedding_or_local_vector(
            note_l2_summary or f"{note_l3_text} {' '.join(note_keywords)}"
        ),
    }

    return {
        "representation": representation,
        "blocks": blocks,
        "claims": claims,
    }


def analyze_note_with_model(note: dict[str, str], raw_blocks: list[dict[str, str]]) -> dict[str, Any]:
    api_key = settings.evolution_api_key.strip() or settings.agent_api_key.strip()
    api_base_url = settings.evolution_api_base_url.strip() or settings.agent_api_base_url.strip()
    model = settings.evolution_model.strip() or settings.agent_model.strip()
    timeout = settings.evolution_timeout_seconds or settings.agent_timeout_seconds

    if not api_key or api_key == "change-me" or api_key == "your-agent-api-key":
        raise EvolutionModelError("未配置 Evolution 大模型 API Key，无法生成 L2/L3/Claims")

    if not model:
        raise EvolutionModelError("未配置 Evolution 大模型模型名")

    payload = {
        "note": {
            "id": note["id"],
            "title": note["title"],
            "tags": note["tags"],
        },
        "blocks": [
            {
                "block_index": index,
                "heading": block["heading"],
                "l1_text": block["text"],
            }
            for index, block in enumerate(raw_blocks)
        ],
    }

    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url or None,
        timeout=timeout,
    )

    try:
        response = retry_call(
            lambda: client.chat.completions.create(
                model=model,
                temperature=settings.evolution_temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=2),
                    },
                ],
            ),
            operation_name="Evolution model request",
        )
    except Exception as exc:
        raise EvolutionModelError(f"调用 Evolution 大模型失败：{exc}") from exc

    content = response.choices[0].message.content or "{}"

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EvolutionModelError("Evolution 大模型没有返回合法 JSON") from exc

    validate_model_analysis(data, expected_blocks=len(raw_blocks))
    return data


def retrieve_candidate_claims(
    note_id: str,
    representation: dict[str, object],
) -> list[dict[str, object]]:
    note_hits = hybrid_retrieve_notes(
        representation,
        exclude_note_id=note_id,
        top_k=settings.retrieval_top_k,
    )

    if not note_hits:
        return []

    candidate_claims: list[dict[str, object]] = []

    for hit in note_hits:
        candidate_claims.extend(database.list_claims(note_id=hit.note_id))

    return candidate_claims


def validate_model_analysis(data: dict[str, Any], expected_blocks: int) -> None:
    note = data.get("note")
    blocks = data.get("blocks")

    if not isinstance(note, dict):
        raise EvolutionModelError("Evolution 大模型返回缺少 note 对象")

    if not clean_text(note.get("l2_summary", "")):
        raise EvolutionModelError("Evolution 大模型返回缺少 note.l2_summary")

    if not isinstance(blocks, list) or len(blocks) != expected_blocks:
        raise EvolutionModelError("Evolution 大模型返回的 blocks 数量与原文分块不一致")

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise EvolutionModelError(f"Evolution 大模型返回的 block {index} 不是对象")

        if int(block.get("block_index", index)) != index:
            raise EvolutionModelError(f"Evolution 大模型返回的 block_index 不连续：{index}")

        if not clean_text(block.get("l2_summary", "")):
            raise EvolutionModelError(f"Evolution 大模型返回缺少 block {index} 的 l2_summary")

        claims = block.get("claims", [])

        if claims is None:
            block["claims"] = []
            continue

        if not isinstance(claims, list):
            raise EvolutionModelError(f"Evolution 大模型返回的 block {index} claims 不是数组")


def normalize_model_blocks(blocks: object, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(blocks, list) or len(blocks) != expected_count:
        raise EvolutionModelError("Evolution 大模型返回的 blocks 不可用")

    normalized: list[dict[str, Any]] = []

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise EvolutionModelError(f"Evolution 大模型返回的 block {index} 不可用")

        normalized.append(block)

    return normalized


def normalize_model_claims(claims: object, block_text: str) -> list[dict[str, object]]:
    if not isinstance(claims, list):
        return []

    normalized: list[dict[str, object]] = []
    seen: set[str] = set()

    for item in claims[:24]:
        if not isinstance(item, dict):
            continue

        claim_text = clean_text(item.get("claim_text", ""))
        source_text = clean_text(item.get("source_text", "")) or claim_text

        if len(claim_text) < 4:
            continue

        fingerprint = normalize_text(claim_text)

        if not fingerprint or fingerprint in seen:
            continue

        seen.add(fingerprint)
        keywords = clean_keywords(item.get("keywords", []), limit=8)
        normalized.append(
            {
                "claim_text": claim_text,
                "subject": clean_text(item.get("subject", ""))[:48],
                "predicate": clean_text(item.get("predicate", ""))[:40] or "related_to",
                "object_text": clean_text(item.get("object_text", ""))[:160],
                "source_text": source_text[:220],
                "keywords": keywords,
                "confidence": normalize_confidence(item.get("confidence", 0.7)),
            }
        )

    return normalized


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_keywords(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    keywords: list[str] = []
    seen: set[str] = set()

    for item in value:
        keyword = clean_text(item).strip("，,;；")

        if not keyword or keyword in seen:
            continue

        seen.add(keyword)
        keywords.append(keyword[:40])

        if len(keywords) >= limit:
            break

    return keywords


def normalize_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.7

    return max(0.0, min(1.0, confidence))


def embedding_or_local_vector(text: str) -> list[float]:
    try:
        vector = embed_text(text)
    except Exception:
        vector = []

    return vector or text_vector(text)


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
        for target_claim in candidate_claims:
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

