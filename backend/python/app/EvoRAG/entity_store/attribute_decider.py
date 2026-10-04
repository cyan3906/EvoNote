from typing import Any

from app.EvoRAG.entity_store.models import (
    AttributeCandidate,
    AttributeDecision,
    AttributeRetrievalResult,
    EntityAttributeInput,
)

NEGATION_MARKERS = ("不会", "不能", "不是", "无需", "不需要", "没有", "无", "not", "never", "no ")
DETAIL_MARKERS = ("如果", "因为", "导致", "从而", "通过", "需要", "当", "：", ":", "，")
UPDATE_MARKERS = ("改为", "替代", "更新为", "应为", "应该是", "修正为", "不再")
MAX_ATTRIBUTE_DECISION_CANDIDATES = 5
ATTRIBUTE_DECISION_SYSTEM_PROMPT = """
你是 EvoRAG 的属性合并裁判。你需要判断 incoming attribute 是否应该并入某个已有 attribute。

动作定义：
- add: incoming 是新知识点，应新增属性。
- merge: incoming 与某个候选表达同一事实，只追加 evidence，不改 value_text。
- enrich: incoming 是某个候选同一事实的补充细节，可以扩写 value_text。
- update: incoming 明显替代或修正某个候选的旧值。
- conflict: incoming 与某个候选冲突，不能覆盖，记录冲突。
- review: 信息不足或风险高，需要人工审核。

严格规则：
- attr_type 相同不代表可以 merge。
- 语义相关不代表同一事实。
- 如果不是同一事实，必须 add。
- 如果不确定，必须 review。
- 只能从候选列表中选择 target_attribute_id。
- 历史摘要只用于风险判断，不要编造历史或候选。

只输出 JSON。
""".strip()


class AttributeDecisionMaker:
    def decide(
        self,
        incoming_attributes: list[EntityAttributeInput],
        retrieval_results: list[AttributeRetrievalResult],
    ) -> list[AttributeDecision]:
        results_by_index = {result.input_index: result for result in retrieval_results}
        decisions: list[AttributeDecision] = []
        for input_index, attribute in enumerate(incoming_attributes):
            result = results_by_index.get(input_index)
            if result is None or not result.candidates and result.exact_match is None:
                decisions.append(
                    AttributeDecision(
                        input_index=input_index,
                        action="add",
                        incoming_attribute=attribute,
                        confidence=attribute.confidence,
                        reason="no attribute candidate found",
                    )
                )
                continue

            if result.exact_match is not None:
                decisions.append(
                    AttributeDecision(
                        input_index=input_index,
                        action="merge",
                        incoming_attribute=attribute,
                        target_attribute_id=result.exact_match.id,
                        target_value_text=result.exact_match.value_text,
                        confidence=1.0,
                        reason="exact fingerprint match",
                    )
                )
                continue

            best = result.candidates[0]
            action = decide_candidate_action(attribute.value_text, best)
            new_value = ""
            if action == "enrich":
                new_value = enriched_value(best.value_text, attribute.value_text)
            elif action == "update":
                new_value = attribute.value_text
            if action == "add":
                decisions.append(
                    AttributeDecision(
                        input_index=input_index,
                        action="add",
                        incoming_attribute=attribute,
                        confidence=attribute.confidence,
                        reason="candidate is related but not proven to be the same fact",
                    )
                )
                continue
            decisions.append(
                AttributeDecision(
                    input_index=input_index,
                    action=action,
                    incoming_attribute=attribute,
                    target_attribute_id=best.attribute_id,
                    target_value_text=best.value_text,
                    new_value_text=new_value,
                    confidence=max(attribute.confidence, best.confidence),
                    reason=f"{action} against top attribute candidate",
                )
            )
        return decisions


def decide_candidate_action(incoming_value: str, candidate: AttributeCandidate) -> str:
    incoming = normalize_text(incoming_value)
    existing = normalize_text(candidate.value_text)
    if is_conflict(incoming, existing):
        return "conflict"
    if is_update(incoming, existing):
        return "update"
    if is_enrichment(incoming, existing):
        return "enrich"
    if is_same_fact(incoming, existing):
        return "merge"
    return "add"


def is_conflict(incoming: str, existing: str) -> bool:
    if not incoming or not existing:
        return False
    if shared_prefix_score(incoming, existing) < 0.35:
        return False
    return contains_negation(incoming) != contains_negation(existing)


def is_enrichment(incoming: str, existing: str) -> bool:
    if not incoming or not existing:
        return False
    if len(incoming) <= len(existing) * 1.35:
        return False
    if shared_prefix_score(incoming, existing) < 0.15 and not any(marker in incoming for marker in DETAIL_MARKERS):
        return False
    return True


def is_update(incoming: str, existing: str) -> bool:
    if not incoming or not existing:
        return False
    if not any(marker in incoming for marker in UPDATE_MARKERS):
        return False
    return shared_prefix_score(incoming, existing) >= 0.35


def is_same_fact(incoming: str, existing: str) -> bool:
    if not incoming or not existing:
        return False
    length_ratio = max(len(incoming), len(existing)) / max(1, min(len(incoming), len(existing)))
    return length_ratio <= 1.35 and shared_prefix_score(incoming, existing) >= 0.85


def enriched_value(existing: str, incoming: str) -> str:
    existing = str(existing or "").strip()
    incoming = str(incoming or "").strip()
    if not existing:
        return incoming
    if not incoming or incoming == existing:
        return existing
    if existing in incoming:
        return incoming
    if incoming in existing:
        return existing
    return f"{existing}：{incoming}"


def contains_negation(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in NEGATION_MARKERS)


def shared_prefix_score(left: str, right: str) -> float:
    left_terms = token_set(left)
    right_terms = token_set(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


def token_set(value: str) -> set[str]:
    normalized = normalize_text(value)
    words = {part for part in normalized.replace("，", " ").replace("。", " ").replace(",", " ").split() if len(part) > 1}
    if words:
        return words
    return {normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))}


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def build_attribute_decision_payload(
    incoming_attribute: EntityAttributeInput,
    retrieval_result: AttributeRetrievalResult,
    *,
    history_summaries: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    history_summaries = history_summaries if history_summaries is not None else retrieval_result.history_summaries
    candidates: list[dict[str, Any]] = []
    for candidate in retrieval_result.candidates[:MAX_ATTRIBUTE_DECISION_CANDIDATES]:
        candidates.append(
            {
                "attribute_id": candidate.attribute_id,
                "attr_type": candidate.attr_type,
                "value_text": candidate.value_text,
                "confidence": candidate.confidence,
                "retrieval": {
                    "rank": candidate.rank,
                    "source": candidate.source,
                    "es_score": candidate.es_score,
                    "vector_score": candidate.vector_score,
                    "fused_score": candidate.fused_score,
                    "exact_match": retrieval_result.exact_match is not None and retrieval_result.exact_match.id == candidate.attribute_id,
                },
                "history_summary": strip_evidence(history_summaries.get(candidate.attribute_id, {})),
            }
        )
    return {
        "incoming": {
            "attr_type": incoming_attribute.attr_type,
            "value_text": incoming_attribute.value_text,
            "note_id": incoming_attribute.note_id,
            "block_id": incoming_attribute.block_id,
            "block_index": incoming_attribute.block_index,
        },
        "retrieval": {
            "mode": retrieval_result.mode,
            "group_size": retrieval_result.group_size,
        },
        "candidates": candidates,
        "allowed_actions": ["add", "merge", "enrich", "update", "conflict", "review"],
    }


def strip_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_evidence(item) for key, item in value.items() if "evidence" not in str(key).lower()}
    if isinstance(value, list):
        return [strip_evidence(item) for item in value]
    return value
