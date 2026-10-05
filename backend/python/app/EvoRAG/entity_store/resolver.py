import asyncio
import json
from collections.abc import Callable
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import CandidateEntity, EntityRelationMemoryRecord, EntityResolutionDecision, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name
from app.EvoRAG.entity_store.relation_memory import is_allow_relation, is_reject_relation
from app.EvoRAG.indexes import EntityHybridIndex
from app.EvoRAG.indexes.entity_hybrid import reciprocal_rank_fusion_entities
from app.EvoRAG.llm import EvoRAGLLMClient


NON_MERGE_RELATION_DECISIONS = {
    "attribute_of_entity",
    "section_title_of_entity",
    "process_step_of_entity",
    "metric_or_property",
}
ADMISSION_PASS_DECISIONS = {"entity", "valid_entity", "candidate_entity"}
ADMISSION_REJECT_DECISIONS = {"attribute", "section_title", "process_step", "metric_or_property", "not_entity"}

ENTITY_RESOLUTION_SYSTEM_PROMPT = """
你是 EvoRAG 的实体消歧裁判。你需要判断 incoming entity 是否和候选实体是同一个知识实体。

判断依据：
- 名称、别名、实体类型
- definition / purpose / mechanism / related 等简述
- 不要引入外部知识；只根据输入判断

输出要求：
- 如果能确定和某个候选是同一实体，decision=matched，给 matched_entity_id
- 如果候选都不是同一实体，decision=new
- 如果无法确定，decision=ambiguous
- 严格输出 JSON，不要 Markdown，不要解释

JSON 格式：
{
  "decision": "matched | new | ambiguous",
  "matched_entity_id": 0,
  "confidence": 0.0,
  "reason": "一句话理由"
}
""".strip()
ENTITY_ADMISSION_GUARD_SYSTEM_PROMPT = """
你是 EvoRAG 的实体准入守门裁判。你会收到一个 incoming entity。目标是在进入实体关系经验和混合检索前，判断它本身是否值得作为实体消歧对象。

规则：
- 如果 incoming 表示独立知识实体、概念、方法、系统、算法或数据结构，decision=entity。
- 如果 incoming 只是属性、指标、章节标题、过程步骤，或明显不是实体，不能进入实体合并；输出 attribute、metric_or_property、section_title、process_step 或 not_entity。
- 如果信息不足但仍可能是实体，decision=entity，不要因为缺少历史关系而拒绝。

只输出 JSON，不要 Markdown，不要解释。

JSON 格式：
{
  "decision": "entity | attribute | section_title | process_step | metric_or_property | not_entity",
  "confidence": 0.0,
  "reason": "一句话理由"
}
""".strip()
ENTITY_RELATION_MEMORY_JUDGE_SYSTEM_PROMPT = """
你是 EvoRAG 的实体关系记忆裁判。你会收到一个 incoming entity，以及历史维护的白名单和黑名单关系。

目标：只根据这些历史关系判断 incoming entity 是否能直接等价到某个候选实体。

规则：
- 白名单表示过去确认过的同一实体或等价表述。
- 黑名单表示过去确认过的非同一实体、属性误导、版本差异或相关但不同。
- 黑名单优先级高于名称相似度；如果白名单和黑名单冲突，输出 ambiguous。
- 如果能确定等价，decision=matched，并填写 matched_entity_id。
- 如果 incoming 只是某个候选实体的属性、指标、章节标题或过程步骤，不能合并；输出 attribute_of_entity、metric_or_property、section_title_of_entity 或 process_step_of_entity，并填写对应 matched_entity_id。
- 如果只是相关但不是同一实体，decision=not_matched。
- 如果不能确定，decision=ambiguous，不要猜测。
- 不要引入候选列表之外的实体。

只输出 JSON，不要 Markdown，不要解释。

JSON 格式：
{
  "decision": "matched | not_matched | ambiguous | attribute_of_entity | section_title_of_entity | process_step_of_entity | metric_or_property",
  "matched_entity_id": 0,
  "confidence": 0.0,
  "reason": "一句话理由"
}
""".strip()


class EntityResolver:
    def __init__(
        self,
        existing_entities: list[StoredEntity],
        config: EvoRAGSettings = settings,
        llm_client: EvoRAGLLMClient | None = None,
        hybrid_index: EntityHybridIndex | None = None,
        alias_lookup: Callable[[IncomingEntity], StoredEntity | None] | None = None,
        rejection_lookup: Callable[[IncomingEntity], set[int]] | None = None,
        relation_memory: Any | None = None,
    ) -> None:
        self.config = config
        self.existing_entities = existing_entities
        self.existing_by_id = {entity.id: entity for entity in existing_entities}
        self.hybrid_index = hybrid_index or EntityHybridIndex(config)
        self.llm = llm_client
        self.alias_lookup = alias_lookup
        self.rejection_lookup = rejection_lookup
        self.relation_memory = relation_memory

    async def resolve_many(self, incoming_entities: list[IncomingEntity]) -> list[EntityResolutionDecision]:
        return await asyncio.gather(*(self.resolve_one(incoming) for incoming in incoming_entities))

    async def resolve_one(self, incoming: IncomingEntity) -> EntityResolutionDecision:
        trace: list[dict[str, Any]] = []

        def finish(decision: EntityResolutionDecision, exit_stage: str) -> EntityResolutionDecision:
            decision.resolution_trace = trace
            decision.resolution_exit_stage = exit_stage
            return decision

        direct_relation = self._direct_relation(incoming)
        if direct_relation is None:
            trace.append({"stage": "redis_direct_relation", "status": "miss", "relation": None})
        else:
            direct_status = "allow" if is_allow_relation(direct_relation.decision) else "non_allow"
            trace.append(
                {
                    "stage": "redis_direct_relation",
                    "status": direct_status,
                    "relation": relation_memory_snapshot(direct_relation),
                    "decision": direct_relation.decision,
                    "reason": direct_relation.reason,
                }
            )
            if is_allow_relation(direct_relation.decision):
                return finish(
                    self._relation_memory_match_decision(incoming, direct_relation, reason="relation memory direct allow"),
                    "redis_direct_relation",
                )

        admission = await self._judge_admission_guard(incoming)
        admission_rejected = admission["decision"] in ADMISSION_REJECT_DECISIONS
        trace.append(
            {
                "stage": "deepseek_admission_guard",
                "status": "ended" if admission_rejected else "continued",
                "judge": admission,
                "output_decision": "new" if admission_rejected else None,
            }
        )
        if admission_rejected:
            return finish(
                EntityResolutionDecision(
                    incoming=incoming,
                    decision="new",
                    score=float(admission.get("confidence") or 0.0),
                    reason=str(admission.get("reason") or f"entity admission guard rejected as {admission['decision']}"),
                    candidates=[],
                ),
                "deepseek_admission_guard",
            )

        relations = self._top_relations(incoming)
        trace.append(
            {
                "stage": "mysql_relation_memory_top30",
                "status": "found" if relations else "no_relation",
                "top_k": 30,
                "relations": [relation_memory_snapshot(relation) for relation in relations],
            }
        )
        if relations:
            memory_judgment = await self._judge_relation_memory(incoming, relations)
            matched_record = find_relation_record(relations, memory_judgment.get("matched_entity_id"))
            relation_guard_status = "fallback_hybrid"
            if memory_judgment["decision"] == "matched" and matched_record is not None:
                relation_guard_status = "matched"
            elif memory_judgment["decision"] in NON_MERGE_RELATION_DECISIONS and matched_record is not None:
                relation_guard_status = "rejected_to_new"
            trace.append(
                {
                    "stage": "deepseek_relation_guard",
                    "status": relation_guard_status,
                    "judge": memory_judgment,
                    "matched_relation": relation_memory_snapshot(matched_record) if matched_record else None,
                    "experience_written": (
                        {
                            "decision": "reject",
                            "relation_type": str(memory_judgment["decision"]),
                            "source": "llm_guard",
                            "confidence": float(memory_judgment.get("confidence") or matched_record.confidence),
                        }
                        if relation_guard_status == "rejected_to_new" and matched_record is not None
                        else None
                    ),
                }
            )
            if memory_judgment["decision"] in NON_MERGE_RELATION_DECISIONS and matched_record is not None:
                score = float(memory_judgment.get("confidence") or matched_record.confidence)
                reason = str(memory_judgment.get("reason") or matched_record.reason)
                return finish(
                    EntityResolutionDecision(
                        incoming=incoming,
                        decision="new",
                        score=score,
                        reason=reason,
                        candidates=[self._relation_candidate(matched_record, score)],
                        record_experience=True,
                        experience_decision="reject",
                        experience_relation_type=str(memory_judgment["decision"]),
                        experience_source="llm_guard",
                        experience_confidence=score,
                    ),
                    "deepseek_relation_guard",
                )
            if memory_judgment["decision"] == "matched" and matched_record is not None:
                return finish(
                    self._relation_memory_match_decision(
                        incoming,
                        matched_record,
                        reason=str(memory_judgment.get("reason") or matched_record.reason),
                        score=float(memory_judgment.get("confidence") or matched_record.confidence),
                    ),
                    "deepseek_relation_guard",
                )
        else:
            trace.append({"stage": "deepseek_relation_guard", "status": "skipped", "judge": None})

        alias_match = self.alias_lookup(incoming) if self.alias_lookup else None
        if alias_match is not None:
            trace.append({"stage": "alias_lookup", "status": "matched", "matched_entity_id": alias_match.id})
            return finish(
                EntityResolutionDecision(
                    incoming=incoming,
                    decision="matched",
                    matched_entity=alias_match,
                    score=1.0,
                    reason="alias map matched",
                    candidates=[],
                ),
                "alias_lookup",
            )
        trace.append({"stage": "alias_lookup", "status": "miss"})

        exact = self._exact_name_match(incoming)
        if exact is not None:
            trace.append({"stage": "exact_name_match", "status": "matched", "matched_entity_id": exact.id})
            return finish(
                EntityResolutionDecision(
                    incoming=incoming,
                    decision="matched",
                    matched_entity=exact,
                    score=1.0,
                    reason="normalized name and entity type matched",
                    candidates=[],
                ),
                "exact_name_match",
            )
        trace.append({"stage": "exact_name_match", "status": "miss"})

        es_hits, milvus_hits, candidates = self._hybrid_search_components(incoming)
        candidates = self._hydrate_candidates(candidates)
        rejected_ids = self.rejection_lookup(incoming) if self.rejection_lookup else set()
        if rejected_ids:
            candidates = [candidate for candidate in candidates if candidate.entity.id not in rejected_ids]
        trace.append(
            {
                "stage": "hybrid_rrf_baseline",
                "status": "used",
                "incoming_embedding": {
                    "generated": bool(incoming.embedding),
                    "dimension": len(incoming.embedding),
                },
                "milvus": {"results": [candidate_trace_payload(candidate) for candidate in milvus_hits]},
                "elasticsearch": {"results": [candidate_trace_payload(candidate) for candidate in es_hits]},
                "rrf": {"results": [candidate_trace_payload(candidate) for candidate in candidates]},
                "rejected_candidate_ids": sorted(rejected_ids),
            }
        )
        if not candidates:
            return finish(
                EntityResolutionDecision(incoming=incoming, decision="new", reason="no candidates", candidates=[]),
                "hybrid_rrf_baseline",
            )

        try:
            llm_decision = await self._judge_with_llm(incoming, candidates)
            trace.append(
                {
                    "stage": "deepseek_final_judge",
                    "status": llm_decision.decision,
                    "judge": {
                        "decision": llm_decision.decision,
                        "matched_entity_id": llm_decision.matched_entity.id if llm_decision.matched_entity else None,
                        "confidence": llm_decision.score,
                        "reason": llm_decision.reason,
                    },
                    "experience_written": experience_payload(llm_decision),
                }
            )
            return finish(llm_decision, "deepseek_final_judge")
        except Exception as exc:
            trace.append(
                {
                    "stage": "deepseek_final_judge",
                    "status": "failed",
                    "error": str(exc),
                }
            )

        best = candidates[0]
        if best.score >= self.config.entity_resolution_auto_match_threshold and descriptions_compatible(incoming, best.entity):
            return finish(
                EntityResolutionDecision(
                    incoming=incoming,
                    decision="matched",
                    matched_entity=best.entity,
                    score=best.score,
                    reason="high vector score and compatible descriptions",
                    candidates=candidates,
                ),
                "hybrid_rrf_baseline",
            )

        if best.score >= self.config.entity_resolution_manual_threshold:
            return finish(
                EntityResolutionDecision(
                    incoming=incoming,
                    decision="ambiguous",
                    score=best.score,
                    reason="candidate score below auto-match threshold; manual review required",
                    candidates=candidates,
                ),
                "hybrid_rrf_baseline",
            )

        return finish(
            EntityResolutionDecision(
                incoming=incoming,
                decision="new",
                score=best.score,
                reason="candidate score below manual threshold; create new entity",
                candidates=candidates,
            ),
            "hybrid_rrf_baseline",
        )

    def _direct_relation(self, incoming: IncomingEntity) -> EntityRelationMemoryRecord | None:
        if self.relation_memory is None:
            return None
        reader = getattr(self.relation_memory, "get_direct_relation", None)
        if reader is None:
            return None
        try:
            return reader(incoming)
        except Exception:
            return None

    def _top_relations(self, incoming: IncomingEntity) -> list[EntityRelationMemoryRecord]:
        if self.relation_memory is None:
            return []
        reader = getattr(self.relation_memory, "list_top_relations", None)
        if reader is None:
            return []
        try:
            return list(reader(incoming, limit=30))
        except Exception:
            return []

    def _relation_memory_match_decision(
        self,
        incoming: IncomingEntity,
        relation: EntityRelationMemoryRecord,
        *,
        reason: str,
        score: float | None = None,
    ) -> EntityResolutionDecision:
        confidence = float(relation.confidence if score is None else score)
        return EntityResolutionDecision(
            incoming=incoming,
            decision="matched",
            matched_entity=relation.candidate,
            score=confidence,
            reason=reason,
            candidates=[self._relation_candidate(relation, confidence)],
        )

    def _relation_candidate(self, relation: EntityRelationMemoryRecord, score: float) -> CandidateEntity:
        return CandidateEntity(
            entity=relation.candidate,
            score=float(score),
            rank=1,
            source="relation_memory",
            vector_score=0.0,
            es_score=0.0,
        )

    def _exact_name_match(self, incoming: IncomingEntity) -> StoredEntity | None:
        incoming_names = {incoming.normalized_name, *(normalize_name(alias) for alias in incoming.aliases)}
        for entity in self.existing_entities:
            existing_names = {entity.normalized_name, *(normalize_name(alias) for alias in entity.aliases)}
            if incoming.entity_type == entity.entity_type and incoming_names & existing_names:
                return entity
        return None

    def _hydrate_candidates(self, candidates: list[CandidateEntity]) -> list[CandidateEntity]:
        hydrated: list[CandidateEntity] = []
        for candidate in candidates:
            entity = self.existing_by_id.get(candidate.entity.id, candidate.entity)
            hydrated.append(
                CandidateEntity(
                    entity=entity,
                    score=candidate.score,
                    rank=candidate.rank,
                    source=candidate.source,
                    vector_score=candidate.vector_score,
                    es_score=candidate.es_score,
                )
            )
        return hydrated

    def _hybrid_search_components(self, incoming: IncomingEntity) -> tuple[list[CandidateEntity], list[CandidateEntity], list[CandidateEntity]]:
        limit = self.config.entity_resolution_top_k
        search_components = getattr(self.hybrid_index, "search_components", None)
        if callable(search_components):
            es_hits, milvus_hits = search_components(incoming, top_k=limit)
            candidates = reciprocal_rank_fusion_entities(
                [es_hits, milvus_hits],
                rrf_k=self.config.entity_resolution_rrf_k,
                top_k=limit,
            )
            return list(es_hits), list(milvus_hits), candidates
        candidates = self.hybrid_index.search(incoming, top_k=limit)
        return [], [], list(candidates)

    async def _judge_with_llm(self, incoming: IncomingEntity, candidates: list[CandidateEntity]) -> EntityResolutionDecision:
        payload = {
            "incoming_entity": {
                "name": incoming.name,
                "entity_type": incoming.entity_type,
                "aliases": incoming.aliases,
                "description": incoming.description_for_match,
            },
            "candidates": [
                {
                    "id": candidate.entity.id,
                    "canonical_name": candidate.entity.canonical_name,
                    "entity_type": candidate.entity.entity_type,
                    "aliases": candidate.entity.aliases,
                    "summary": candidate.entity.summary,
                    "identity_description": candidate.entity.identity_description,
                    "description": candidate.entity.description_for_match,
                    "score": candidate.score,
                    "source": candidate.source,
                    "vector_score": candidate.vector_score,
                    "es_score": candidate.es_score,
                }
                for candidate in candidates
            ],
        }
        llm = self._llm_client()
        data = await llm.chat_json(
            system_prompt=ENTITY_RESOLUTION_SYSTEM_PROMPT,
            user_payload=payload,
            operation_name=f"EvoRAG entity resolution {incoming.name}",
        )
        decision = str(data.get("decision") or "ambiguous")
        matched_id = int(data.get("matched_entity_id") or 0)
        score = float(data.get("confidence") or candidates[0].score)
        reason = str(data.get("reason") or "")
        matched_entity = next((candidate.entity for candidate in candidates if candidate.entity.id == matched_id), None)

        if decision == "matched" and matched_entity is not None:
            return EntityResolutionDecision(
                incoming=incoming,
                decision="matched",
                matched_entity=matched_entity,
                score=score,
                reason=reason,
                candidates=candidates,
                record_experience=score >= self.config.entity_resolution_llm_threshold,
                experience_decision="allow",
                experience_relation_type="llm_judge_match",
                experience_source="llm_judge",
                experience_confidence=score,
            )
        if decision == "new":
            best_score = candidates[0].score if candidates else 0.0
            record_experience = score >= self.config.entity_resolution_llm_threshold and best_score >= self.config.entity_resolution_auto_match_threshold
            return EntityResolutionDecision(
                incoming=incoming,
                decision="new",
                score=score,
                reason=reason,
                candidates=candidates,
                record_experience=record_experience,
                experience_decision="reject",
                experience_relation_type="high_score_llm_reject",
                experience_source="llm_judge",
                experience_confidence=score,
            )
        return EntityResolutionDecision(
            incoming=incoming,
            decision="ambiguous",
            score=score,
            reason=reason or json.dumps(data, ensure_ascii=False),
            candidates=candidates,
        )

    async def _try_final_llm_judge(self, incoming: IncomingEntity, candidates: list[CandidateEntity]) -> EntityResolutionDecision | None:
        try:
            return await self._judge_with_llm(incoming, candidates)
        except Exception:
            return None

    async def _judge_admission_guard(self, incoming: IncomingEntity) -> dict[str, Any]:
        if not self.config.entity_admission_judge_enabled:
            return {"decision": "entity", "confidence": 0.0, "reason": ""}
        try:
            data = await self._llm_client().chat_json(
                system_prompt=ENTITY_ADMISSION_GUARD_SYSTEM_PROMPT,
                user_payload={"incoming_entity": incoming_snapshot(incoming)},
                operation_name=f"EvoRAG entity admission guard {incoming.name}",
                model=self.config.entity_admission_judge_model,
            )
            return normalize_admission_guard_judgment(data)
        except Exception:
            return {"decision": "entity", "confidence": 0.0, "reason": ""}

    async def _judge_relation_memory(self, incoming: IncomingEntity, relations: list[EntityRelationMemoryRecord]) -> dict[str, Any]:
        allow_relations = [relation for relation in relations if is_allow_relation(relation.decision)]
        reject_relations = [relation for relation in relations if is_reject_relation(relation.decision)]
        try:
            data = await self._llm_client().chat_json(
                system_prompt=ENTITY_RELATION_MEMORY_JUDGE_SYSTEM_PROMPT,
                user_payload={
                    "incoming_entity": incoming_snapshot(incoming),
                    "allow_relations": [relation_memory_snapshot(relation) for relation in allow_relations],
                    "reject_relations": [relation_memory_snapshot(relation) for relation in reject_relations],
                },
                operation_name=f"EvoRAG entity relation memory judge {incoming.name}",
                model=self.config.entity_admission_judge_model,
            )
            return normalize_relation_memory_judgment(data)
        except Exception:
            return {"decision": "ambiguous", "matched_entity_id": None, "confidence": 0.0, "reason": ""}

    def _llm_client(self) -> EvoRAGLLMClient:
        if self.llm is None:
            self.llm = EvoRAGLLMClient(self.config)
        return self.llm


def normalize_admission_guard_judgment(data: dict[str, Any]) -> dict[str, Any]:
    decision = str(data.get("decision") or "entity").strip().lower()
    if decision not in {*ADMISSION_PASS_DECISIONS, *ADMISSION_REJECT_DECISIONS}:
        decision = "entity"
    return {
        "decision": decision,
        "confidence": float(data.get("confidence") or 0.0),
        "reason": str(data.get("reason") or ""),
    }


def normalize_relation_memory_judgment(data: dict[str, Any]) -> dict[str, Any]:
    decision = str(data.get("decision") or "ambiguous").strip().lower()
    if decision not in {"matched", "not_matched", "ambiguous", *NON_MERGE_RELATION_DECISIONS}:
        decision = "ambiguous"
    matched_entity_id = data.get("matched_entity_id")
    return {
        "decision": decision,
        "matched_entity_id": int(matched_entity_id) if matched_entity_id else None,
        "confidence": float(data.get("confidence") or 0.0),
        "reason": str(data.get("reason") or ""),
    }


def incoming_snapshot(incoming: IncomingEntity) -> dict[str, Any]:
    return {
        "name": incoming.name,
        "normalized_name": incoming.normalized_name,
        "entity_type": incoming.entity_type,
        "aliases": incoming.aliases,
        "identity_description": incoming.identity_description,
        "description_for_match": incoming.description_for_match,
        "source_count": incoming.source_count,
    }


def relation_memory_snapshot(relation: EntityRelationMemoryRecord) -> dict[str, Any]:
    entity = relation.candidate
    return {
        "relation_id": relation.id,
        "decision": relation.decision,
        "relation_type": relation.relation_type,
        "confidence": relation.confidence,
        "hit_count": relation.hit_count,
        "source": relation.source,
        "reason": relation.reason,
        "candidate_entity": {
            "id": entity.id,
            "canonical_name": entity.canonical_name,
            "normalized_name": entity.normalized_name,
            "entity_type": entity.entity_type,
            "aliases": entity.aliases,
            "identity_description": entity.identity_description,
            "summary": entity.summary,
            "description_for_match": entity.description_for_match,
        },
    }


def candidate_trace_payload(candidate: CandidateEntity) -> dict[str, Any]:
    return {
        "entity_id": candidate.entity.id,
        "canonical_name": candidate.entity.canonical_name,
        "score": candidate.score,
        "rank": candidate.rank,
        "source": candidate.source,
        "vector_score": candidate.vector_score,
        "es_score": candidate.es_score,
    }


def experience_payload(decision: EntityResolutionDecision) -> dict[str, Any] | None:
    if not decision.record_experience:
        return None
    return {
        "decision": decision.experience_decision,
        "relation_type": decision.experience_relation_type,
        "source": decision.experience_source,
        "confidence": decision.experience_confidence,
    }


def find_relation_record(
    relations: list[EntityRelationMemoryRecord],
    matched_entity_id: Any,
) -> EntityRelationMemoryRecord | None:
    if matched_entity_id is None:
        return None
    try:
        entity_id = int(matched_entity_id)
    except (TypeError, ValueError):
        return None
    return next((relation for relation in relations if relation.candidate.id == entity_id), None)


def descriptions_compatible(incoming: IncomingEntity, existing: StoredEntity) -> bool:
    if incoming.entity_type != existing.entity_type:
        return False
    incoming_text = normalize_name(incoming.identity_description or incoming.description_for_match)
    existing_text = normalize_name(existing.identity_description or existing.description_for_match or existing.summary)
    if not incoming_text or not existing_text:
        return True
    incoming_terms = set(incoming_text.split())
    existing_terms = set(existing_text.split())
    if not incoming_terms or not existing_terms:
        return True
    overlap = len(incoming_terms & existing_terms) / max(1, min(len(incoming_terms), len(existing_terms)))
    return overlap >= 0.2
