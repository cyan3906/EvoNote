import asyncio
import json

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import CandidateEntity, EntityResolutionDecision, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name
from app.EvoRAG.indexes import EntityHybridIndex
from app.EvoRAG.llm import EvoRAGLLMClient


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


class EntityResolver:
    def __init__(
        self,
        existing_entities: list[StoredEntity],
        config: EvoRAGSettings = settings,
        llm_client: EvoRAGLLMClient | None = None,
        hybrid_index: EntityHybridIndex | None = None,
    ) -> None:
        self.config = config
        self.existing_entities = existing_entities
        self.existing_by_id = {entity.id: entity for entity in existing_entities}
        self.hybrid_index = hybrid_index or EntityHybridIndex(config)
        self.llm = llm_client or EvoRAGLLMClient(config)

    async def resolve_many(self, incoming_entities: list[IncomingEntity]) -> list[EntityResolutionDecision]:
        return await asyncio.gather(*(self.resolve_one(incoming) for incoming in incoming_entities))

    async def resolve_one(self, incoming: IncomingEntity) -> EntityResolutionDecision:
        exact = self._exact_name_match(incoming)
        if exact is not None:
            return EntityResolutionDecision(
                incoming=incoming,
                decision="matched",
                matched_entity=exact,
                score=1.0,
                reason="normalized name and entity type matched",
            )

        candidates = self.hybrid_index.search(incoming, top_k=self.config.entity_resolution_top_k)
        candidates = self._hydrate_candidates(candidates)
        if not candidates:
            return EntityResolutionDecision(incoming=incoming, decision="new", reason="no candidates")

        best = candidates[0]
        if best.score >= self.config.entity_resolution_auto_match_threshold and descriptions_compatible(incoming, best.entity):
            return EntityResolutionDecision(
                incoming=incoming,
                decision="matched",
                matched_entity=best.entity,
                score=best.score,
                reason="high vector score and compatible descriptions",
            )

        if best.score >= self.config.entity_resolution_llm_threshold:
            return await self._judge_with_llm(incoming, candidates[: self.config.entity_resolution_top_k])

        if best.score < self.config.entity_resolution_manual_threshold:
            return EntityResolutionDecision(
                incoming=incoming,
                decision="ambiguous",
                score=best.score,
                reason="candidate score below manual threshold",
            )

        return EntityResolutionDecision(
            incoming=incoming,
            decision="new",
            score=best.score,
            reason="candidate score between manual and llm thresholds",
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
        data = await self.llm.chat_json(
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
            return EntityResolutionDecision(incoming=incoming, decision="matched", matched_entity=matched_entity, score=score, reason=reason)
        if decision == "new":
            return EntityResolutionDecision(incoming=incoming, decision="new", score=score, reason=reason)
        return EntityResolutionDecision(incoming=incoming, decision="ambiguous", score=score, reason=reason or json.dumps(data, ensure_ascii=False))
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
