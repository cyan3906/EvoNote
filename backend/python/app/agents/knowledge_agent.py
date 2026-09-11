from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# print(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print(PROJECT_ROOT)

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import uuid4
from openai import OpenAI
from app.core.config import settings

RelationType = Literal["包括", "关联", "冲突"]

ALLOWED_RELATIONS: tuple[RelationType, ...] = ("包括", "关联", "冲突")


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    title: str
    content: str
    category: str = "计算机基础"
    keywords: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class KnowledgeRelation:
    source_id: str
    target_id: str
    relation: RelationType
    reason: str
    evidence: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class KnowledgeAssociationResult:
    source: KnowledgeItem
    candidates: tuple[KnowledgeItem, ...]
    relations: tuple[KnowledgeRelation, ...]
    merged_content: str
    prompt: str


DEFAULT_KNOWLEDGE_ITEMS: tuple[KnowledgeItem, ...] = (
    KnowledgeItem(
        id="K001",
        title="进程与线程",
        category="操作系统",
        content=(
            "进程是资源分配的基本单位，线程是 CPU 调度的基本单位。"
            "同一进程内的线程共享内存空间，但进程之间通常拥有独立地址空间。"
        ),
        keywords=("进程", "线程", "资源分配", "CPU 调度", "共享内存", "地址空间"),
        tags=("操作系统", "并发"),
    ),
    KnowledgeItem(
        id="K002",
        title="虚拟内存",
        category="操作系统",
        content=(
            "虚拟内存通过页表完成虚拟地址到物理地址的映射。"
            "它包括分页、缺页中断、页面置换等机制，用于实现地址隔离和更高的内存利用率。"
        ),
        keywords=("虚拟内存", "页表", "分页", "缺页中断", "地址隔离", "页面置换"),
        tags=("操作系统", "内存"),
    ),
    KnowledgeItem(
        id="K003",
        title="TCP 三次握手",
        category="计算机网络",
        content=(
            "TCP 三次握手用于建立可靠连接，包括 SYN、SYN-ACK、ACK 三个阶段。"
            "它会同步双方初始序列号，并确认双方收发能力。"
        ),
        keywords=("TCP", "三次握手", "SYN", "ACK", "可靠连接", "序列号"),
        tags=("网络", "TCP"),
    ),
    KnowledgeItem(
        id="K004",
        title="HTTP 与 HTTPS",
        category="计算机网络",
        content=(
            "HTTPS 是在 HTTP 基础上加入 TLS 加密和证书校验。"
            "HTTP 语义本身是无状态的，但底层可以复用 TCP 连接。"
        ),
        keywords=("HTTP", "HTTPS", "TLS", "证书", "加密", "TCP", "无状态"),
        tags=("网络", "HTTP"),
    ),
    KnowledgeItem(
        id="K005",
        title="数据库索引",
        category="数据库",
        content=(
            "数据库索引通过额外的数据结构减少扫描范围，提升查询效率。"
            "常见索引包括 B+ 树索引、哈希索引、联合索引，但索引会增加写入维护成本。"
        ),
        keywords=("数据库", "索引", "B+ 树", "哈希", "查询", "写入成本"),
        tags=("数据库", "MySQL"),
    ),
)

DEFAULT_SOURCE_KNOWLEDGE = KnowledgeItem(
    id="K100",
    title="HTTPS 为什么需要 TCP",
    category="计算机网络",
    content=(
        "HTTPS 使用 TLS 保护 HTTP 数据传输，通常运行在 TCP 之上。"
        "TCP 负责可靠连接，TLS 负责加密、身份认证和完整性保护。"
    ),
    keywords=("HTTPS", "TLS", "HTTP", "TCP", "可靠连接", "加密"),
    tags=("网络", "HTTPS"),
)

RELATION_PROMPT_TEMPLATE = """
你是一个计算机知识整合智能体。

你的任务是读取一条新知识 source，并将它分别与候选知识 candidates 中的每一条进行比较，判断 source 与候选知识之间是否存在关系。

只允许输出三类关系：
1. 包括：source 明确包含 candidate，或者 candidate 明确包含 source，或者一方是另一方的组成部分。
2. 关联：二者共享关键概念、上下游流程、同一主题、依赖关系，或者能在面试答案中互相补充。
3. 冲突：二者存在概念边界、目标取舍、性能代价、适用条件差异，或者表述上容易被误解为矛盾。

判断规则：
- 不写 RAG 检索逻辑，candidates 已经视为检索结果。
- 不使用 LangChain。
- 每条关系都必须包含 target_id、relation、reason、evidence、confidence。
- target_id 必须来自 candidates。
- relation 只能是：包括、关联、冲突。
- 如果没有足够关系，不要输出该候选知识。
- 最后生成 merged_content，用自然语言整合 source 与所有有关联的候选知识。

严格返回 JSON，格式如下：
{
  "relations": [
    {
      "target_id": "候选知识ID",
      "relation": "包括 | 关联 | 冲突",
      "reason": "判断原因",
      "evidence": ["证据1", "证据2"],
      "confidence": 0.0
    }
  ],
  "merged_content": "整合后的知识说明"
}
""".strip()


class KnowledgeAssociationAgent:
    def __init__(self, prompt_template: str = RELATION_PROMPT_TEMPLATE) -> None:
        
        
        print(settings.agent_api_key)
        print(settings.agent_model)
        print(settings.agent_timeout_seconds)
        
        self.client = OpenAI(
            api_key=settings.agent_api_key,
            base_url=settings.agent_api_base_url.strip(),
            timeout=settings.agent_timeout_seconds,
        )
        
        self.prompt_template = prompt_template


    def associate(
        self,
        source: KnowledgeItem,
        candidates: tuple[KnowledgeItem, ...] | None = None,
    ) -> KnowledgeAssociationResult:
        candidate_items = candidates or DEFAULT_KNOWLEDGE_ITEMS
        prompt = self._build_prompt(source=source, candidates=candidate_items)
        data = self._decide_with_model(prompt=prompt)

        if data is None:
            data = self._decide_locally(source=source, candidates=candidate_items)

        relations = self._parse_relations(source=source, candidates=candidate_items, data=data)
        merged_content = str(data.get("merged_content") or self._merge_locally(source, candidate_items, relations))

        return KnowledgeAssociationResult(
            source=source,
            candidates=candidate_items,
            relations=tuple(relations),
            merged_content=merged_content,
            prompt=prompt,
        )

    def run(self, items: tuple[KnowledgeItem, ...] | None = None) -> KnowledgeAssociationResult:
        return self.associate(source=DEFAULT_SOURCE_KNOWLEDGE, candidates=items or DEFAULT_KNOWLEDGE_ITEMS)

    def _build_prompt(self, source: KnowledgeItem, candidates: tuple[KnowledgeItem, ...]) -> str:
        payload = {
            "source": asdict(source),
            "candidates": [asdict(candidate) for candidate in candidates],
        }

        return f"{self.prompt_template}\n\n输入知识：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    def _decide_with_model(self, prompt: str) -> dict[str, Any] | None:
        if not settings.agent_api_key or settings.agent_api_key == "change-me":
            return None

        response = self.client.chat.completions.create(
            model=settings.agent_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "你是严格的知识关系判断智能体，只输出 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    def _decide_locally(self, source: KnowledgeItem, candidates: tuple[KnowledgeItem, ...]) -> dict[str, Any]:
        relations = []

        for candidate in candidates:
            relation = self._detect_include(source, candidate)
            if relation is None:
                relation = self._detect_conflict(source, candidate)
            if relation is None:
                relation = self._detect_related(source, candidate)
            if relation is not None:
                relations.append(
                    {
                        "target_id": relation.target_id,
                        "relation": relation.relation,
                        "reason": relation.reason,
                        "evidence": list(relation.evidence),
                        "confidence": relation.confidence,
                    }
                )

        parsed_relations = self._parse_relations(source=source, candidates=candidates, data={"relations": relations})

        return {
            "relations": relations,
            "merged_content": self._merge_locally(source, candidates, parsed_relations),
        }

    def _detect_include(self, source: KnowledgeItem, candidate: KnowledgeItem) -> KnowledgeRelation | None:
        source_text = self._joined_text(source)
        candidate_text = self._joined_text(candidate)
        include_markers = ("包括", "包含", "组成", "属于", "一部分")

        source_mentions_candidate = candidate.title in source_text
        candidate_mentions_source = source.title in candidate_text
        has_include_marker = any(marker in source_text or marker in candidate_text for marker in include_markers)

        if has_include_marker and (source_mentions_candidate or candidate_mentions_source):
            evidence = candidate.title if source_mentions_candidate else source.title
            return KnowledgeRelation(
                source_id=source.id,
                target_id=candidate.id,
                relation="包括",
                reason=f"{source.title} 与 {candidate.title} 存在明确的包含或组成关系。",
                evidence=(evidence,),
                confidence=0.9,
            )

        return None

    def _detect_related(self, source: KnowledgeItem, candidate: KnowledgeItem) -> KnowledgeRelation | None:
        shared_keywords = self._shared_values(source.keywords, candidate.keywords)
        shared_tags = self._shared_values(source.tags, candidate.tags)
        same_category = source.category == candidate.category

        if len(shared_keywords) >= 2 or shared_tags or same_category:
            evidence = shared_keywords[:4] or shared_tags[:4] or (source.category,)
            confidence = 0.84 if len(shared_keywords) >= 2 else 0.72
            return KnowledgeRelation(
                source_id=source.id,
                target_id=candidate.id,
                relation="关联",
                reason=f"{source.title} 与 {candidate.title} 在主题、关键词或知识链路上相关。",
                evidence=evidence,
                confidence=confidence,
            )

        return None

    def _detect_conflict(self, source: KnowledgeItem, candidate: KnowledgeItem) -> KnowledgeRelation | None:
        source_text = self._joined_text(source)
        candidate_text = self._joined_text(candidate)
        conflict_pairs = (
            ("无状态", "有连接"),
            ("共享内存", "独立地址空间"),
            ("加密", "性能开销"),
            ("查询效率", "写入成本"),
            ("可靠连接", "网络开销"),
            ("内存利用率", "缺页中断"),
        )

        for left, right in conflict_pairs:
            source_has_left = left in source_text
            source_has_right = right in source_text
            candidate_has_left = left in candidate_text
            candidate_has_right = right in candidate_text

            if source_has_left and candidate_has_right:
                return self._build_conflict(source, candidate, left, right)
            if source_has_right and candidate_has_left:
                return self._build_conflict(source, candidate, right, left)

        return None

    def _build_conflict(
        self,
        source: KnowledgeItem,
        candidate: KnowledgeItem,
        source_signal: str,
        candidate_signal: str,
    ) -> KnowledgeRelation:
        return KnowledgeRelation(
            source_id=source.id,
            target_id=candidate.id,
            relation="冲突",
            reason=(
                f"{source.title} 中的 {source_signal} 与 {candidate.title} 中的 "
                f"{candidate_signal} 存在概念边界或取舍差异。"
            ),
            evidence=(source_signal, candidate_signal),
            confidence=0.8,
        )

    def _parse_relations(
        self,
        source: KnowledgeItem,
        candidates: tuple[KnowledgeItem, ...],
        data: dict[str, Any],
    ) -> list[KnowledgeRelation]:
        candidate_ids = {candidate.id for candidate in candidates}
        relations: list[KnowledgeRelation] = []

        for item in data.get("relations", []):
            target_id = str(item.get("target_id", ""))
            relation = item.get("relation")

            if target_id not in candidate_ids or relation not in ALLOWED_RELATIONS:
                continue

            confidence = float(item.get("confidence", 0.0))
            confidence = max(0.0, min(confidence, 1.0))
            evidence = tuple(str(value) for value in item.get("evidence", []) if value)

            relations.append(
                KnowledgeRelation(
                    source_id=source.id,
                    target_id=target_id,
                    relation=relation,
                    reason=str(item.get("reason") or "未提供原因"),
                    evidence=evidence,
                    confidence=confidence,
                )
            )

        return self._deduplicate(relations)

    def _deduplicate(self, relations: list[KnowledgeRelation]) -> list[KnowledgeRelation]:
        seen: set[tuple[str, str]] = set()
        unique_relations: list[KnowledgeRelation] = []

        for relation in sorted(relations, key=lambda item: item.confidence, reverse=True):
            key = (relation.target_id, relation.relation)
            if key in seen:
                continue
            seen.add(key)
            unique_relations.append(relation)

        return unique_relations

    def _merge_locally(
        self,
        source: KnowledgeItem,
        candidates: tuple[KnowledgeItem, ...],
        relations: list[KnowledgeRelation],
    ) -> str:
        candidate_map = {candidate.id: candidate for candidate in candidates}
        related_titles = [candidate_map[relation.target_id].title for relation in relations if relation.target_id in candidate_map]

        if not related_titles:
            return f"{source.title}：{source.content}"

        return (
            f"{source.title} 可以与 {'、'.join(related_titles)} 一起理解。"
            f"核心内容是：{source.content}"
        )

    def _joined_text(self, item: KnowledgeItem) -> str:
        return " ".join((item.title, item.content, item.category, *item.keywords, *item.tags))

    def _shared_values(self, left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
        right_values = set(right)
        return tuple(value for value in left if value in right_values)


def build_knowledge_item(
    title: str,
    content: str,
    category: str = "计算机基础",
    keywords: tuple[str, ...] | None = None,
    tags: tuple[str, ...] | None = None,
    item_id: str | None = None,
) -> KnowledgeItem:
    return KnowledgeItem(
        id=item_id or f"K-{uuid4().hex[:8]}",
        title=title,
        content=content,
        category=category,
        keywords=keywords or (),
        tags=tags or (),
    )



if __name__ == "__main__":
    agent = KnowledgeAssociationAgent()
    result = agent.associate(
        source=build_knowledge_item("内存利用率", "内存利用率是指计算机内存中未被使用的比例。"),
        candidates=(
            build_knowledge_item("内存利用率", "内存利用率是指计算机内存中未被使用的比例。"),
            build_knowledge_item("缺页中断", "缺页中断是指计算机在执行程序时，由于内存中没有足够的空间来存储当前程序的指令或数据，而需要从磁盘读取数据到内存中的情况。"),
        ),
    )
    print(result)
