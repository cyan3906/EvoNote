from contextlib import contextmanager

from app.EvoRAG.entity_store.attribute_decider import AttributeDecisionMaker, build_attribute_decision_payload
from app.EvoRAG.entity_store.models import (
    AttributeCandidate,
    AttributeDecision,
    AttributeRetrievalResult,
    EntityAttributeInput,
    EntityScope,
    StoredEntityAttribute,
)
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.config import EvoRAGSettings


def incoming(attr_type: str, value: str, evidence: str = "source evidence") -> EntityAttributeInput:
    return EntityAttributeInput(attr_type=attr_type, value_text=value, evidence=evidence, confidence=0.8)


def stored(attribute_id: int, attr_type: str, value: str) -> StoredEntityAttribute:
    return StoredEntityAttribute(
        id=attribute_id,
        entity_id=3,
        scope=EntityScope(),
        attr_type=attr_type,
        value_text=value,
        value_fingerprint="fp",
        confidence=0.7,
        status="active",
    )


def result(input_index: int, attr_type: str, value: str, *candidates: AttributeCandidate, exact=None) -> AttributeRetrievalResult:
    return AttributeRetrievalResult(
        input_index=input_index,
        attr_type=attr_type,
        value_text=value,
        mode="hybrid" if candidates else "full_scan",
        group_size=len(candidates),
        exact_match=exact,
        candidates=list(candidates),
    )


def candidate(attribute_id: int, attr_type: str, value: str, *, rank: int = 1) -> AttributeCandidate:
    return AttributeCandidate(
        attribute_id=attribute_id,
        entity_id=3,
        attr_type=attr_type,
        value_text=value,
        confidence=0.7,
        rank=rank,
        source="mysql",
    )


def test_attribute_decider_maps_retrieval_results_to_conservative_actions() -> None:
    decider = AttributeDecisionMaker()
    attrs = [
        incoming("mechanism", "使用 SET NX EX 命令获取锁"),
        incoming("limitation", "如果业务执行时间超过锁 TTL，锁会提前释放，其他客户端可能获取同一资源的锁"),
        incoming("limitation", "单节点 Redis 分布式锁在主从切换时不会产生安全问题"),
            incoming("advantage", "实现成本低"),
            incoming("purpose", "减少读写阻塞"),
    ]
    exact = stored(10, "mechanism", "使用 SET NX EX 命令获取锁")

    decisions = decider.decide(
        attrs,
        [
            result(0, "mechanism", attrs[0].value_text, exact=exact),
            result(1, "limitation", attrs[1].value_text, candidate(20, "limitation", "Redis 分布式锁可能出现锁过期问题")),
            result(2, "limitation", attrs[2].value_text, candidate(30, "limitation", "单节点 Redis 分布式锁在主从切换时可能出现安全问题")),
            result(3, "advantage", attrs[3].value_text),
            result(4, "purpose", attrs[4].value_text, candidate(40, "purpose", "提高并发读性能")),
        ],
    )

    assert [decision.action for decision in decisions] == ["merge", "enrich", "conflict", "add", "add"]
    assert decisions[0].target_attribute_id == 10
    assert decisions[1].target_attribute_id == 20
    assert decisions[1].new_value_text == "Redis 分布式锁可能出现锁过期问题：如果业务执行时间超过锁 TTL，锁会提前释放，其他客户端可能获取同一资源的锁"
    assert decisions[2].target_attribute_id == 30
    assert decisions[3].target_attribute_id is None
    assert decisions[4].target_attribute_id is None
    assert decisions[4].reason == "candidate is related but not proven to be the same fact"


def test_attribute_decision_payload_limits_candidates_to_top5_and_removes_history_evidence() -> None:
    attr = incoming("mechanism", "通过 Read View 判断版本可见性", evidence="incoming evidence")
    retrieval = result(
        0,
        "mechanism",
        attr.value_text,
        *[candidate(index, "mechanism", f"candidate {index}", rank=index) for index in range(1, 8)],
    )

    payload = build_attribute_decision_payload(
        attr,
        retrieval,
        history_summaries={
            1: {
                "event_count": 3,
                "last_actions": ["add", "merge"],
                "recent_corrections": [{"original_action": "merge", "corrected_to": "add", "evidence": "hidden"}],
                "evidence": "hidden",
            }
        },
    )

    assert [item["attribute_id"] for item in payload["candidates"]] == [1, 2, 3, 4, 5]
    assert "evidence" not in str(payload["candidates"])


class ScriptedCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple | None]] = []
        self._rows: list[dict] = []
        self.lastrowid = 41

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, params=None) -> None:
        self.statements.append((query, params))
        if "SELECT id FROM evorag_entity_attributes" in query:
            self._rows = [{"id": self.lastrowid}]
        else:
            self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, cursor: ScriptedCursor) -> None:
        self.cursor_obj = cursor
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self) -> ScriptedCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True


def test_repository_applies_attribute_decisions_to_expected_write_paths() -> None:
    cursor = ScriptedCursor()
    connection = FakeConnection(cursor)
    repository = MySQLEntityRepository(EvoRAGSettings(_env_file=None))

    @contextmanager
    def connect():
        yield connection

    repository.connect = connect  # type: ignore[method-assign]
    decider = AttributeDecisionMaker()
    decisions = decider.decide(
        [
            incoming("mechanism", "同义补证据", "merge evidence"),
            incoming("limitation", "更完整说明", "enrich evidence"),
            incoming("limitation", "矛盾说明", "conflict evidence"),
            incoming("advantage", "新增优点", "add evidence"),
        ],
        [
            result(0, "mechanism", "同义补证据", candidate(10, "mechanism", "已有机制")),
            result(1, "limitation", "更完整说明", candidate(20, "limitation", "已有限制")),
            result(2, "limitation", "矛盾说明", candidate(30, "limitation", "已有矛盾限制")),
            result(3, "advantage", "新增优点"),
        ],
    )
    decisions[0].action = "merge"
    decisions[0].target_attribute_id = 10
    decisions[0].target_value_text = "已有机制"
    decisions[1].action = "enrich"
    decisions[1].target_attribute_id = 20
    decisions[1].target_value_text = "已有限制"
    decisions[1].new_value_text = "已有限制：更完整说明"
    decisions[2].action = "conflict"
    decisions[2].target_attribute_id = 30
    decisions[2].target_value_text = "已有矛盾限制"

    outcome = repository.apply_attribute_decisions(3, decisions)

    sql = "\n".join(statement for statement, _ in cursor.statements)
    assert "INSERT INTO evorag_entity_attribute_evidence" in sql
    assert "UPDATE evorag_entity_attributes" in sql
    assert "INSERT INTO evorag_entity_attribute_conflicts" in sql
    assert "INSERT INTO evorag_entity_attribute_decision_audit" in sql
    assert "INSERT INTO evorag_entity_attribute_events" in sql
    assert outcome.attribute_count == 2
    assert outcome.evidence_count == 3
    assert outcome.conflict_count == 1
    assert set(outcome.changed_attribute_ids) == {10, 20, 41}
    assert connection.committed is True


def test_repository_applies_update_decision_and_records_before_after_event() -> None:
    cursor = ScriptedCursor()
    connection = FakeConnection(cursor)
    repository = MySQLEntityRepository(EvoRAGSettings(_env_file=None))

    @contextmanager
    def connect():
        yield connection

    repository.connect = connect  # type: ignore[method-assign]
    decision = AttributeDecision(
        input_index=0,
        action="update",
        incoming_attribute=incoming("mechanism", "通过 Read View 判断版本可见性", "update evidence"),
        target_attribute_id=50,
        target_value_text="通过 Undo Log 判断版本可见性",
        new_value_text="通过 Read View 判断版本可见性",
        confidence=0.86,
        reason="incoming corrects outdated mechanism",
    )

    outcome = repository.apply_attribute_decisions(3, [decision])

    event_params = [
        params
        for statement, params in cursor.statements
        if "INSERT INTO evorag_entity_attribute_events" in statement
    ][0]
    assert "UPDATE evorag_entity_attributes" in "\n".join(statement for statement, _ in cursor.statements)
    assert outcome.changed_attribute_ids == [50]
    assert event_params[3] == "update"
    assert "通过 Undo Log 判断版本可见性" in event_params[5]
    assert "通过 Read View 判断版本可见性" in event_params[6]
