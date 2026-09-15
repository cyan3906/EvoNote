from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for LLM judging.
    OpenAI = None  # type: ignore[assignment]


SupportStatus = Literal["supported", "partially_supported", "unsupported"]


DEFAULT_GOLD_PATH = Path("backend/testdata/408_notes_claims_fixture.json")
DEFAULT_OUTPUT_PATH = Path("backend/evaluation/results/layer1_l1_l3_claim_metrics.json")


@dataclass(frozen=True)
class Claim:
    id: str
    text: str
    evidence: str


@dataclass(frozen=True)
class NoteExtraction:
    id: str
    l1: str
    l2: str
    l3: list[str]
    claims: list[Claim]


class LLMJudge:
    def __init__(self) -> None:
        self.model = os.getenv("EVAL_JUDGE_MODEL") or os.getenv("AGENT_MODEL") or "gpt-4o-mini"
        api_key = os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("EVAL_JUDGE_BASE_URL") or os.getenv("AGENT_API_BASE_URL") or None
        self.enabled = bool(api_key and OpenAI is not None)
        self.client = OpenAI(api_key=api_key, base_url=base_url) if self.enabled else None

    def judge_l2(self, l1: str, predicted_l2: str, gold_l2: str) -> dict[str, float]:
        if not self.enabled:
            return heuristic_l2_judgment(l1, predicted_l2, gold_l2)

        data = self._json_judge(
            system="你是严格的摘要评估裁判，只输出JSON。",
            user={
                "task": "判断predicted_l2是否忠实于l1，并是否覆盖gold_l2中的核心信息。",
                "labels": {
                    "faithfulness": "1表示predicted_l2没有引入L1之外的事实；0表示有编造或明显歪曲。",
                    "coverage": "0到1的小数，表示predicted_l2覆盖gold_l2核心信息的程度。",
                },
                "l1": l1,
                "gold_l2": gold_l2,
                "predicted_l2": predicted_l2,
            },
        )
        return {
            "faithfulness": as_binary(data.get("faithfulness")),
            "coverage": clamp01(data.get("coverage")),
        }

    def semantic_match(self, gold: str, predicted: str, kind: str) -> bool:
        if not self.enabled:
            return heuristic_text_match(gold, predicted)

        data = self._json_judge(
            system="你是严格的语义等价裁判，只输出JSON。",
            user={
                "task": f"判断predicted_{kind}是否与gold_{kind}语义等价或覆盖同一个核心知识点。",
                "gold": gold,
                "predicted": predicted,
                "output_schema": {"match": "0或1"},
            },
        )
        return bool(as_binary(data.get("match")))

    def claim_support(self, l1: str, claim: Claim) -> SupportStatus:
        if not self.enabled:
            return heuristic_claim_support(l1, claim)

        data = self._json_judge(
            system="你是严格的证据支持性裁判，只输出JSON。",
            user={
                "task": "判断predicted_claim是否被L1和predicted_evidence支持。",
                "labels": {
                    "supported": "claim完全被原文支持，且没有引入额外事实。",
                    "partially_supported": "claim部分被支持，但遗漏条件、过度泛化、混入额外信息或evidence不充分。",
                    "unsupported": "claim没有被原文支持，或主要内容来自原文之外。",
                },
                "l1": l1,
                "predicted_claim": claim.text,
                "predicted_evidence": claim.evidence,
                "output_schema": {"support_status": "supported | partially_supported | unsupported"},
            },
        )
        status = str(data.get("support_status") or "").strip()
        if status in {"supported", "partially_supported", "unsupported"}:
            return status  # type: ignore[return-value]
        return "unsupported"

    def _json_judge(self, *, system: str, user: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


def evaluate_layer1(
    gold_path: Path,
    prediction_path: Path,
    output_path: Path,
    *,
    use_llm: bool,
) -> dict[str, Any]:
    gold_notes = load_notes(gold_path, field_name="expected")
    predicted_notes = load_notes(prediction_path, field_name="predicted")
    judge = LLMJudge()

    if not use_llm:
        judge.enabled = False

    note_results: list[dict[str, Any]] = []
    l2_faithful_count = 0.0
    l2_coverage_total = 0.0
    l3_matched_predicted = 0
    l3_matched_gold = 0
    l3_predicted_total = 0
    l3_gold_total = 0
    claim_matched_predicted = 0
    claim_matched_gold = 0
    claim_predicted_total = 0
    claim_gold_total = 0
    supported_claims = 0
    partially_supported_claims = 0
    unsupported_claims = 0

    for note_id, gold_note in gold_notes.items():
        predicted_note = predicted_notes.get(note_id, empty_prediction(note_id, gold_note.l1))
        l2_result = judge.judge_l2(gold_note.l1, predicted_note.l2, gold_note.l2)
        l2_faithful_count += l2_result["faithfulness"]
        l2_coverage_total += l2_result["coverage"]

        l3_matches = match_items(
            gold_note.l3,
            predicted_note.l3,
            lambda gold, pred: judge.semantic_match(gold, pred, "l3_entity"),
        )
        claim_matches = match_items(
            [claim.text for claim in gold_note.claims],
            [claim.text for claim in predicted_note.claims],
            lambda gold, pred: judge.semantic_match(gold, pred, "claim"),
        )

        support_counts = {"supported": 0, "partially_supported": 0, "unsupported": 0}
        for predicted_claim in predicted_note.claims:
            status = judge.claim_support(gold_note.l1, predicted_claim)
            support_counts[status] += 1

        l3_matched_predicted += len(l3_matches)
        l3_matched_gold += len({gold_index for gold_index, _ in l3_matches})
        l3_predicted_total += len(predicted_note.l3)
        l3_gold_total += len(gold_note.l3)
        claim_matched_predicted += len(claim_matches)
        claim_matched_gold += len({gold_index for gold_index, _ in claim_matches})
        claim_predicted_total += len(predicted_note.claims)
        claim_gold_total += len(gold_note.claims)
        supported_claims += support_counts["supported"]
        partially_supported_claims += support_counts["partially_supported"]
        unsupported_claims += support_counts["unsupported"]

        note_results.append(
            {
                "note_id": note_id,
                "l2": {
                    "faithfulness": l2_result["faithfulness"],
                    "coverage": l2_result["coverage"],
                },
                "l3": {
                    "gold_count": len(gold_note.l3),
                    "predicted_count": len(predicted_note.l3),
                    "matched_count": len(l3_matches),
                },
                "claims": {
                    "gold_count": len(gold_note.claims),
                    "predicted_count": len(predicted_note.claims),
                    "matched_count": len(claim_matches),
                    "support_counts": support_counts,
                },
            }
        )

    note_count = len(gold_notes)
    metrics = {
        "l2_faithfulness_rate": safe_divide(l2_faithful_count, note_count),
        "l2_coverage_score": safe_divide(l2_coverage_total, note_count),
        "l3_precision": safe_divide(l3_matched_predicted, l3_predicted_total),
        "l3_recall": safe_divide(l3_matched_gold, l3_gold_total),
        "claim_precision": safe_divide(claim_matched_predicted, claim_predicted_total),
        "claim_recall": safe_divide(claim_matched_gold, claim_gold_total),
        "claim_evidence_support_rate": safe_divide(supported_claims, claim_predicted_total),
        "claim_hallucination_rate": safe_divide(unsupported_claims, claim_predicted_total),
        "claim_partial_support_rate": safe_divide(partially_supported_claims, claim_predicted_total),
    }
    report = {
        "layer": "layer1_l1_l3_claim",
        "judge": "llm" if judge.enabled else "heuristic",
        "gold_path": str(gold_path),
        "prediction_path": str(prediction_path),
        "metrics": metrics,
        "counts": {
            "notes": note_count,
            "gold_l3": l3_gold_total,
            "predicted_l3": l3_predicted_total,
            "matched_l3": l3_matched_gold,
            "gold_claims": claim_gold_total,
            "predicted_claims": claim_predicted_total,
            "matched_claims": claim_matched_gold,
            "supported_claims": supported_claims,
            "partially_supported_claims": partially_supported_claims,
            "unsupported_claims": unsupported_claims,
        },
        "notes": note_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_notes(path: Path, *, field_name: str) -> dict[str, NoteExtraction]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_notes = data.get("notes", data if isinstance(data, list) else [])
    notes: dict[str, NoteExtraction] = {}

    for index, raw_note in enumerate(raw_notes):
        note_id = str(raw_note.get("id") or raw_note.get("note_id") or f"note_{index + 1:03d}")
        payload = select_payload(raw_note, field_name)
        l1 = clean_text(raw_note.get("l1") or raw_note.get("content") or payload.get("l1") or "")
        l2 = clean_text(
            payload.get("l2")
            or payload.get("l2_summary")
            or raw_note.get("l2")
            or raw_note.get("l2_summary")
            or raw_note.get("representation", {}).get("l2_summary", "")
        )
        l3 = normalize_l3(payload, raw_note)
        claims = normalize_claims(payload.get("claims") or raw_note.get("claims") or flatten_block_claims(raw_note))
        notes[note_id] = NoteExtraction(id=note_id, l1=l1, l2=l2, l3=l3, claims=claims)

    return notes


def select_payload(raw_note: dict[str, Any], field_name: str) -> dict[str, Any]:
    if field_name in raw_note and isinstance(raw_note[field_name], dict):
        return raw_note[field_name]
    if "expected" in raw_note and isinstance(raw_note["expected"], dict):
        return raw_note["expected"]
    if "predicted" in raw_note and isinstance(raw_note["predicted"], dict):
        return raw_note["predicted"]
    return raw_note


def normalize_l3(payload: dict[str, Any], raw_note: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for source in (
        payload.get("l3"),
        payload.get("keywords"),
        raw_note.get("l3"),
        raw_note.get("keywords"),
        raw_note.get("representation", {}).get("keywords", []),
    ):
        if isinstance(source, list):
            values.extend(source)
        elif isinstance(source, str):
            values.append(source)

    if not values:
        l3_text = payload.get("l3_text") or raw_note.get("l3_text") or raw_note.get("representation", {}).get("l3_text", "")
        values.extend(split_l3_text(str(l3_text)))

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_text(value)
        key = normalize_text(item)
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def normalize_claims(raw_claims: Any) -> list[Claim]:
    if not isinstance(raw_claims, list):
        return []

    claims: list[Claim] = []
    for index, raw_claim in enumerate(raw_claims):
        if isinstance(raw_claim, str):
            text = clean_text(raw_claim)
            evidence = ""
            claim_id = f"claim_{index + 1:03d}"
        elif isinstance(raw_claim, dict):
            text = clean_text(raw_claim.get("claim") or raw_claim.get("claim_text") or raw_claim.get("text") or "")
            evidence = clean_text(raw_claim.get("evidence") or raw_claim.get("source_text") or "")
            claim_id = str(raw_claim.get("id") or raw_claim.get("claim_id") or f"claim_{index + 1:03d}")
        else:
            continue

        if text:
            claims.append(Claim(id=claim_id, text=text, evidence=evidence))
    return claims


def flatten_block_claims(raw_note: dict[str, Any]) -> list[Any]:
    claims: list[Any] = []
    for block in raw_note.get("blocks", []):
        if isinstance(block, dict) and isinstance(block.get("claims"), list):
            claims.extend(block["claims"])
    return claims


def empty_prediction(note_id: str, l1: str) -> NoteExtraction:
    return NoteExtraction(id=note_id, l1=l1, l2="", l3=[], claims=[])


def match_items(
    gold_items: list[str],
    predicted_items: list[str],
    is_match: Any,
) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    used_gold: set[int] = set()

    for predicted_index, predicted in enumerate(predicted_items):
        for gold_index, gold in enumerate(gold_items):
            if gold_index in used_gold:
                continue
            if is_match(gold, predicted):
                used_gold.add(gold_index)
                matches.append((gold_index, predicted_index))
                break
    return matches


def heuristic_l2_judgment(l1: str, predicted_l2: str, gold_l2: str) -> dict[str, float]:
    if not predicted_l2:
        return {"faithfulness": 0.0, "coverage": 0.0}
    if normalize_text(predicted_l2) == normalize_text(gold_l2):
        return {"faithfulness": 1.0, "coverage": 1.0}

    l1_tokens = set(tokenize(l1))
    predicted_tokens = set(tokenize(predicted_l2))
    gold_tokens = set(tokenize(gold_l2))
    extra_tokens = predicted_tokens - l1_tokens
    meaningful_extra = {token for token in extra_tokens if len(token) > 1}
    faithfulness = 1.0 if len(meaningful_extra) <= max(2, len(predicted_tokens) // 4) else 0.0
    coverage = jaccard(gold_tokens, predicted_tokens)
    return {"faithfulness": faithfulness, "coverage": coverage}


def heuristic_claim_support(l1: str, claim: Claim) -> SupportStatus:
    if claim.evidence and claim.evidence in l1:
        return "supported"
    claim_tokens = set(tokenize(claim.text))
    l1_tokens = set(tokenize(l1))
    if not claim_tokens:
        return "unsupported"

    overlap = len(claim_tokens & l1_tokens) / len(claim_tokens)
    if overlap >= 0.8:
        return "supported"
    if overlap >= 0.45:
        return "partially_supported"
    return "unsupported"


def heuristic_text_match(gold: str, predicted: str) -> bool:
    gold_norm = normalize_text(gold)
    predicted_norm = normalize_text(predicted)
    if not gold_norm or not predicted_norm:
        return False
    if gold_norm == predicted_norm or gold_norm in predicted_norm or predicted_norm in gold_norm:
        return True

    gold_tokens = set(tokenize(gold))
    predicted_tokens = set(tokenize(predicted))
    return jaccard(gold_tokens, predicted_tokens) >= 0.6


def split_l3_text(text: str) -> list[str]:
    return [item for item in re.split(r"[,，;；、\s]+", text) if item]


def tokenize(text: str) -> list[str]:
    normalized = str(text or "").lower()
    return re.findall(r"[a-z0-9_+#.-]{2,}|[\u4e00-\u9fff]{2,}", normalized)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def as_binary(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return 1.0 if float(value) >= 0.5 else 0.0
    except (TypeError, ValueError):
        return 0.0


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Evonote layer1 L1/L2/L3/claim extraction quality.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH, help="Gold fixture JSON path.")
    parser.add_argument("--pred", type=Path, default=None, help="Prediction JSON path. Defaults to --gold for smoke baseline.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output metrics JSON path.")
    parser.add_argument("--judge", choices=("auto", "llm", "heuristic"), default="auto", help="Judge mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = args.pred or args.gold
    use_llm = args.judge == "llm" or args.judge == "auto"
    report = evaluate_layer1(args.gold, prediction_path, args.out, use_llm=use_llm)
    print(json.dumps({"output_path": str(args.out), "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

