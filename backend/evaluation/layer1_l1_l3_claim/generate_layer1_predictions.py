from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_GOLD_PATH = Path("backend/testdata/408_notes_claims_fixture.json")
DEFAULT_OUTPUT_PATH = Path("backend/evaluation/results/layer1_l1_l3_claim_predictions.json")


def ensure_backend_importable() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    backend_python = repo_root / "backend" / "python"

    if str(backend_python) not in sys.path:
        sys.path.insert(0, str(backend_python))


def generate_predictions(gold_path: Path, output_path: Path, *, limit: int | None = None) -> dict[str, Any]:
    ensure_backend_importable()

    from app.core.evolution import analyze_note

    data = json.loads(gold_path.read_text(encoding="utf-8-sig"))
    notes = data.get("notes", [])
    predicted_notes: list[dict[str, Any]] = []

    for raw_note in notes[:limit]:
        note = {
            "id": str(raw_note["id"]),
            "title": str(raw_note.get("title", "")),
            "tags": ",".join(str(item) for item in raw_note.get("expected", {}).get("l3", [])),
            "body": str(raw_note.get("l1", "")),
            "created_at": "",
            "updated_at": "",
        }
        analysis = analyze_note(note)
        representation = analysis["representation"]
        claims = [
            {
                "claim": claim.get("claim_text", ""),
                "evidence": claim.get("source_text", ""),
                "subject": claim.get("subject", ""),
                "predicate": claim.get("predicate", ""),
                "object_text": claim.get("object_text", ""),
                "keywords": claim.get("keywords", []),
                "confidence": claim.get("confidence", 0.0),
            }
            for claim in analysis["claims"]
        ]
        predicted_notes.append(
            {
                "id": note["id"],
                "title": note["title"],
                "l1": note["body"],
                "predicted": {
                    "l2": representation.get("l2_summary", ""),
                    "l3": representation.get("keywords", []),
                    "l3_text": representation.get("l3_text", ""),
                    "claims": claims,
                },
            }
        )

    report = {
        "schema_version": "1.0",
        "input_path": str(gold_path),
        "notes": predicted_notes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real Evonote Layer1 predictions with the configured LLM.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH, help="Gold fixture JSON path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH, help="Prediction output JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of notes to generate for a quick live run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_predictions(args.gold, args.out, limit=args.limit)
    print(json.dumps({"output_path": str(args.out), "notes": len(report["notes"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
