"""Prepare explicit docs/aggregate input, without exposing bank data to an AI reviewer."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("README.md", "docs/TASK.md", "docs/REQUIREMENTS.md", "docs/PITCH.md", "docs/JURY_BRIEF.md",
         "docs/TESTING.md", "docs/METRICS.md", "docs/ARCHITECTURE.md")


def build(root: Path) -> str:
    sections = ["Review this packet only. No data/model/tool access is needed. "
                "Assess supported claims and missing evidence; do not invent independent quality/SLA scores."]
    for relative in FILES:
        path = root / relative
        if path.is_file() and path.resolve().is_relative_to(root.resolve()):
            sections.append(f"\n--- {relative} ---\n{path.read_text(encoding='utf-8')}")
    path = root / "service/reports/quality-latest.json"
    if path.is_file() and path.resolve().is_relative_to(root.resolve()):
        report = json.loads(path.read_text(encoding="utf-8"))
        fields = ("purpose", "coverage", "binary", "joint_events", "per_type", "label_confidence", "input_sha256")
        safe = {key: report[key] for key in fields if key in report}
        sections.append("\n--- Aggregate quality ---\n" + json.dumps(safe, ensure_ascii=False, indent=2))
    return "\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(ROOT), encoding="utf-8")
    print(f"Review packet: {args.out}; no external reviewer launched")


if __name__ == "__main__":
    main()
