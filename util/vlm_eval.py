#!/usr/bin/env python3
"""VLM evaluation harness.

Compares chat.vlm_assessments predictions against locations.classification
ground truth and reports confusion matrix + per-class P/R/F1 + weighted F1.

Usage:
    python util/vlm_eval.py --disaster-id hurricane-florence --out data/eval.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sqlalchemy as sa

from app.db import get_engine
from app.env_loader import load_app_env

load_app_env()

CLASSES = ["none", "minor", "severe", "destroyed"]
ALL_CLASSES = CLASSES + ["unknown"]

CANONICAL_TO_SHORT = {
    "no-damage": "none",
    "minor-damage": "minor",
    "major-damage": "severe",
    "destroyed": "destroyed",
    "unknown": "unknown",
}


@dataclass
class EvalReport:
    confusion: dict[tuple[str, str], int]
    per_class: dict[str, dict[str, float]]
    weighted_f1_uniform: float
    weighted_f1_support: float
    overall_accuracy: float
    total: int
    by_class_count: dict[str, int]

    def to_markdown(self, *, title: str = "VLM Evaluation") -> str:
        lines: list[str] = [f"# {title}", ""]
        lines.append(f"- **Total evaluated:** {self.total}")
        lines.append(f"- **Overall accuracy:** {self.overall_accuracy:.3f}")
        lines.append(f"- **Weighted F1 (by support):** {self.weighted_f1_support:.3f}")
        lines.append(f"- **Macro F1 (uniform):** {self.weighted_f1_uniform:.3f}")
        lines.append("")
        lines.append("## Per-class metrics")
        lines.append("")
        lines.append("| Class | Support | Precision | Recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for cls in ALL_CLASSES:
            m = self.per_class.get(cls, {})
            lines.append(
                f"| {cls} | {self.by_class_count.get(cls, 0)} "
                f"| {m.get('precision', 0.0):.3f} "
                f"| {m.get('recall', 0.0):.3f} "
                f"| {m.get('f1', 0.0):.3f} |"
            )
        lines.append("")
        lines.append("## Confusion matrix (rows = truth, cols = predicted)")
        lines.append("")
        header = "| truth \\ pred | " + " | ".join(ALL_CLASSES) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(ALL_CLASSES) + 1))
        for truth in ALL_CLASSES:
            row = [str(self.confusion.get((truth, pred), 0)) for pred in ALL_CLASSES]
            lines.append(f"| {truth} | " + " | ".join(row) + " |")
        lines.append("")
        return "\n".join(lines)


def _normalize_prediction(canonical_label: str | None) -> str:
    if not canonical_label:
        return "unknown"
    return CANONICAL_TO_SHORT.get(canonical_label, canonical_label)


def _compute_metrics(
    pairs: list[tuple[str, str]],
) -> EvalReport:
    confusion: dict[tuple[str, str], int] = Counter()
    by_class: dict[str, int] = Counter()
    correct = 0
    for truth, pred in pairs:
        confusion[(truth, pred)] += 1
        by_class[truth] += 1
        if truth == pred:
            correct += 1

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for truth, pred in pairs:
        for cls in ALL_CLASSES:
            if truth == cls and pred == cls:
                tp[cls] += 1
            elif truth != cls and pred == cls:
                fp[cls] += 1
            elif truth == cls and pred != cls:
                fn[cls] += 1

    per_class: dict[str, dict[str, float]] = {}
    weighted_sum = 0.0
    total_support = 0
    macro_sum = 0.0
    macro_count = 0
    for cls in ALL_CLASSES:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = tp[cls] + fn[cls]
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        weighted_sum += f1 * support
        total_support += support
        if support > 0:
            macro_sum += f1
            macro_count += 1

    weighted_f1 = weighted_sum / total_support if total_support else 0.0
    macro_f1 = macro_sum / macro_count if macro_count else 0.0
    total = len(pairs)

    return EvalReport(
        confusion=dict(confusion),
        per_class=per_class,
        weighted_f1_uniform=macro_f1,
        weighted_f1_support=weighted_f1,
        overall_accuracy=(correct / total) if total else 0.0,
        total=total,
        by_class_count=dict(by_class),
    )


def evaluate(
    *,
    disaster_id: str | None = None,
    include_unknown_truth: bool = False,
) -> EvalReport:
    engine = get_engine()
    where = ["a.damage_level IS NOT NULL"]
    params: dict[str, object] = {}
    if disaster_id:
        where.append("ip.disaster_id = :disaster_id")
        params["disaster_id"] = disaster_id
    if not include_unknown_truth:
        where.append("l.classification <> 'unknown'")
    query = f"""
        SELECT l.classification AS truth, a.damage_level AS pred
        FROM locations l
        JOIN image_pairs ip ON ip.id = l.image_pair_id
        JOIN chat.vlm_assessments a ON a.location_id = l.id
        WHERE { ' AND '.join(where) }
    """
    with engine.connect() as conn:
        rows = conn.execute(sa.text(query), params).all()
    pairs = [(row.truth, _normalize_prediction(row.pred)) for row in rows]
    return _compute_metrics(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions vs ground truth.")
    parser.add_argument("--disaster-id", default="hurricane-florence")
    parser.add_argument(
        "--include-unknown-truth",
        action="store_true",
        help="Include locations where ground truth is 'unknown' (noisy).",
    )
    parser.add_argument(
        "--out", default=None, help="Write Markdown report to this path."
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Also write a JSON dump of the raw metrics.",
    )
    parser.add_argument(
        "--title",
        default="VLM Evaluation for Hurricane Florence",
    )
    args = parser.parse_args()

    report = evaluate(
        disaster_id=args.disaster_id,
        include_unknown_truth=args.include_unknown_truth,
    )
    md = report.to_markdown(title=args.title)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\nReport written to {args.out}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "confusion": {f"{k[0]}->{k[1]}": v for k, v in report.confusion.items()},
                    "per_class": report.per_class,
                    "weighted_f1_support": report.weighted_f1_support,
                    "weighted_f1_uniform": report.weighted_f1_uniform,
                    "overall_accuracy": report.overall_accuracy,
                    "total": report.total,
                    "by_class_count": report.by_class_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
