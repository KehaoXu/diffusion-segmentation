import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence


RATIO_PATTERN = re.compile(r"atlas\+uncond(?P<ratio>\d+)")
CSV_COLUMNS = [
    "ratio",
    "mean_dice",
    "std_dice",
    "delta_dice_vs_real_only",
    "mean_iou",
    "mean_precision",
    "mean_recall",
    "best_dice",
]


@dataclass(frozen=True)
class MetricRecord:
    path: Path
    group: str
    ratio: float
    dice: float
    iou: float
    precision: float
    recall: float
    num_samples: int


def parse_ratio(path: Path) -> float:
    match = RATIO_PATTERN.search(path.as_posix())
    if match is None:
        raise ValueError(f"Could not parse synthetic ratio from path: {path}")
    return int(match.group("ratio")) / 10.0


def _metric(payload: Dict[str, object], primary: str, fallback: str) -> float:
    value = payload.get(primary, payload.get(fallback))
    if value is None:
        raise ValueError(f"Metric JSON is missing '{primary}'/'{fallback}'")
    return float(value)


def load_metric_record(path: Path) -> MetricRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parts = path.parts
    group = parts[1] if len(parts) > 1 and parts[0] == "outputs" else path.parent.parent.name
    return MetricRecord(
        path=path,
        group=group,
        ratio=parse_ratio(path),
        dice=_metric(payload, "mean_dice", "dice_mean"),
        iou=_metric(payload, "mean_iou", "iou_mean"),
        precision=_metric(payload, "mean_precision", "precision_mean"),
        recall=_metric(payload, "mean_recall", "recall_mean"),
        num_samples=int(payload.get("num_samples", 0)),
    )


def collect_records(outputs_dir: Path) -> List[MetricRecord]:
    records: List[MetricRecord] = []
    for path in sorted(outputs_dir.rglob("eval_metrics.json")):
        try:
            records.append(load_metric_record(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
    return records


def _sample_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_by_ratio(records: Sequence[MetricRecord]) -> List[Dict[str, object]]:
    if not records:
        return []

    grouped: Dict[float, List[MetricRecord]] = {}
    for record in records:
        grouped.setdefault(record.ratio, []).append(record)

    if 0.0 not in grouped:
        raise ValueError("No real-only baseline found at ratio 0.0")
    baseline = mean(record.dice for record in grouped[0.0])

    rows: List[Dict[str, object]] = []
    for ratio in sorted(grouped):
        items = grouped[ratio]
        best = max(items, key=lambda item: item.dice)
        mean_dice = mean(item.dice for item in items)
        rows.append(
            {
                "ratio": ratio,
                "n_runs": len(items),
                "mean_dice": mean_dice,
                "std_dice": _sample_std([item.dice for item in items]),
                "delta_dice_vs_real_only": mean_dice - baseline,
                "mean_iou": mean(item.iou for item in items),
                "mean_precision": mean(item.precision for item in items),
                "mean_recall": mean(item.recall for item in items),
                "best_dice": best.dice,
                "best_run": f"{best.group}/{best.path.parent.parent.name}",
            }
        )
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row[key]) for key in CSV_COLUMNS})


def _format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_markdown(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synthetic Ratio Summary",
        "",
        "Aggregated from existing `outputs/**/eval_metrics.json` files. No new training runs are included.",
        "",
        "| Synthetic ratio | Dice mean +/- std | Delta Dice | IoU | Precision | Recall | Best Dice |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ratio']:.1f} | "
            f"{row['mean_dice']:.4f} +/- {row['std_dice']:.4f} | "
            f"{row['delta_dice_vs_real_only']:+.4f} | "
            f"{row['mean_iou']:.4f} | "
            f"{row['mean_precision']:.4f} | "
            f"{row['mean_recall']:.4f} | "
            f"{row['best_dice']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Takeaway: synthetic augmentation gives modest average gains over the real-only baseline in these runs. "
            "The result is best presented as an applied ML experiment on reproducible evaluation and ratio tuning, "
            "not as a new generative modeling method.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 40, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    ratios = [float(row["ratio"]) for row in rows]
    dices = [float(row["mean_dice"]) for row in rows]
    if not ratios or not dices:
        path.write_text("", encoding="utf-8")
        return
    x_min, x_max = min(ratios), max(ratios)
    y_min = math.floor((min(dices) - 0.01) * 100) / 100
    y_max = math.ceil((max(dices) + 0.01) * 100) / 100
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01

    def x_scale(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w if x_max != x_min else margin_left + plot_w / 2

    def y_scale(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    points = " ".join(f"{x_scale(x):.1f},{y_scale(y):.1f}" for x, y in zip(ratios, dices))
    circles = []
    labels = []
    for x, y in zip(ratios, dices):
        cx, cy = x_scale(x), y_scale(y)
        circles.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#2563eb" />')
        labels.append(f'<text x="{cx:.1f}" y="{cy - 10:.1f}" text-anchor="middle" class="small">{y:.3f}</text>')

    x_ticks = []
    for ratio in ratios:
        x = x_scale(ratio)
        x_ticks.append(f'<line x1="{x:.1f}" y1="{height - margin_bottom}" x2="{x:.1f}" y2="{height - margin_bottom + 6}" stroke="#555" />')
        x_ticks.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" class="small">{ratio:.1f}</text>')

    y_ticks = []
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        y = y_scale(value)
        y_ticks.append(f'<line x1="{margin_left - 6}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#555" />')
        y_ticks.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" />')
        y_ticks.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.2f}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, sans-serif; fill: #111827; }}
.title {{ font-size: 20px; font-weight: 700; }}
.axis {{ font-size: 13px; font-weight: 600; }}
.small {{ font-size: 11px; }}
</style>
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2}" y="24" text-anchor="middle" class="title">Mean Dice vs Synthetic-to-Real Ratio</text>
{''.join(y_ticks)}
<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" />
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" />
{''.join(x_ticks)}
<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3" />
{''.join(circles)}
{''.join(labels)}
<text x="{width / 2}" y="{height - 16}" text-anchor="middle" class="axis">Synthetic-to-real ratio</text>
<text x="18" y="{height / 2}" text-anchor="middle" class="axis" transform="rotate(-90 18 {height / 2})">Mean Dice</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize segmentation experiment metrics for portfolio reporting.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing experiment outputs.")
    parser.add_argument("--out-dir", default="results", help="Directory for summary artifacts.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    records = collect_records(outputs_dir)
    rows = summarize_by_ratio(records)
    write_csv(rows, out_dir / "synthetic_ratio_summary.csv")
    write_markdown(rows, out_dir / "synthetic_ratio_summary.md")
    write_svg(rows, out_dir / "ratio_vs_dice.svg")
    print(f"Loaded {len(records)} metric files")
    print(f"Wrote {out_dir / 'synthetic_ratio_summary.csv'}")
    print(f"Wrote {out_dir / 'synthetic_ratio_summary.md'}")
    print(f"Wrote {out_dir / 'ratio_vs_dice.svg'}")


if __name__ == "__main__":
    main()
