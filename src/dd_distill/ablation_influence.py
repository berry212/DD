from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _compact_hash(config: dict[str, Any], ignore_keys: set[str]) -> str:
    filtered = {k: config[k] for k in sorted(config.keys()) if k not in ignore_keys}
    payload = json.dumps(filtered, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _diff_config(
    without_cfg: dict[str, Any],
    with_cfg: dict[str, Any],
    allowed_changed_keys: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    changed: dict[str, dict[str, Any]] = {}
    for key in sorted(set(without_cfg.keys()) | set(with_cfg.keys())):
        left = without_cfg.get(key)
        right = with_cfg.get(key)
        if left != right:
            changed[key] = {
                "without_influence": left,
                "with_influence": right,
            }

    unexpected = {k: v for k, v in changed.items() if k not in allowed_changed_keys}
    return changed, unexpected


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_metrics(summary: dict[str, Any], keys: list[str]) -> dict[str, float | None]:
    return {key: _to_float_or_none(summary.get(key)) for key in keys}


def _metric_delta(
    without_metrics: dict[str, float | None],
    with_metrics: dict[str, float | None],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in sorted(set(without_metrics.keys()) | set(with_metrics.keys())):
        left = without_metrics.get(key)
        right = with_metrics.get(key)
        if left is None or right is None:
            out[key] = None
        else:
            out[key] = right - left
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "nan"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    fair = bool(report["fairness"]["is_fair"])
    status = "PASS" if fair else "FAIL"

    without_student = report["student_metrics"]["without_influence"]
    with_student = report["student_metrics"]["with_influence"]
    student_delta = report["student_metrics"]["delta"]

    without_distill = report["distill_metrics"]["without_influence"]
    with_distill = report["distill_metrics"]["with_influence"]
    distill_delta = report["distill_metrics"]["delta"]

    lines: list[str] = []
    lines.append("# Influence Ablation Report")
    lines.append("")
    lines.append(f"- dataset: {report.get('dataset', '')}")
    lines.append(f"- fairness: {status}")
    lines.append(f"- distill_config_hash_without: {report['fairness']['distill_config_hash_without']}")
    lines.append(f"- distill_config_hash_with: {report['fairness']['distill_config_hash_with']}")
    lines.append(f"- student_config_hash_without: {report['fairness']['student_config_hash_without']}")
    lines.append(f"- student_config_hash_with: {report['fairness']['student_config_hash_with']}")
    lines.append("")

    lines.append("## Student Metrics")
    lines.append("")
    lines.append("| metric | without_influence | with_influence | delta(with-without) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in sorted(student_delta.keys()):
        lines.append(
            f"| {key} | {_fmt(without_student.get(key))} | {_fmt(with_student.get(key))} | {_fmt(student_delta.get(key))} |"
        )
    lines.append("")

    lines.append("## Distill Metrics")
    lines.append("")
    lines.append("| metric | without_influence | with_influence | delta(with-without) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in sorted(distill_delta.keys()):
        lines.append(
            f"| {key} | {_fmt(without_distill.get(key))} | {_fmt(with_distill.get(key))} | {_fmt(distill_delta.get(key))} |"
        )
    lines.append("")

    unexpected_distill = report["fairness"]["distill_unexpected_changes"]
    unexpected_student = report["fairness"]["student_unexpected_changes"]
    if unexpected_distill or unexpected_student:
        lines.append("## Fairness Violations")
        lines.append("")
        if unexpected_distill:
            lines.append("### Distill Unexpected Changes")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(unexpected_distill, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
        if unexpected_student:
            lines.append("### Student Unexpected Changes")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(unexpected_student, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    without_distill_dir = Path(args.without_influence_distill_dir).expanduser().resolve()
    with_distill_dir = Path(args.with_influence_distill_dir).expanduser().resolve()
    without_student_dir = Path(args.without_influence_student_dir).expanduser().resolve()
    with_student_dir = Path(args.with_influence_student_dir).expanduser().resolve()

    without_distill_summary = _load_json(without_distill_dir / "summary.json")
    with_distill_summary = _load_json(with_distill_dir / "summary.json")
    without_distill_cfg = _load_json(without_distill_dir / "run_config.json")
    with_distill_cfg = _load_json(with_distill_dir / "run_config.json")

    without_student_summary = _load_json(without_student_dir / "summary.json")
    with_student_summary = _load_json(with_student_dir / "summary.json")
    without_student_cfg = _load_json(without_student_dir / "run_config.json")
    with_student_cfg = _load_json(with_student_dir / "run_config.json")

    distill_allowed_changed_keys = {
        "influence_mode",
        "output_dir",
        "lora_path",
    }
    student_allowed_changed_keys = {
        "output_dir",
        "distilled_data",
    }

    distill_changed, distill_unexpected = _diff_config(
        without_cfg=without_distill_cfg,
        with_cfg=with_distill_cfg,
        allowed_changed_keys=distill_allowed_changed_keys,
    )
    student_changed, student_unexpected = _diff_config(
        without_cfg=without_student_cfg,
        with_cfg=with_student_cfg,
        allowed_changed_keys=student_allowed_changed_keys,
    )

    distill_metric_keys = [
        "num_distilled",
        "center_influence_mean",
        "center_influence_min",
        "center_influence_max",
    ]
    student_metric_keys = [
        "best_val_acc",
        "test_acc_at_best_val",
        "final_test_acc",
        "macro_f1",
        "auc_macro",
    ]

    distill_without_metrics = _collect_metrics(without_distill_summary, distill_metric_keys)
    distill_with_metrics = _collect_metrics(with_distill_summary, distill_metric_keys)
    student_without_metrics = _collect_metrics(without_student_summary, student_metric_keys)
    student_with_metrics = _collect_metrics(with_student_summary, student_metric_keys)

    dataset = str(with_distill_summary.get("dataset") or without_distill_summary.get("dataset") or "")
    fairness_ok = (len(distill_unexpected) == 0) and (len(student_unexpected) == 0)

    report = {
        "dataset": dataset,
        "paths": {
            "without_influence_distill_dir": str(without_distill_dir),
            "with_influence_distill_dir": str(with_distill_dir),
            "without_influence_student_dir": str(without_student_dir),
            "with_influence_student_dir": str(with_student_dir),
        },
        "fairness": {
            "is_fair": fairness_ok,
            "distill_allowed_changed_keys": sorted(distill_allowed_changed_keys),
            "student_allowed_changed_keys": sorted(student_allowed_changed_keys),
            "distill_changed_keys": sorted(distill_changed.keys()),
            "student_changed_keys": sorted(student_changed.keys()),
            "distill_unexpected_changes": distill_unexpected,
            "student_unexpected_changes": student_unexpected,
            "distill_config_hash_without": _compact_hash(without_distill_cfg, ignore_keys=distill_allowed_changed_keys),
            "distill_config_hash_with": _compact_hash(with_distill_cfg, ignore_keys=distill_allowed_changed_keys),
            "student_config_hash_without": _compact_hash(without_student_cfg, ignore_keys=student_allowed_changed_keys),
            "student_config_hash_with": _compact_hash(with_student_cfg, ignore_keys=student_allowed_changed_keys),
        },
        "distill_metrics": {
            "without_influence": distill_without_metrics,
            "with_influence": distill_with_metrics,
            "delta": _metric_delta(distill_without_metrics, distill_with_metrics),
        },
        "student_metrics": {
            "without_influence": student_without_metrics,
            "with_influence": student_with_metrics,
            "delta": _metric_delta(student_without_metrics, student_with_metrics),
        },
    }

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else with_student_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "influence_ablation_report.json"
    md_path = output_dir / "influence_ablation_report.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    _write_markdown_report(md_path, report)

    print(f"[Ablation] Report JSON: {json_path}")
    print(f"[Ablation] Report Markdown: {md_path}")
    print(f"[Ablation] Fairness: {'PASS' if fairness_ok else 'FAIL'}")

    if args.fail_on_unfair and not fairness_ok:
        raise SystemExit(2)

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare influence ablation runs (with image-grad influence vs without influence), "
            "validate fairness, and export a report."
        )
    )
    parser.add_argument("--without-influence-distill-dir", type=str, required=True)
    parser.add_argument("--with-influence-distill-dir", type=str, required=True)
    parser.add_argument("--without-influence-student-dir", type=str, required=True)
    parser.add_argument("--with-influence-student-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--fail-on-unfair", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
