import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from ..core.evaluation import EvaluationError, SessionResult, Summary, evaluate_session, summarise
from ..core.session import SessionFormatError, load_session

HIT_SIZES_DEG = tuple(range(1, 11))


def format_summary(summary: Summary, indent: str = "  ") -> list[str]:
    s90 = summary.size_for_hit_rate(0.9)
    curve = "  ".join(
        f"{d}\u00b0:{100 * r:.0f}%"
        for d, r in zip(HIT_SIZES_DEG, summary.hit_rates(HIT_SIZES_DEG), strict=True)
    )
    return [
        f"{indent}accuracy   mean {summary.mean_offset_deg:.2f}\u00b0   "
        f"median {summary.median_offset_deg:.2f}\u00b0   ({summary.n_val_dots} validation dots)",
        f"{indent}precision  RMS sample-to-sample {summary.precision_rms_deg:.2f}\u00b0   "
        f"SD {summary.precision_sd_deg:.2f}\u00b0",
        f"{indent}data loss  no face {summary.no_face_pct:.1f}%   "
        f"not gaze-valid {summary.not_valid_pct:.1f}%",
        f"{indent}90% hit-rate target: {s90:.1f}\u00b0 = {summary.size_cm(s90):.1f} cm "
        f"at {summary.mean_distance_mm / 10:.0f} cm",
        f"{indent}hit-rate by target size: {curve}",
    ]


def format_result(label: str, r: SessionResult) -> str:
    c = r.conditions
    tags = f"participant={c.participant}" + (f"  note: {c.notes}" if c.notes else "")
    lines = [
        f"{label}",
        f"  {tags}",
        f"  model={r.mapper_name} (alpha={r.fitted.alpha_:.3g}) features={r.feature_set} "
        f"cal-points={r.calibration_points}   "
        f"calibration dots {r.n_cal_used}/{r.n_cal_total}, "
        f"validation dots {r.n_val_used}/{r.n_val_total}",
        *format_summary(r.summary),
    ]
    return "\n".join(lines)


def run_eval(
    files: Sequence[Path], mapper: str = "ridge", feature_set: str = "iris", calibration: int = 9
) -> int:
    results: list[SessionResult] = []
    failed = False
    for path in files:
        try:
            result = evaluate_session(
                load_session(path),
                mapper=mapper,
                feature_set=feature_set,
                calibration=calibration,
            )
        except (EvaluationError, SessionFormatError, OSError, ValueError) as exc:
            print(f"{path}: error: {exc}", file=sys.stderr)
            failed = True
            continue
        results.append(result)
        print(format_result(str(path), result) + "\n")

    by_participant: dict[str, list[SessionResult]] = defaultdict(list)
    for r in results:
        by_participant[r.conditions.participant].append(r)
    if len(results) > 1:
        for name, group in by_participant.items():
            print(f"== participant {name}: {len(group)} session(s) pooled ==")
            print("\n".join(format_summary(summarise(group))) + "\n")
        if len(by_participant) > 1:
            print(f"== ALL: {len(results)} sessions pooled ==")
            print("\n".join(format_summary(summarise(results))))
    return 1 if failed else 0
