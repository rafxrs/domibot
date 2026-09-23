"""Parse one or more domibot 2 (PPO) training logs and plot eval win rates
vs. iteration -- BigMoney, BigMoney+terminal, and (if present) a reference
checkpoint, e.g. `domibot_v4.4.pt`.

    python -m training.ppo.plot_eval logs/domibot2/domibot2_stage1_run2.log
    python -m training.ppo.plot_eval logs/domibot2/domibot2_stage1_run2.log --out eval.png

Pass multiple logs to see one continuous history across resumed runs
(e.g. run1 = iterations 1-400, run2 resumed from run1's checkpoint =
401-2400) -- each series is merged by label and sorted by iteration, so
it reads as one training run even though it was launched in stages:

    python -m training.ppo.plot_eval logs/domibot2/domibot2_stage1_run1.log \\
        logs/domibot2/domibot2_stage1_run2.log

Reads the exact lines `training/ppo/train.py` prints (see `main`'s
`  eval vs ...:` prints): each eval block is preceded by an `iter N/M`
line, so parsing just tracks the most recent iteration number and
attaches every `eval vs <label>: W/G wins, T ties` line under it.
Handles any number of distinct eval labels (not hardcoded to exactly
three), so this works unchanged whether or not `--eval-reference-checkpoint`
was set, and for whatever its name is.

Each series also gets a least-squares linear trendline (see
`linear_trend`) overlaid in the same color, with its slope (percentage
points per 100 iterations) and r2 (how much of the point-to-point
variance the line actually explains -- near 0 for a flat/noisy series,
regardless of the slope's sign) printed alongside the raw range. Pass
`--no-trend` to skip it.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

ITER_RE = re.compile(r"^iter (\d+)/\d+")
EVAL_RE = re.compile(r"^\s*eval vs ([^:]+):\s*(\d+)/(\d+) wins,\s*(\d+) ties")


def parse_eval_log(path: str | Path) -> dict[str, tuple[list[int], list[float]]]:
    """Returns `{label: (iterations, win_rates)}`, one series per distinct
    "eval vs <label>" line encountered, in the order each first appears."""
    series: dict[str, tuple[list[int], list[float]]] = {}
    current_iter: int | None = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ITER_RE.match(line)
            if m:
                current_iter = int(m.group(1))
                continue
            m = EVAL_RE.match(line)
            if m and current_iter is not None:
                label, wins, games, _ties = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
                iters, rates = series.setdefault(label, ([], []))
                iters.append(current_iter)
                rates.append(100.0 * wins / games)
    return series


def merge_series(
    per_file: list[dict[str, tuple[list[int], list[float]]]],
) -> dict[str, tuple[list[int], list[float]]]:
    """Combines several `parse_eval_log` results into one series per label
    (concatenated in file order, then sorted by iteration) -- e.g. a run
    resumed partway through from an earlier run's checkpoint, so the two
    logs' iteration numbers are naturally contiguous but live in separate
    files. A single-file input is a no-op pass-through."""
    merged: dict[str, tuple[list[int], list[float]]] = {}
    for series in per_file:
        for label, (iters, rates) in series.items():
            all_iters, all_rates = merged.setdefault(label, ([], []))
            all_iters.extend(iters)
            all_rates.extend(rates)
    for label, (iters, rates) in merged.items():
        order = sorted(range(len(iters)), key=lambda i: iters[i])
        merged[label] = ([iters[i] for i in order], [rates[i] for i in order])
    return merged


def linear_trend(iters: list[int], rates: list[float]) -> tuple[float, float, float]:
    """Least-squares line through (iters, rates). Returns (slope, intercept,
    r_squared) -- slope is in win-rate-percentage-points per iteration; r2
    is how much of the point-to-point variance the line actually explains
    (near 0 for a flat/noisy series, near 1 for a clean trend), so it's
    worth reporting alongside the line, not just the line itself, since a
    noisy series can have a "trend" that explains almost none of what's
    actually happening iteration to iteration."""
    x = np.asarray(iters, dtype=np.float64)
    y = np.asarray(rates, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), r_squared


def plot(series: dict[str, tuple[list[int], list[float]]], title: str, out: Path, trend: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (iters, rates) in series.items():
        line, = ax.plot(iters, rates, marker="o", markersize=3, linewidth=1.2, alpha=0.55, label=label)
        if trend and len(iters) >= 2:
            slope, intercept, r_squared = linear_trend(iters, rates)
            x_fit = [iters[0], iters[-1]]
            y_fit = [slope * x + intercept for x in x_fit]
            ax.plot(x_fit, y_fit, linewidth=2.2, color=line.get_color(),
                     label=f"{label} trend ({slope * 100:+.2f}%/100 iter, r2={r_squared:.2f})")

    ax.set_xlabel("iteration")
    ax.set_ylabel("win rate (%)")
    ax.set_ylim(-2, 102)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_files", nargs="+",
                         help="one or more domibot2 training logs, e.g. logs/domibot2/domibot2_stage1_run1.log "
                              "logs/domibot2/domibot2_stage1_run2.log -- multiple logs are merged into one "
                              "continuous history per eval label, sorted by iteration")
    parser.add_argument("--out", type=str, default=None,
                         help="output image path (default: <first log's stem>_eval.png, or "
                              "combined_<stem1>+<stem2>+..._eval.png for multiple logs, next to the first log)")
    parser.add_argument("--no-trend", action="store_true",
                         help="skip the least-squares trendline (see linear_trend) overlaid on each series")
    args = parser.parse_args()

    log_paths = [Path(p) for p in args.log_files]
    series = merge_series([parse_eval_log(p) for p in log_paths])
    if not series:
        raise SystemExit(f"no 'eval vs ...' lines found in {', '.join(str(p) for p in log_paths)}")

    for label, (iters, rates) in series.items():
        line = (f"{label}: {len(iters)} eval points, iter {iters[0]}-{iters[-1]}, "
                f"win rate {min(rates):.1f}-{max(rates):.1f}% (latest {rates[-1]:.1f}%)")
        if not args.no_trend and len(iters) >= 2:
            slope, _intercept, r_squared = linear_trend(iters, rates)
            line += f"  |  trend: {slope * 100:+.2f} pp/100 iter, r2={r_squared:.2f}"
        print(line)

    if args.out:
        out = Path(args.out)
    elif len(log_paths) == 1:
        out = log_paths[0].with_name(log_paths[0].stem + "_eval.png")
    else:
        out = log_paths[0].with_name("combined_" + "+".join(p.stem for p in log_paths) + "_eval.png")
    title = f"domibot2 eval win rate -- {'+'.join(p.name for p in log_paths)}"
    plot(series, title=title, out=out, trend=not args.no_trend)


if __name__ == "__main__":
    main()
