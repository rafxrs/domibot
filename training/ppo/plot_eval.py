"""Plot eval win rates against iteration from one or more training logs.

    python -m training.ppo.plot_eval logs/domibot2/runs/a_to_2.1/01_iter0001-0400.log \\
        logs/domibot2/runs/a_to_2.1/02_iter0401-2400.log --smooth 8 --vline "8000:2.1"

Several logs (a run resumed in stages) merge into one series per eval label.
Each series gets a least-squares trendline unless `--no-trend`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

ITER_RE = re.compile(r"^iter (\d+)/\d+")
EVAL_RE = re.compile(r"^\s*eval vs ([^:]+):\s*(\d+)/(\d+) wins,\s*(\d+) ties")

Series = dict[str, tuple[list[int], list[float]]]


def parse_eval_logs(paths: list[str | Path]) -> Series:
    """`{label: (iterations, win rates in %)}`, merged across `paths` and sorted by iteration."""
    series: Series = {}
    for path in paths:
        current = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if m := ITER_RE.match(line):
                    current = int(m.group(1))
                elif (m := EVAL_RE.match(line)) and current is not None:
                    iters, rates = series.setdefault(m.group(1), ([], []))
                    iters.append(current)
                    rates.append(100.0 * int(m.group(2)) / int(m.group(3)))
    for label, (iters, rates) in series.items():
        order = sorted(range(len(iters)), key=lambda i: iters[i])
        series[label] = ([iters[i] for i in order], [rates[i] for i in order])
    return series


def linear_trend(iters: list[int], rates: list[float]) -> tuple[float, float, float]:
    """Least-squares (slope per iteration, intercept, r2)."""
    x, y = np.asarray(iters, dtype=np.float64), np.asarray(rates, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((y - (slope * x + intercept)) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), r_squared


def plot(series: Series, title: str, out: Path, trend: bool = True, smooth: int = 1,
         vlines: list[tuple[int, str]] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (iters, rates) in series.items():
        if smooth > 1 and len(iters) >= smooth:
            raw, = ax.plot(iters, rates, linewidth=0.8, alpha=0.2)
            smoothed = np.convolve(rates, np.ones(smooth) / smooth, mode="valid")  # trailing mean
            line, = ax.plot(iters[smooth - 1:], smoothed, linewidth=2.2, color=raw.get_color(), label=f"vs {label}")
        else:
            line, = ax.plot(iters, rates, marker="o", markersize=3, linewidth=1.2, alpha=0.55, label=f"vs {label}")
        if trend and len(iters) >= 2:
            slope, intercept, r2 = linear_trend(iters, rates)
            ax.plot([iters[0], iters[-1]], [slope * x + intercept for x in (iters[0], iters[-1])], linewidth=2.2,
                    color=line.get_color(), label=f"{label} trend ({slope * 100:+.2f}%/100 iter, r2={r2:.2f})")
    for x, text in vlines or []:
        ax.axvline(x, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.text(x, 3, f" {text}", rotation=90, va="bottom", fontsize=9, alpha=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("win rate (%)")
    ax.set_ylim(-2, 102)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.3))  # below the curves, clear of the --vline labels
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_files", nargs="+")
    parser.add_argument("--out", type=str, default=None, help="default: next to the first log")
    parser.add_argument("--no-trend", action="store_true")
    parser.add_argument("--smooth", type=int, default=1, help="plot a trailing mean over this many eval points")
    parser.add_argument("--vline", action="append", default=[], metavar="ITER:LABEL",
                        help="a labeled vertical line, e.g. '8000:2.1' (repeatable)")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    series = parse_eval_logs(args.log_files)
    if not series:
        raise SystemExit(f"no 'eval vs ...' lines in {', '.join(args.log_files)}")
    for label, (iters, rates) in series.items():
        line = (f"{label}: {len(iters)} eval points, iter {iters[0]}-{iters[-1]}, "
                f"win rate {min(rates):.1f}-{max(rates):.1f}% (latest {rates[-1]:.1f}%)")
        if not args.no_trend and len(iters) >= 2:
            slope, _, r2 = linear_trend(iters, rates)
            line += f"  |  trend: {slope * 100:+.2f} pp/100 iter, r2={r2:.2f}"
        print(line)
    paths = [Path(p) for p in args.log_files]
    stem = paths[0].stem if len(paths) == 1 else "combined_" + "+".join(p.stem for p in paths)
    out = Path(args.out) if args.out else paths[0].with_name(stem + "_eval.png")
    vlines = [(int(v.split(":", 1)[0]), v.split(":", 1)[1] if ":" in v else "") for v in args.vline]
    plot(series, args.title or f"eval win rate: {', '.join(Path(p).name for p in args.log_files)}", out,
         trend=not args.no_trend, smooth=args.smooth, vlines=vlines)


if __name__ == "__main__":
    main()
