"""Render README figures from committed measurements, never synthetic timings.

Usage (from the repository root):
    python -m pip install matplotlib
    python docs/media/render_charts.py
    python docs/media/render_charts.py --data path/to/results.json

Matplotlib is only needed to rebuild documentation, not to run Fluxion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MEDIA = Path(__file__).resolve().parent
INK = "#142b49"
MUTED = "#63758b"
BLUE = "#2563eb"
TEAL = "#078a81"
BACKGROUND = "#f4f7fb"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#d6deeb",
            "axes.facecolor": "white",
            "figure.facecolor": BACKGROUND,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "svg.fonttype": "none",
            "svg.hashsalt": "fluxion-readme-measurements",
        }
    )


def canvas(title: str, subtitle: str) -> tuple:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.76, bottom=0.20, wspace=0.28)
    fig.text(0.065, 0.93, title, fontsize=23, weight="bold")
    fig.text(0.065, 0.875, subtitle, fontsize=12, color=MUTED)
    for ax in axes:
        ax.grid(axis="y", color="#e6ebf3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=8)
    return fig, axes


def save(fig, name: str) -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            MEDIA / f"{name}.{extension}",
            dpi=120,
            facecolor=BACKGROUND,
            metadata={"Creator": "Fluxion render_charts.py"},
        )
    plt.close(fig)


def training_chart(data: dict) -> None:
    training = data["training"]
    config = training["config"]
    recorded = data["recorded_at_utc"][:10]
    steps = np.asarray(training["steps"])
    loss = np.asarray(training["loss"])
    if not np.all(np.isfinite(loss)) or np.any(loss <= 0):
        raise ValueError("The logarithmic loss figure requires positive finite losses.")
    fig, (ax, predictions) = canvas(
        "A neural network learns using Fluxion's own gradients",
        "Seeded teaching example: Linear(1, 8) → ReLU → Linear(8, 1), trained on y = 2x.",
    )
    ax.semilogy(steps, loss, color=BLUE, linewidth=2.6)
    ax.set_title("Loss falls during training", loc="left", pad=18)
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel("Mean squared error · logarithmic scale")
    ax.set_xlim(0, max(steps))
    ax.text(
        0.98, 0.96,
        f"Start  {loss[0]:.3g}\nFinish  {loss[-1]:.3g}",
        transform=ax.transAxes, va="top", ha="right", color=BLUE,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#edf3ff", "edgecolor": "none"},
    )

    train = training["train"]
    unseen = training["unseen"]
    x_all = np.asarray(train["x"] + unseen["x"])
    target_all = np.asarray(train["target"] + unseen["target"])
    order = np.argsort(x_all)
    predictions.plot(
        x_all[order], target_all[order], "--", color=MUTED, linewidth=1.7,
        label="Target: y = 2x",
    )
    predictions.scatter(
        train["x"], train["prediction"], s=65, color=BLUE,
        edgecolor="white", linewidth=1.2, zorder=3, label="Training inputs",
    )
    predictions.scatter(
        unseen["x"], unseen["prediction"], s=75, marker="D", color=TEAL,
        edgecolor="white", linewidth=1.2, zorder=3, label="Unseen inputs",
    )
    predictions.set_title("Predictions after training", loc="left", pad=18)
    predictions.set_xlabel("Input x")
    predictions.set_ylabel("Predicted value")
    predictions.set_xlim(0, max(x_all) + 1)
    predictions.set_ylim(0, max(target_all) * 1.12)
    predictions.legend(loc="upper left", frameon=False, fontsize=10)
    fig.text(
        0.065, 0.09,
        "Forward pass → MSE loss → reverse-mode backpropagation → SGD update. No PyTorch execution in this example.",
        fontsize=11, color=MUTED,
    )
    fig.text(
        0.065, 0.045,
        f"Recorded {recorded} · seed {config['seed']} · {config['updates']:,} updates · learning rate {config['learning_rate']} · data and reproduction in docs/benchmarks/.",
        fontsize=10, color=MUTED,
    )
    save(fig, "training-progress")


def scaling_chart(data: dict) -> None:
    rows = data["scaling"]["rows"]
    config = data["scaling"]["config"]
    platform = data["environment"]["platform"]
    host = "macOS" if platform.startswith("Darwin") else platform.split("-", 1)[0]
    recorded = data["recorded_at_utc"][:10]
    if not rows:
        raise ValueError("No scaling measurements found.")
    backends = list(dict.fromkeys(row["backend"] for row in rows))
    trials = len(rows[0]["trial_ms"])
    fig, (time_axis, throughput_axis) = canvas(
        "Longer context changes the cost of a GPT training step",
        "Measured CPU execution: forward pass + cross-entropy loss + backward pass + Adam update.",
    )
    palette = [BLUE, TEAL, "#9b5bb5"]
    for index, backend in enumerate(backends):
        selected = sorted(
            (row for row in rows if row["backend"] == backend),
            key=lambda row: row["sequence_length"],
        )
        x = np.array([row["sequence_length"] for row in selected])
        trial_values = np.array([row["trial_ms"] for row in selected])
        if not np.all(np.isfinite(trial_values)) or np.any(trial_values <= 0):
            raise ValueError("Timing values must be finite and positive.")
        median = np.median(trial_values, axis=1)
        q1, q3 = np.quantile(trial_values, [0.25, 0.75], axis=1)
        token_count = np.array([row["batch_size"] * row["sequence_length"] for row in selected])
        color = palette[index % len(palette)]
        label = {
            "numpy": "Fluxion · NumPy",
            "native": "Fluxion · C++ Linear",
            "native_cblas": "Fluxion · C++ Linear",
            "native_accelerate": "Fluxion · C++ Linear",
        }.get(backend, backend)
        time_axis.plot(x, median, "o-", color=color, linewidth=2.4, markersize=6, label=label)
        time_axis.fill_between(x, q1, q3, color=color, alpha=0.12)
        time_axis.scatter(np.repeat(x, trials), trial_values.flatten(), color=color, alpha=0.22, s=16)
        throughput_axis.plot(
            x, token_count * 1000 / median, "o-", color=color,
            linewidth=2.4, markersize=6, label=label,
        )
        throughput_axis.fill_between(
            x, token_count * 1000 / q3, token_count * 1000 / q1,
            color=color, alpha=0.12,
        )

    for ax in (time_axis, throughput_axis):
        ax.set_xscale("log", base=2)
        lengths = sorted({row["sequence_length"] for row in rows})
        ax.set_xticks(lengths, [str(length) for length in lengths])
        ax.set_xlabel("Sequence length · tokens per example")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", frameon=False, fontsize=10)
    time_axis.set_title("Training-step time", loc="left", pad=18)
    time_axis.set_ylabel("Milliseconds per step · lower is faster")
    throughput_axis.set_title("Training throughput", loc="left", pad=18)
    throughput_axis.set_ylabel("Tokens per second · higher is faster")
    throughput_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    fig.text(
        0.065, 0.09,
        f"Lines: median of {trials} trial means. Shading: interquartile range. Time-panel dots: individual trial means.",
        fontsize=11, color=MUTED,
    )
    fig.text(
        0.065, 0.045,
        f"Recorded {recorded} · {host} CPU · {config['dtype']} · one BLAS thread · local workload, not production capacity or GPU performance.",
        fontsize=10, color=MUTED,
    )
    save(fig, "sequence-scaling")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "docs/benchmarks/2026-10-04/results.json",
    )
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    style()
    training_chart(data)
    scaling_chart(data)
    print("Rendered training-progress and sequence-scaling as PNG and editable SVG.")


if __name__ == "__main__":
    main()
