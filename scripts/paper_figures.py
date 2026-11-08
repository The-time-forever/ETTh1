#!/usr/bin/env python
r"""Generate publication figures for the ETTh1 paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#999999",
    "black": "#222222",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#F5F5F5", ec="#444444"):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=8)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=8,
            linewidth=0.8, color="#444444", shrinkA=2, shrinkB=2,
        )
    )


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.58
    xs = [0.09, 0.28, 0.47, 0.66, 0.85]
    texts = [
        "OT window\n96 h",
        "Local level\n$\\ell_t$ = mean(24 h)",
        "Center\n$X'=X-\\ell_t$",
        "Ridge\ndirect multi-step",
        "Restore\n$\\hat Y=\\hat Y'+\\ell_t$",
    ]
    for x, t in zip(xs, texts):
        box(ax, x, y, 0.155, 0.34, t, fc="#F7F7F7")
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 0.0775, y, xs[i + 1] - 0.0775, y)
    ax.text(
        0.5, 0.16,
        "Translation equivariance:  $X,Y \\rightarrow X+c,\\,Y+c$  implies  $\\hat Y \\rightarrow \\hat Y+c$",
        ha="center", va="center", fontsize=8, color=OKABE["black"],
    )
    ax.text(0.5, 0.90, "Anchored linear forecaster", ha="center", va="center", fontsize=9.5, weight="bold")
    save(fig, "fig1_pipeline")


def fig_main():
    models = [
        ("Anchored Ridge", 1.6324, OKABE["red"]),
        ("Anchored DLinear", 1.6350, OKABE["orange"]),
        ("PatchTST (small)", 1.6438, OKABE["purple"]),
        ("Anchored MLP", 1.6633, OKABE["blue"]),
        ("Anchored GRU", 1.6707, OKABE["sky"]),
        ("No-Attention", 1.6843, OKABE["green"]),
        ("SegRNN (last)", 1.6937, OKABE["yellow"]),
        ("Anchored LSTM", 1.7586, OKABE["grey"]),
    ]
    models = sorted(models, key=lambda x: x[1])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(models))
    ax.barh(y, [m[1] for m in models], color=[m[2] for m in models], height=0.62)
    for i, (_, v, _) in enumerate(models):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=7.2)
    ax.axvline(1.8214, color=OKABE["grey"], ls="--", lw=0.8)
    ax.text(1.826, len(models) - 0.35, "B2b", color=OKABE["grey"], fontsize=7.2)
    ax.set_yticks(y, [m[0] for m in models])
    ax.set_xlim(1.55, 1.87)
    ax.set_xlabel("Test MAE (°C), lower is better")
    ax.set_title("(a) Anchored models, S 96→96")

    ax = axes[1]
    families = ["Ridge", "MLP", "GRU", "LSTM", "DLinear", "PatchTST"]
    plain = [0.7805, 2.6776, 3.3314, 3.3388, 1.2641, 2.3305]
    anchored = [0.1051, -0.0337, -0.0298, -0.2022, 0.0932, 0.0842]
    x = np.arange(len(families))
    w = 0.36
    ax.bar(x - w / 2, plain, w, label="plain", color=OKABE["grey"])
    ax.bar(x + w / 2, anchored, w, label="anchored", color=OKABE["red"])
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x, families, rotation=25, ha="right")
    ax.set_ylabel("Bias (°C)")
    ax.set_title("(b) Anchoring removes level bias")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig2_main")


def fig_horizon():
    horizons = [96, 192, 336, 720]
    ridge = [1.6324, 1.9140, 2.1288, 2.1162]
    b1 = [1.8654, 2.1627, 2.4336, 2.6007]
    b2b = [1.8214, 2.0632, 2.2751, 2.3952]
    skill = [10.37, 7.23, 6.43, 11.65]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    x = np.arange(len(horizons))
    ax.plot(x, ridge, "o-", color=OKABE["red"], lw=1.5, ms=4, label="Anchored Ridge")
    ax.plot(x, b2b, "s--", color=OKABE["green"], lw=1.0, ms=3.5, label="B2b (96 h mean)")
    ax.plot(x, b1, "^:", color=OKABE["grey"], lw=1.0, ms=3.5, label="B1 (last value)")
    ax.set_xticks(x, [str(h) for h in horizons])
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("(a) Error vs horizon")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar([str(h) for h in horizons], skill, color=OKABE["blue"], width=0.55)
    for i, v in enumerate(skill):
        ax.text(i, v + 0.25, f"{v:.1f}%", ha="center", fontsize=7.2)
    ax.set_ylim(0, 13.5)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE skill vs B2b (%)")
    ax.set_title("(b) Relative gain over local level")
    fig.tight_layout()
    save(fig, "fig3_horizon")


def fig_ablations():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3))

    ax = axes[0, 0]
    ax.bar(["S (OT)", "MS (7 ch.)"], [1.6327, 1.7953], color=[OKABE["blue"], OKABE["orange"]], width=0.55)
    for i, v in enumerate([1.6327, 1.7953]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.2)
    ax.set_ylim(1.5, 1.88)
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("(a) Load inputs: S vs MS")

    ax = axes[0, 1]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [1.6324, 1.6438], w, label="L=96", color=OKABE["blue"])
    ax.bar(x + w / 2, [1.6250, 1.7577], w, label="L=336", color=OKABE["red"])
    ax.set_xticks(x, ["Ridge", "PatchTST"])
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("(b) Longer history: L=96 vs 336")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    x = np.arange(2)
    ax.bar(x - w / 2, [1.6438, 1.6843], w, label="mean", color=OKABE["purple"])
    ax.bar(x + w / 2, [1.6003, 1.6596], w, label="ensemble", color=OKABE["orange"])
    ax.set_xticks(x, ["Attention", "Token mixing"])
    ax.set_ylim(1.55, 1.72)
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("(c) Attention ablation")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ax.bar(["Plain", "Anchored"], [1.8432, 1.6350], color=[OKABE["grey"], OKABE["red"]], width=0.55)
    for i, v in enumerate([1.8432, 1.6350]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.2)
    ax.set_ylim(1.55, 1.90)
    ax.set_ylabel("Test MAE (°C)")
    ax.set_title("(d) DLinear: plain vs anchored")

    fig.tight_layout()
    save(fig, "fig4_ablations")


if __name__ == "__main__":
    fig_pipeline()
    fig_main()
    fig_horizon()
    fig_ablations()
    print("figures written to", OUT)
