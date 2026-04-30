"""Generate Model9 architecture diagram (matplotlib).

Outputs:
    model9/results/architecture.png  (also .pdf)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.use14corefonts": True,
        "ps.useafm": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
    }
)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLOR_INPUT = "#E8F0FE"
COLOR_TEMPORAL = "#FCE8E6"
COLOR_SPATIAL = "#E6F4EA"
COLOR_FREQ = "#FEF7E0"
COLOR_FUSE = "#F3E8FD"
COLOR_HEAD = "#E8EAED"
COLOR_ANOM = "#FCE4EC"
EDGE = "#3C4043"
SEQUENCE_LENGTH = 30
NUM_NODES = 16
FREQ_BINS = (SEQUENCE_LENGTH // 2) + 1


def box(ax, x, y, w, h, text, face, fontsize=9, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=EDGE,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        zorder=3,
    )


def arrow(ax, x0, y0, x1, y1, color=EDGE):
    a = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color=color,
        zorder=1,
    )
    ax.add_patch(a)


def branch_title_box(ax, x, y, w, title, face):
    h = 0.62
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=EDGE,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        title,
        ha="center",
        va="center",
        fontsize=9.8,
        weight="bold",
        color="#202124",
        linespacing=0.95,
        zorder=3,
    )


def main(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.05)
    ax.set_axis_off()

    # === Input ===
    box(ax, 4.85, 9.30, 4.30, 0.55,
        f"Input  x  in  R^(B x T={SEQUENCE_LENGTH} x N={NUM_NODES})",
        COLOR_INPUT, fontsize=10.5, weight="bold")
    ax.text(
        7.0,
        9.05,
        "16 feature-as-nodes (15 features + sum_granted_prbs history)",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#5F6368",
    )

    # === Three branches ===
    # Layout: y_top = 8.20, y_bot = 3.20
    # Each branch is a vertical stack of boxes.

    # ---- Temporal Branch (left) ----
    branch_title_box(ax, 0.55, 8.28, 3.70, "Temporal Branch\n(BiLSTM)", COLOR_TEMPORAL)
    bx = 0.55
    bw = 3.70
    ty = [7.55, 6.85, 6.00, 5.30, 4.55, 3.80]
    box(ax, bx, ty[0], bw, 0.50, "per-node split:  (B*N, T, 1)", COLOR_TEMPORAL, fontsize=9)
    box(ax, bx, ty[1], bw, 0.50, "Linear(1 -> 16)  input_proj", COLOR_TEMPORAL, fontsize=9)
    box(ax, bx, ty[2], bw, 0.65, "BiLSTM\nhidden=64,  L=2,  dropout=0.2", COLOR_TEMPORAL, fontsize=9)
    box(ax, bx, ty[3], bw, 0.50, "take last time step  (B*N, 128)", COLOR_TEMPORAL, fontsize=9)
    box(ax, bx, ty[4], bw, 0.55, "Linear(128 -> 64) + ReLU + Dropout", COLOR_TEMPORAL, fontsize=9)
    box(ax, bx, ty[5], bw, 0.50, "reshape ->  (B, N, 64)", COLOR_TEMPORAL, fontsize=9.5, weight="bold")

    # ---- Spatial Branch (middle) ----
    branch_title_box(ax, 5.10, 8.28, 3.70, "Spatial Branch\n(GAT)", COLOR_SPATIAL)
    bx = 5.10
    bw = 3.70
    sy = [7.55, 6.85, 6.00, 5.30, 4.55, 3.80]
    box(ax, bx, sy[0], bw, 0.50, "transpose ->  (B, N, T)", COLOR_SPATIAL, fontsize=9)
    box(ax, bx, sy[1], bw, 0.50, f"Linear(T={SEQUENCE_LENGTH} -> 32)  input_proj", COLOR_SPATIAL, fontsize=9)
    box(ax, bx, sy[2], bw, 0.65,
        "GATLayer x 2,  heads=4\nadj = binary fully-connected + self-loop",
        COLOR_SPATIAL, fontsize=9)
    box(ax, bx, sy[3], bw, 0.50, "ELU between layers", COLOR_SPATIAL, fontsize=9)
    box(ax, bx, sy[4], bw, 0.55,
        "head merge: mean/mean or concat/mean\n(CLI configurable)",
        COLOR_SPATIAL, fontsize=8.5)
    box(ax, bx, sy[5], bw, 0.50, "(B, N, 64)", COLOR_SPATIAL, fontsize=9.5, weight="bold")

    # ---- Frequency Branch (right) ----
    branch_title_box(ax, 9.65, 8.28, 3.70, "Frequency Branch\n(FFT Dual-Transformer)", COLOR_FREQ)
    bx = 9.65
    bw = 3.70
    fy_top = 7.55
    box(ax, bx, fy_top, bw, 0.50, f"rfft over T  ->  {FREQ_BINS} freq bins", COLOR_FREQ, fontsize=9)
    # split mag/phase
    box(ax, 9.65, 6.75, 1.75, 0.55, "Magnitude  abs(.)", COLOR_FREQ, fontsize=9)
    box(ax, 11.60, 6.75, 1.75, 0.55, "Phase  angle(.)", COLOR_FREQ, fontsize=9)
    box(ax, 9.65, 5.75, 3.70, 0.85,
        "independent mag/phase FFTEncoder\neach: Linear(1->64)+PosEmb+TxEnc(L=2)",
        COLOR_FREQ, fontsize=8.5)
    box(ax, 9.65, 5.00, 3.70, 0.55, "freq readout: mean / cls / last  ->  (B*N, 64)", COLOR_FREQ, fontsize=8.5)
    box(ax, 9.65, 4.30, 3.70, 0.55, "fuse  Linear(128 -> 64) + ReLU", COLOR_FREQ, fontsize=9)
    box(ax, 9.65, 3.65, 3.70, 0.45, "(B, N, 64)", COLOR_FREQ, fontsize=9.5, weight="bold")

    # ----- Arrows from input → title boxes -----
    arrow(ax, 6.2, 9.30, 2.40, 8.90)   # to temporal title
    arrow(ax, 7.0, 9.30, 6.95, 8.90)   # to spatial title
    arrow(ax, 7.8, 9.30, 11.50, 8.90)  # to frequency title

    # ----- Title → first inner box -----
    arrow(ax, 2.40, 8.28, 2.40, 8.05)
    arrow(ax, 6.95, 8.28, 6.95, 8.05)
    arrow(ax, 11.50, 8.28, 11.50, 8.05)

    # ----- Internal arrows: Temporal -----
    cx_t = 2.40
    for y0, y1 in [(ty[0], ty[1] + 0.50), (ty[1], ty[2] + 0.65),
                   (ty[2], ty[3] + 0.50), (ty[3], ty[4] + 0.55),
                   (ty[4], ty[5] + 0.50)]:
        arrow(ax, cx_t, y0, cx_t, y1)

    # ----- Internal arrows: Spatial -----
    cx_s = 6.95
    for y0, y1 in [(sy[0], sy[1] + 0.50), (sy[1], sy[2] + 0.65),
                   (sy[2], sy[3] + 0.50), (sy[3], sy[4] + 0.55),
                   (sy[4], sy[5] + 0.50)]:
        arrow(ax, cx_s, y0, cx_s, y1)

    # ----- Internal arrows: Frequency -----
    # rfft -> mag / phase
    arrow(ax, 11.50, fy_top, 10.525, 7.30)
    arrow(ax, 11.50, fy_top, 12.475, 7.30)
    # mag/phase -> transformer block
    arrow(ax, 10.525, 6.75, 11.0, 6.60)
    arrow(ax, 12.475, 6.75, 12.0, 6.60)
    # transformer -> mean pool -> fuse -> output
    arrow(ax, 11.50, 5.75, 11.50, 5.55)
    arrow(ax, 11.50, 5.00, 11.50, 4.85)
    arrow(ax, 11.50, 4.30, 11.50, 4.10)

    # === Concat + Head ===
    box(ax, 4.5, 2.30, 5.0, 0.55,
        "Concat per node  ->  (B, N, 64+64+64 = 192)",
        COLOR_FUSE, fontsize=10, weight="bold")

    # arrows from each branch's final box → concat
    arrow(ax, 2.40, ty[5], 5.0, 2.85)
    arrow(ax, 6.95, sy[5], 6.95, 2.85)
    arrow(ax, 11.50, 3.65, 9.0, 2.85)

    # node_head — wider + 2 lines
    box(ax, 3.5, 1.45, 7.0, 0.65,
        "node_head\nLinear(192 -> 64) + ReLU + Dropout  ->  Linear(64 -> horizon)",
        COLOR_HEAD, fontsize=9.5)
    arrow(ax, 7.0, 2.30, 7.0, 2.10)

    # Readout (centered, below node_head)
    box(ax, 4.5, 0.55, 5.0, 0.55,
        "Readout over N nodes:  mean / attention / gated",
        COLOR_HEAD, fontsize=9.5)
    arrow(ax, 7.0, 1.45, 7.0, 1.10)

    # Output (right of readout)
    box(ax, 10.0, 0.55, 3.5, 0.55,
        "y_hat  in  R^(B x horizon)",
        COLOR_INPUT, fontsize=10.5, weight="bold")
    arrow(ax, 9.5, 0.82, 10.0, 0.82)

    # === Anomaly Head (post-hoc, left-bottom) ===
    box(ax, 0.30, 0.55, 3.8, 0.65,
        "Chebyshev anomaly  (post-hoc)\nval mu/sigma, test residual -> flags",
        COLOR_ANOM, fontsize=9.5)

    # residual flow from prediction to anomaly flagging (grey)
    arrow(ax, 10.0, 0.55, 4.10, 0.55, color="#9AA0A6")
    ax.text(
        7.0,
        0.40,
        "primary k=3, modes=signed/abs/both; flag_tensor=(num_k, mode, num_test)",
        fontsize=8,
        color="#5F6368",
        ha="center",
        va="center",
        zorder=3,
    )

    # Legend (bottom-center, below everything)
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_TEMPORAL, edgecolor=EDGE, label="Temporal (BiLSTM)"),
        mpatches.Patch(facecolor=COLOR_SPATIAL, edgecolor=EDGE, label="Spatial (GAT)"),
        mpatches.Patch(facecolor=COLOR_FREQ, edgecolor=EDGE, label="Frequency (FFT-Tx)"),
        mpatches.Patch(facecolor=COLOR_FUSE, edgecolor=EDGE, label="Fusion"),
        mpatches.Patch(facecolor=COLOR_HEAD, edgecolor=EDGE, label="Pred head / Readout"),
        mpatches.Patch(facecolor=COLOR_ANOM, edgecolor=EDGE, label="Anomaly (post-hoc)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        fontsize=8.5,
        frameon=True,
        framealpha=0.9,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "architecture.png"
    pdf_path = out_dir / "architecture.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
