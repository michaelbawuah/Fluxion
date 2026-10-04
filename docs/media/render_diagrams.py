"""Render source-backed README diagrams; no benchmark values are synthesized.

Run from any directory: python docs/media/render_diagrams.py
Requires Matplotlib (a documentation-only dependency).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
BG = "#f4f7fb"
INK = "#142b49"
MUTED = "#63758b"
BLUE = "#2563eb"
TEAL = "#078a81"
AMBER = "#966312"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def label(ax, x, y, text, size=17, color=INK, weight="normal", ha="left", va="top"):
    ax.text(x, y, text, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, linespacing=1.45)


def box(ax, x, y, w, h, title, body="", color=BLUE, fill="#ffffff", title_size=19):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=16",
                               linewidth=1.4, edgecolor="#d5dfeb", facecolor=fill))
    ax.plot([x + 20, x + 20], [y + 22, y + h - 22], color=color, lw=4,
            solid_capstyle="round")
    label(ax, x + 42, y + 21, title, title_size, color, "bold")
    if body:
        label(ax, x + 42, y + 58, body, 15, MUTED)


def arrow(ax, start, end, color=BLUE, style="-", rad=0, lw=2.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                connectionstyle=f"arc3,rad={rad}", color=color,
                                linewidth=lw, linestyle=style))


def export(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", facecolor=BG, dpi=150)
    plt.close(fig)


def architecture():
    fig, ax = canvas(1380, 1010)
    label(ax, 55, 38, "FLUXION", 15, BLUE, "bold")
    label(ax, 55, 74, "From arrays to a trainable Transformer", 31, INK, "bold")
    label(ax, 55, 126, "The graph and gradients stay in Fluxion. Execution choices are explicit.", 18, MUTED)

    label(ax, 70, 191, "MODEL & TRAINING", 14, MUTED, "bold")
    label(ax, 835, 191, "OPERATOR EXECUTION", 14, MUTED, "bold")
    box(ax, 60, 231, 635, 130, "Modules and GPT", "Embeddings → causal attention → residual blocks\nFinal normalization → vocabulary logits")
    box(ax, 60, 392, 635, 145, "Tensor operations + dynamic DAG", "NumPy data · parent tensors · local backward rules\nBroadcasting, reductions, indexing\nand batched matrix multiplication")
    box(ax, 60, 574, 635, 130, "loss.backward()", "Seed the output gradient; traverse root → leaves\nAccumulate every shared-input gradient contribution")
    box(ax, 60, 759, 635, 113, "SGD / Adam", "Read parameter gradients and update NumPy arrays", color=TEAL)
    arrow(ax, (376, 361), (376, 389))
    arrow(ax, (376, 537), (376, 571))
    arrow(ax, (376, 704), (376, 756), color=TEAL)
    label(ax, 405, 719, "parameter gradients", 13, TEAL)

    box(ax, 825, 231, 490, 130, "NumPy CPU", "Default tensor operations\nRegular Linear = matmul + bias", color=BLUE)
    box(ax, 825, 406, 490, 165, "NativeLinear · optional CPU", "Fused Linear forward / backward\npybind11 extension\nmacOS: Accelerate · Linux: CBLAS", color=TEAL, title_size=18)
    box(ax, 825, 605, 490, 181, "cuda_linear · experimental", "Standalone fused Linear operator\nCustom CUDA forward / backward\nNumPy → GPU → NumPy each call\nNot a device-resident GPT runtime", color=AMBER, title_size=18)
    arrow(ax, (695, 431), (822, 296), rad=-.12)
    arrow(ax, (695, 457), (822, 480), color=TEAL)
    arrow(ax, (695, 484), (822, 695), color=AMBER, style="--", rad=.12)
    label(ax, 815, 829, "Solid: default / NativeLinear\nDashed: explicit experimental operator", 14, MUTED)
    ax.plot([55, 1325], [915, 915], color="#d5dfeb", lw=1.3)
    label(ax, 55, 937, "PyTorch is the numerical reference for outputs and gradients; it is outside Fluxion execution.", 16, MUTED)
    export(fig, "architecture")


def tensor_node(ax, x, y, title, sub, color=BLUE):
    box(ax, x, y, 245, 150, title, sub, color=color, title_size=20)


def autograd():
    fig, ax = canvas(1380, 965)
    label(ax, 55, 38, "REVERSE-MODE AUTOGRAD", 15, BLUE, "bold")
    label(ax, 55, 74, "One shared input. Three gradient contributions.", 29, INK, "bold")
    label(ax, 55, 126, "Example: y = x × x + x, evaluated at x = 3", 20, MUTED)

    label(ax, 65, 202, "1  FORWARD: record tensor parents", 17, BLUE, "bold")
    tensor_node(ax, 65, 329, "x = 3", "Shared leaf tensor")
    tensor_node(ax, 550, 329, "p = x × x", "Value: 9\nParents: (x, x)")
    tensor_node(ax, 1040, 329, "y = p + x", "Value: 12\nParents: (p, x)")
    arrow(ax, (310, 366), (547, 366), rad=-.08)
    arrow(ax, (310, 412), (547, 412), rad=.08)
    label(ax, 395, 315, "input 1", 15, BLUE)
    label(ax, 395, 447, "input 2", 15, BLUE)
    arrow(ax, (795, 389), (1037, 389))
    arrow(ax, (188, 329), (1163, 329), rad=-.17)
    label(ax, 566, 272, "x also goes directly into the addition", 15, BLUE)

    ax.plot([55, 1325], [498, 498], color="#d5dfeb", lw=1.3)
    label(ax, 65, 523, "2  BACKWARD: seed y, then visit root → leaves", 17, TEAL, "bold")
    tensor_node(ax, 65, 650, "x.grad = 7", "1 + 3 + 3", color=TEAL)
    tensor_node(ax, 550, 650, "p.grad = 1", "Local rule:\nmultiply", color=TEAL)
    tensor_node(ax, 1040, 650, "y.grad = 1", "Seed for\nscalar output", color=TEAL)
    arrow(ax, (1040, 710), (798, 710), color=TEAL)
    label(ax, 885, 675, "× 1", 18, TEAL)
    arrow(ax, (550, 689), (313, 689), color=TEAL, rad=.08)
    arrow(ax, (550, 735), (313, 735), color=TEAL, rad=-.08)
    label(ax, 390, 661, "+ 3", 16, TEAL)
    label(ax, 390, 754, "+ 3", 16, TEAL)
    arrow(ax, (1163, 650), (188, 650), color=TEAL, rad=.17)
    label(ax, 542, 590, "direct path contributes + 1", 16, TEAL)

    box(ax, 65, 831, 1220, 91, "Shared nodes are visited once; gradient contributions still add.",
        "First contribution: copy into grad. Later contributions: accumulate in place.",
        color=TEAL, title_size=18)
    export(fig, "autograd")


if __name__ == "__main__":
    architecture()
    autograd()
