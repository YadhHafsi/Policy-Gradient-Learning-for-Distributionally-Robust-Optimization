r"""Publication-quality matplotlib defaults.

Every figure is saved as ``.pdf`` (Type-42 TrueType fonts) and ``.png``
(for quick inspection). The figure sizing uses the standard 5.5-inch
textwidth and the Wong (2011) colour-blind palette: the first four entries
remain distinguishable under grayscale.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


TEXTWIDTH_IN = 5.5                     # standard paper textwidth (inches).

_PAPER_RC = {
    # Fonts -- Times for body, STIX for math, Type-42 embedded TrueType.
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "STIX", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "text.usetex":        False,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    # Sizes tuned for 2-up layout at 5.5 in textwidth.
    "axes.labelsize":     9.5,
    "axes.titlesize":     10.0,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8.5,
    "legend.frameon":     False,
    "legend.handlelength":1.7,
    "legend.columnspacing":1.2,
    "legend.borderaxespad":0.4,
    # Minimalist academic axes.
    "axes.grid":          True,
    "grid.alpha":         0.22,
    "grid.linestyle":     "-",
    "grid.linewidth":     0.4,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.75,
    "axes.titlepad":      6.0,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "lines.linewidth":    1.6,
    "lines.markersize":   3.8,
    # Rendering.
    "figure.dpi":         120,
    "figure.constrained_layout.use": False,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "savefig.transparent":False,
}


def use_paper_style() -> None:
    """Activate the paper visual style (idempotent)."""
    mpl.rcParams.update(_PAPER_RC)


@contextmanager
def paper_style():
    """Scoped variant of :func:`use_paper_style`."""
    with mpl.rc_context(_PAPER_RC):
        yield


# Wong (2011) colour-blind-friendly palette.
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "cyan":   "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#555555",
}


def eps_colors(n_or_list) -> list:
    r"""Cycle :data:`PALETTE` once per value of :math:`\varepsilon`."""
    n    = n_or_list if isinstance(n_or_list, int) else len(n_or_list)
    keys = ["blue", "orange", "red", "green", "purple", "cyan"]
    return [PALETTE[keys[i % len(keys)]] for i in range(n)]


def new_fig(width_frac: float = 1.0, height: float = 2.6, *,
            columns: int = 1, aspect: Optional[float] = None):
    """Create a figure of standard width; if ``aspect`` is given,
    ``height = width_frac * TEXTWIDTH_IN * aspect``."""
    width = TEXTWIDTH_IN * width_frac
    if aspect is not None:
        height = width * aspect
    fig, axes = plt.subplots(1, columns, figsize=(width, height))
    return fig, axes


def savefig(fig, path: str, *, pdf: bool = True, png: bool = True) -> None:
    """Save ``fig`` to ``{path}.pdf`` and ``{path}.png`` (no extension in ``path``)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if pdf: fig.savefig(path + ".pdf")
    if png: fig.savefig(path + ".png")


def learning_curve(ax, history, *, label: str = r"$J(\theta)$",
                   color: Optional[str] = None,
                   reference: Optional[tuple] = None) -> None:
    """Standard learning-curve panel used across every experiment."""
    col = color or PALETTE["blue"]
    xs  = [h["iter"] for h in history]
    ys  = [h.get("J", h.get("J_hat")) for h in history]
    ax.plot(xs, ys, color=col, label=label, linewidth=1.7)
    if reference is not None:
        ref_value, ref_label = reference
        ax.axhline(ref_value, color=PALETTE["grey"], linestyle="--",
                   linewidth=1.1, label=ref_label)
    ax.set_xlabel("outer iteration")
    ax.set_ylabel(r"$J(\theta)$")
    ax.legend(loc="lower right")
