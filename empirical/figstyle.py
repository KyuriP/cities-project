"""Shared style for the paper figures: sized for print (7.0 in = 180 mm full width), 7 pt Arial/Helvetica,
open axes, outward ticks, black for observations, greys for model variants, one accent colour."""
import matplotlib as mpl
import matplotlib.pyplot as plt

W = 7.0                  # full page width (in)
OBS = "k"                # observations
MOD = "k"                # main model
ALT = "0.55"             # alternative model / secondary series
BAND = "0.88"            # uncertainty bands
ACC = "#b2182b"          # single accent colour (used sparingly)


def apply():
    mpl.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "lines.linewidth": 1.0, "lines.markersize": 3,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def letters(axes, labels="abcdefgh", dx=-0.02, dy=1.0):
    for a, l in zip(axes, labels):
        a.text(dx, dy, l, transform=a.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def label(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 6.5); kw.setdefault("va", "center")
    ax.text(x, y, s, **kw)
