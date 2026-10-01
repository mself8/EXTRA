"""논문 그림 공통 matplotlib 스타일 (serif·8pt·tight·colorblind-safe). 신규 유틸."""
import matplotlib

# Okabe-Ito colorblind-safe palette
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F",
}


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.3,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    })
