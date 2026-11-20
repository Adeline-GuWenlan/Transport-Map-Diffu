"""Read the plotted data of the paper's Fig. 3 out of its vector source figures (arXiv 2502.02483 v3 source tarball,
figures/mmd_2d.pdf and figures/std_over_eps_avg_over_xt.pdf). Parses the PDF content stream: tick marks give the axis mapping, stroked
polylines in a series colour give the curves. These are values READ OFF THE PAPER'S FIGURE (paper values), never our results.

Needs pypdf (in /scratch/wg2381/.cache-home/ddm2d_site). Usage: python scripts/extract_paper_fig3.py <figures_dir> <out_dir>
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/wg2381/.cache-home/ddm2d_site")
from pypdf import PdfReader  # noqa: E402

COLORS = {"0.1215686275 0.4666666667 0.7058823529": "C (beta=0.1, lambda=1)",
          "1 0.4980392157 0.0549019608": "B (beta=2, lambda=0)",
          "0.1725490196 0.6274509804 0.1725490196": None}   # green: 'Diffusion model' (left) / 'Posterior' (right)


def parse(path):
    c = PdfReader(path).pages[0].get_contents().get_data().decode("latin1")
    # ticks: "x y1 m x y2 l" with label text after; collect (pos, label)
    xt = [(float(a), float(lab)) for a, lab in re.findall(r"([\d.]+) 47\.23125 m\n\1 43\.73125 l\n\nB\n.*?\[ \((.*?)\) \] TJ", c, re.S)]
    yt = [(float(b), float(lab)) for _, b, _, lab in re.findall(r"([\d.]+) ([\d.]+) m\n([\d.]+) \2 l\n\nB\n.*?\[ \((.*?)\) \] TJ", c, re.S)]
    curves = {}
    for col, body in re.findall(r"re W n 2 J 1\.5 w ([\d.\s]+?) RG /DeviceRGB cs\n\n((?:[\d.]+ [\d.]+ [ml]\n)+)", c):
        col = " ".join(col.split())
        pts = [tuple(map(float, l.split()[:2])) for l in body.strip().splitlines()]
        curves[col] = pts
    return xt, yt, curves


def lin(ticks):
    (p0, v0), (p1, v1) = ticks[0], ticks[-1]
    return lambda p: v0 + (p - p0) * (v1 - v0) / (p1 - p0)


def main():
    figs, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, green, fname in [("fig3_left_mmd2", "Diffusion model (arm A)", "mmd_2d.pdf"),
                               ("fig3_right_posterior_std", "exact posterior", "std_over_eps_avg_over_xt.pdf")]:
        xt, yt, curves = parse(figs / fname)
        fx, fy = lin(xt), lin(yt)
        series = {}
        for col, pts in curves.items():
            label = COLORS.get(col, col) or green
            series[label] = [[round(fx(x), 4), round(fy(y), 6)] for x, y in pts]
        res[name] = {"source": f"arXiv 2502.02483 v3 source, figures/{fname}", "x_ticks": xt, "y_ticks": yt, "series": series,
                     "note": "values read off the paper's vector figure (paper values, not ours); exact up to the PDF's 6-decimal coordinates"}
        print(name, json.dumps(series, indent=0)[:2000])
    (out / "fig3_paper_values.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
