"""Recover the plotted data of the PRL figures from the vector graphics of the
PDFs (and Fig. 1(b) from its raster image).  Writes data/digitized/*.csv.

Usage: python scripts/00_digitize.py <main.pdf> <supp.pdf>
"""
import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pdfvec import extract
from glyphs import GlyphClusters, is_glyph
from axisread import read_labels, tick_marks, calibrate, markers, error_bars

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "digitized")
os.makedirs(OUT, exist_ok=True)
MAIN, SUPP = sys.argv[1], sys.argv[2]

MAIN_CHARS = {0: "1", 1: "0", 2: "2", 10: "3", 11: "4", 12: "5", 13: "6", 25: "-", 28: ".", 30: "."}
SUPP_CHARS = {0: "1", 1: "0", 2: "2", 3: "4", 11: "5", 20: "6", 21: "8", 30: "3", 31: "-"}


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow([f"{v:.6g}" if isinstance(v, (float, np.floating)) else v for v in r])
    print(f"wrote {name}: {len(rows)} rows")


class Axis:
    def __init__(self, xs, xi, ys, yi):
        self.xs, self.xi, self.ys, self.yi = xs, xi, ys, yi
    def x(self, px): return self.xs * px + self.xi
    def y(self, py): return self.ys * py + self.yi


def axis_from_glyphs(paths, gc, chars, box, xbox=None, yscale=1.0, xscale=1.0):
    big = (box[0] - 40, box[1] - 40, box[2] + 40, box[3] + 40)
    labs = read_labels(paths, gc, chars, region=big)
    xt, yt = tick_marks(paths, box)
    if xbox is not None:        # take x ticks/labels from another panel (shared axis)
        labs2 = read_labels(paths, gc, chars, region=(xbox[0] - 40, xbox[1] - 40, xbox[2] + 40, xbox[3] + 40))
        xt2, _ = tick_marks(paths, xbox)
        xs, xi, _, rx = calibrate(xt2, labs2, "x", xbox)
    else:
        xs, xi, _, rx = calibrate(xt, labs, "x", box)
    ys, yi, _, ry = calibrate(yt, labs, "y", box)
    assert rx < 0.02 * abs(xs) * (box[2] - box[0]) and ry < 0.02 * abs(ys) * (box[3] - box[1]), (rx, ry)
    return Axis(xs * xscale, xi * xscale, ys * yscale, yi * yscale)


def series(paths, box, ax, color, fill=True, size=(2.5, 7.0), errcolor=None, shape=None):
    """Markers with vertical error bars -> rows (x, y, yerr_lo, yerr_hi)."""
    pts = markers(paths, box, color, fill=fill, size=size)
    if shape is not None:
        pts = [q for q in pts if shape(q[2])]
    vert, horiz = error_bars(paths, box, errcolor or color)
    rows = []
    for x, y, p in sorted(pts):
        elo = ehi = 0.0
        for vx, lo, hi in vert:
            if abs(vx - x) < 0.6 and lo - 0.6 <= y <= hi + 0.6:
                elo, ehi = ax.y(y) - ax.y(lo), ax.y(hi) - ax.y(y)
        xlo = xhi = 0.0
        for hy, lo, hi in horiz:
            if abs(hy - y) < 0.6 and lo - 0.6 <= x <= hi + 0.6:
                xlo, xhi = ax.x(x) - ax.x(lo), ax.x(hi) - ax.x(x)
        rows.append((ax.x(x), ax.y(y), abs(elo), abs(ehi), abs(xlo), abs(xhi)))
    return rows


def polyline(paths, box, ax, color, minpts=5, lw=None):
    out = []
    for p in paths:
        if not p.stroke or p.fill or tuple(round(v, 2) for v in p.stroke_color) != color:
            continue
        if sum(len(s) for s in p.subpaths) < minpts or p.size < 15:
            continue
        x0, y0, x1, y1 = p.bbox
        if not (box[0] - 2 <= x0 and x1 <= box[2] + 2 and box[1] - 2 <= y0 and y1 <= box[3] + 2):
            continue
        for s in p.subpaths:
            out.append([(ax.x(x), ax.y(y)) for x, y in s])
    return out


def is_circle(p):
    return len(p.subpaths[0]) > 12


def is_square(p):
    return len(p.subpaths[0]) in (4, 5)


# ----------------------------------------------------------------------------
# Main paper
it = extract(MAIN, [1, 2])
gc = GlyphClusters()
for p in it.paths:
    if p.page in (1, 2) and is_glyph(p):
        gc.assign(p)
P1 = [p for p in it.paths if p.page == 1]
P2 = [p for p in it.paths if p.page == 2]

box2 = {"a": (344.66, 319.15, 432.09, 404.32), "b": (433.84, 319.15, 521.26, 404.32),
        "c": (344.66, 232.27, 432.09, 317.44), "d": (433.84, 232.27, 521.26, 317.44)}
ax2 = {"a": axis_from_glyphs(P1, gc, MAIN_CHARS, box2["a"], xbox=box2["c"], yscale=1e4),
       "b": axis_from_glyphs(P1, gc, MAIN_CHARS, box2["b"], xbox=box2["d"]),
       "c": axis_from_glyphs(P1, gc, MAIN_CHARS, box2["c"]),
       "d": axis_from_glyphs(P1, gc, MAIN_CHARS, box2["d"])}
BLUE, RED = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
RED, BLUE = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
for tag, col in (("nodiss", BLUE), ("diss", RED)):
    rows = {}
    for k, name in zip("abcd", ("N", "T_nK", "F", "mu_nK")):
        for x, y, elo, ehi, _, _ in series(P1, box2[k], ax2[k], col):
            t = round(x, 2)
            rows.setdefault(t, {})[name] = (y, 0.5 * (elo + ehi))
    out = []
    for t in sorted(rows):
        r = rows[t]
        out.append([t] + sum([list(r.get(n, (np.nan, np.nan))) for n in ("N", "T_nK", "F", "mu_nK")], []))
    write(f"fig2_{tag}.csv", ["t_s", "N", "N_err", "T_nK", "T_err", "F", "F_err", "mu_nK", "mu_err"], out)
# equilibrium curves and bands of Fig. 2(c), BEC chemical potential lines of 2(d)
for tag, col in (("nodiss", BLUE), ("diss", RED)):
    for seg in polyline(P1, box2["c"], ax2["c"], col, minpts=20):
        write(f"fig2c_Feq_{tag}.csv", ["t_s", "F_eq"], seg)
    for seg in polyline(P1, box2["d"], ax2["d"], col, minpts=5):
        write(f"fig2d_muBEC_{tag}.csv", ["t_s", "mu_BEC_nK"], seg)
for p in P1:
    if p.fill and p.size > 10 and tuple(round(c, 2) for c in p.fill_color) in ((1.0, 0.5, 0.5), (0.7, 0.7, 1.0)):
        tag = "diss" if p.fill_color[0] > 0.9 else "nodiss"
        write(f"fig2c_Feq_band_{tag}.csv", ["t_s", "F"], [(ax2["c"].x(x), ax2["c"].y(y)) for x, y in p.points])

# Fig. 3: tick labels are real text whose positions are not recoverable from
# our parser, so the calibration uses the tick marks (0.5..1.5 step 0.1 on x,
# 0.0..0.8 step 0.1 on y, read from the rendered figure).
box3 = (85.2, 243.7, 293.9, 411.8)
xt3, yt3 = tick_marks(P2, box3)
assert len(xt3) == 11 and len(yt3) == 9, (xt3, yt3)
xs, xi = np.polyfit(xt3, np.arange(0.5, 1.51, 0.1), 1)
ys, yi = np.polyfit(yt3, np.arange(0.0, 0.81, 0.1), 1)
ax3 = Axis(xs, xi, ys, yi)
# open symbols: white-filled marker + blue outline (no dissipation)
rows = []
for x, y, p in markers(P2, box3, (1.0, 1.0, 1.0), fill=True, size=(3.5, 7.5)):
    rows.append((x, y, len(p.subpaths[0])))
vert, horiz = error_bars(P2, box3, BLUE)
out = []
for x, y, n in sorted(rows):
    elo = ehi = xlo = xhi = 0.0
    for vx, lo, hi in vert:
        if abs(vx - x) < 0.6 and lo - 0.6 <= y <= hi + 0.6:
            elo, ehi = ax3.y(y) - ax3.y(lo), ax3.y(hi) - ax3.y(y)
    for hy, lo, hi in horiz:
        if abs(hy - y) < 0.6 and lo - 0.6 <= x <= hi + 0.6:
            xlo, xhi = ax3.x(x) - ax3.x(lo), ax3.x(hi) - ax3.x(x)
    out.append((ax3.x(x), ax3.y(y), max(xlo, xhi), max(elo, ehi), n))
write("fig3_open_nodiss.csv", ["tau", "F", "tau_err", "F_err", "marker_npts"], out)
# filled markers (dissipation trajectories), any colour, with their colour
out = []
for p in P2:
    if not p.fill or p.stroke or not (3.5 <= p.size <= 7.5):
        continue
    c = tuple(round(v, 2) for v in p.fill_color)
    if c in ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)):
        continue
    x0, y0, x1, y1 = p.bbox
    if not (box3[0] <= x0 and x1 <= box3[2] and box3[1] <= y0 and y1 <= box3[3]):
        continue
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    vert, horiz = error_bars(P2, box3, c)
    elo = ehi = xlo = xhi = 0.0
    for vx, lo, hi in vert:
        if abs(vx - x) < 0.6 and lo - 0.6 <= y <= hi + 0.6:
            elo, ehi = ax3.y(y) - ax3.y(lo), ax3.y(hi) - ax3.y(y)
    for hy, lo, hi in horiz:
        if abs(hy - y) < 0.6 and lo - 0.6 <= x <= hi + 0.6:
            xlo, xhi = ax3.x(x) - ax3.x(lo), ax3.x(hi) - ax3.x(x)
    shape = "circle" if is_circle(p) else ("square" if is_square(p) else f"n{len(p.subpaths[0])}")
    out.append((ax3.x(x), ax3.y(y), max(xlo, xhi), max(elo, ehi), shape, f"{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}"))
write("fig3_filled_diss.csv", ["tau", "F", "tau_err", "F_err", "marker", "rgb"], out)
# Eq.(1) curve and dotted model lines
for seg in polyline(P2, box3, ax3, BLUE, minpts=7):
    write("fig3_eq1_curve.csv", ["tau", "F_eq"], seg)
dots = {}
for p in P2:
    if p.stroke and not p.fill and p.size < 0.5:
        c = tuple(round(v, 2) for v in p.stroke_color)
        x0, y0, x1, y1 = p.bbox
        if box3[0] <= x0 and x1 <= box3[2] and box3[1] <= y0 and y1 <= box3[3]:
            dots.setdefault(c, []).append((ax3.x((x0 + x1) / 2), ax3.y((y0 + y1) / 2)))
for c, pts in dots.items():
    write(f"fig3_model_dotted_{c[0]:.2f}_{c[1]:.2f}_{c[2]:.2f}.csv", ["tau", "F"], sorted(pts))

# Fig. 4
box4 = (378.05, 176.18, 528.61, 283.73)
ax4 = axis_from_glyphs(P2, gc, MAIN_CHARS, box4)
out = [(x, y, 0.5 * (elo + ehi)) for x, y, elo, ehi, _, _ in series(P2, box4, ax4, RED)]
write("fig4_deltaF.csv", ["t_s", "dF", "dF_err"], out)
for seg in polyline(P2, box4, ax4, (0.12, 0.47, 0.71), minpts=5):
    write("fig4_fit_curve.csv", ["t_s", "dF_fit"], seg)

# ----------------------------------------------------------------------------
# Supplement
its = extract(SUPP, [0, 1, 4, 6])
S = {k: [p for p in its.paths if p.page == k] for k in (0, 1, 4, 6)}
T = {k: [t for t in its.texts if t.page == k] for k in (0, 1, 4, 6)}

# Fig. S2 (i) and (ii): glyph-outline labels, markers with error bars
gcs = GlyphClusters()
for p in S[1]:
    if is_glyph(p):
        gcs.assign(p)
boxS2 = {"ia": (87.46, 632.09, 172.82, 715.24), "ib": (174.52, 632.09, 259.88, 715.24),
         "ic": (87.46, 547.27, 172.82, 630.42), "id": (174.52, 547.27, 259.88, 630.42),
         "iia": (348.51, 631.14, 433.86, 714.3), "iib": (435.57, 631.14, 520.92, 714.3),
         "iic": (348.51, 546.33, 433.86, 629.48), "iid": (435.57, 546.33, 520.92, 629.48)}
axS2 = {}
for k, b in boxS2.items():
    xb = boxS2[k[:-1] + ("c" if k[-1] in "ac" else "d")]
    ysc = 1e4 if k.endswith("a") else 1.0
    axS2[k] = axis_from_glyphs(S[1], gcs, SUPP_CHARS, b, xbox=xb, yscale=ysc)
names = {"i": ("N", "T_nK", "F", "mu_nK"), "ii": ("N", "tau", "F", "T_nK")}
for grp in ("i", "ii"):
    for tag, col in (("nodiss", BLUE), ("diss", RED)):
        rows = {}
        for k, name in zip("abcd", names[grp]):
            for x, y, elo, ehi, _, _ in series(S[1], boxS2[grp + k], axS2[grp + k], col, fill=None, size=(2.5, 6.0)):
                rows.setdefault(round(x, 2), {})[name] = (y, 0.5 * (elo + ehi))
        out = []
        for t in sorted(rows):
            r = rows[t]
            out.append([t] + sum([list(r.get(n, (np.nan, np.nan))) for n in names[grp]], []))
        hdr = ["t_s"] + sum([[n, n.split("_")[0] + "_err"] for n in names[grp]], [])
        write(f"figS2{grp}_{tag}.csv", hdr, out)
        for k in "abcd":
            for j, seg in enumerate(polyline(S[1], boxS2[grp + k], axS2[grp + k], col, minpts=5)):
                write(f"figS2{grp}{k}_curve_{tag}_{j}.csv", ["t_s", names[grp]["abcd".index(k)]], seg)

# Fig. S1: real text labels (DejaVu Sans, digit advance 0.636 em)
boxS1 = (225.9, 151.3, 422.7, 251.8)
xt, yt = tick_marks(S[0], boxS1, length=(2.0, 4.5))
labs = []
for t in T[0]:
    try:
        v = float(t.text)
    except ValueError:
        continue
    w = 0.636 * t.size * len(t.text)
    labs.append((v, t.x + w / 2, t.y + 0.36 * t.size, t.size))
xs, xi, px, rx = calibrate(xt, labs, "x", boxS1, tol=4)
ys, yi, py, ry = calibrate(yt, labs, "y", boxS1, tol=4)
axS1 = Axis(xs, xi, ys, yi)
calc = [(x, y) for x, y, *_ in series(S[0], boxS1, axS1, RED, fill=None, size=(2.5, 6.0))]
meas = [(x, y) for x, y, *_ in series(S[0], boxS1, axS1, BLUE, fill=None, size=(2.5, 6.0))]
write("figS1_omega_calculated.csv", ["t_s", "f_Hz"], calc)
write("figS1_omega_measured.csv", ["t_s", "f_Hz"], meas)

# Fig. S5: CID-font labels (code = char - 29); two panels
def cid(s):
    return "".join(chr(b + 29) for b in s.encode("latin-1")[1::2])
for name, box, yvals in (("bec", (236.9, 610.6, 389.4, 711.2), None), ("thermal", (236.9, 495.6, 389.4, 596.2), None)):
    xt, yt = tick_marks(S[4], box, length=(2.4, 3.2))   # major ticks only
    xt = [t for t in xt if abs(t - box[0]) > 0.5 or True]
    labs = []
    for t in T[4]:
        if box[1] - 15 < t.y < box[3] + 5 and t.x < box[0]:
            try:
                v = float(cid(t.text))
            except ValueError:
                continue
            labs.append((v * 1e3, t.x + 0.3 * t.size * len(cid(t.text)), t.y + 0.36 * t.size, t.size))
    ys, yi, py, ry = calibrate(yt, labs, "y", box, tol=4)
    xmaj = sorted(xt)
    assert len(xmaj) == 6, xmaj
    xs, xi = np.polyfit(xmaj, [0, 1000, 2000, 3000, 4000, 5000], 1)
    axS5 = Axis(xs, xi, ys, yi)
    legend = (283.69, 692.69, 384.18, 708.21) if name == "bec" else (285.49, 562.01, 393.68, 577.53)
    for tag, col in (("withK", (0.0, 1.0, 0.0)), ("withoutK", BLUE)):
        rows = [(x, y) for x, y, *_ in series(S[4], box, axS5, col, fill=None, size=(1.5, 3.0))
                if not (axS5.x(x) > 0 and legend[0] <= (x - axS5.xi) / axS5.xs <= legend[2] and legend[1] <= (y - axS5.yi) / axS5.ys <= legend[3])]
        write(f"figS5_{name}_{tag}.csv", ["t_ms", "N"], rows)
        for j, seg in enumerate(polyline(S[4], box, axS5, col, minpts=50)):
            write(f"figS5_{name}_{tag}_fit_{j}.csv", ["t_ms", "N"], seg[::10])

# Fig. S7: per-character text; three panels
boxes7 = {"Nth": (77.3, 632.5, 216.5, 734.9), "N0": (248.9, 632.5, 388.1, 734.9), "F": (420.2, 632.5, 559.4, 734.9)}
cols7 = {"Nth": BLUE, "N0": (1.0, 0.5, 0.0), "F": (1.0, 0.0, 1.0)}
for name, box in boxes7.items():
    xt, yt = tick_marks(S[6], box, length=(1.5, 2.5))
    chars = [t for t in T[6] if box[0] - 25 < t.x < box[2] + 5 and box[1] - 12 < t.y < box[3] + 3 and len(t.text) == 1]
    chars.sort(key=lambda t: (round(t.y), t.x))
    labs, cur = [], []
    for t in chars:
        if cur and abs(t.y - cur[-1].y) < 1 and 0 < t.x - cur[-1].x < 5:
            cur.append(t)
        else:
            if cur: labs.append(cur)
            cur = [t]
    if cur: labs.append(cur)
    L = []
    for grp in labs:
        try:
            v = float("".join(t.text for t in grp))
        except ValueError:
            continue
        xc = (grp[0].x + grp[-1].x + 0.64 * grp[-1].size) / 2
        L.append((v, xc, grp[0].y + 0.36 * grp[0].size, grp[0].size))
    xs, xi, px, rx = calibrate(xt, L, "x", box, tol=4)
    ys, yi, py, ry = calibrate(yt, L, "y", box, tol=4)
    ax7 = Axis(xs, xi, ys, yi)
    rows = [(x, y) for x, y, *_ in series(S[6], box, ax7, cols7[name], fill=None, size=(1.5, 3.0))]
    write(f"figS7_{name}_data.csv", ["t_s", name], rows)
    for seg in polyline(S[6], box, ax7, cols7[name], minpts=50):
        write(f"figS7_{name}_model.csv", ["t_s", name], seg[::4])
print("done")
