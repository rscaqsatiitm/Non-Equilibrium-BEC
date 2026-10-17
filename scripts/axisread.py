"""Read tick labels (glyph outlines) and tick marks of vector figures and
build axis calibrations.  Also helpers to collect markers and error bars."""
import numpy as np
from pdfvec import Path
from glyphs import GlyphClusters, is_glyph, signature, dist

# cluster index -> character, built from the contact sheet of the main paper
# (scripts/glyphs.py output, inspected by eye).  Supplement figures contain
# real text and do not need this table.
CHAR_MAP = {0: "1", 1: "0", 2: "2", 10: "3", 11: "4", 12: "5", 13: "6", 25: "-"}


def read_labels(paths, gc: GlyphClusters, char_map=CHAR_MAP, region=None):
    """Return list of (text, xc, yc, h) numeric labels made of glyph outlines."""
    glyphs = []
    for p in paths:
        if not is_glyph(p, smin=0.5):
            continue
        x0, y0, x1, y1 = p.bbox
        if region and not (region[0] - 15 <= x0 and x1 <= region[2] + 15 and region[1] - 15 <= y0 and y1 <= region[3] + 15):
            continue
        if p.size < 1.5:
            ch = "."
        else:
            ci = gc.assign(p)
            ch = char_map.get(ci)
            if ch is None:
                continue
        glyphs.append((ch, (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0, x0, x1, y0, y1))
    # group into labels: rows by baseline (y0 within 2.5 pt), then x order
    glyphs.sort(key=lambda g: g[6])
    rows = []
    for g in glyphs:
        if rows and abs(g[6] - rows[-1][-1][6]) < 2.5:
            rows[-1].append(g)
        else:
            rows.append([g])
    labels = []
    for row in rows:
        row.sort(key=lambda g: g[4])
        for g in row:
            if labels and labels[-1][-1] in row and -0.5 <= g[4] - labels[-1][-1][5] < 3.0:
                labels[-1].append(g)
            else:
                labels.append([g])
    out = []
    for lab in labels:
        txt = "".join(g[0] for g in lab)
        try:
            val = float(txt)
        except ValueError:
            continue
        xs = [g[4] for g in lab] + [g[5] for g in lab]
        ys = [g[6] for g in lab] + [g[7] for g in lab]
        out.append((val, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(ys) - min(ys)))
    return out


def tick_marks(paths, box, length=(1.5, 4.5)):
    """Short black stroked segments attached to the box edges."""
    x0, y0, x1, y1 = box
    xt, yt = [], []
    for p in paths:
        if not p.stroke or len(p.subpaths) != 1 or len(p.subpaths[0]) != 2:
            continue
        a, b = p.subpaths[0]
        L = np.hypot(*(b - a))
        if not (length[0] <= L <= length[1]):
            continue
        if abs(a[1] - b[1]) < 1e-3:   # horizontal segment -> y tick
            if min(abs(a[0] - x0), abs(b[0] - x0), abs(a[0] - x1), abs(b[0] - x1)) < 0.5 and y0 - 0.5 <= a[1] <= y1 + 0.5:
                yt.append(a[1])
        elif abs(a[0] - b[0]) < 1e-3:  # vertical -> x tick
            if min(abs(a[1] - y0), abs(b[1] - y0), abs(a[1] - y1), abs(b[1] - y1)) < 0.5 and x0 - 0.5 <= a[0] <= x1 + 0.5:
                xt.append(a[0])
    return sorted(set(np.round(xt, 2))), sorted(set(np.round(yt, 2)))


def calibrate(ticks, labels, axis, box, tol=2.5):
    """Match labels to ticks along `axis` ('x' or 'y'); return (slope, intercept)
    such that value = slope*coord + intercept, and the matched pairs."""
    pairs = []
    for val, xc, yc, h in labels:
        for t in ticks:
            if axis == "x" and abs(xc - t) < tol and box[1] - 25 < yc < box[1] - 2.5:
                pairs.append((t, val))
            if axis == "y" and abs(yc - t) < tol and (box[0] - 30 < xc < box[0] - 2.5 or box[2] + 2.5 < xc < box[2] + 30):
                pairs.append((t, val))
    if len(pairs) < 2:
        raise RuntimeError(f"calibration failed on {axis}: {pairs}")
    P = np.array(pairs)
    slope, inter = np.polyfit(P[:, 0], P[:, 1], 1)
    resid = np.abs(np.polyval([slope, inter], P[:, 0]) - P[:, 1]).max()
    return slope, inter, pairs, resid


def markers(paths, box, color, fill=None, size=(2.5, 7.0)):
    """Centroids of marker paths of a given colour inside the box."""
    out = []
    for p in paths:
        c = p.fill_color if p.fill else p.stroke_color
        if tuple(round(v, 2) for v in c) != color:
            continue
        if fill is not None and p.fill != fill:
            continue
        if not (size[0] <= p.size <= size[1]):
            continue
        x0, y0, x1, y1 = p.bbox
        if not (box[0] - 3 <= x0 and x1 <= box[2] + 3 and box[1] - 3 <= y0 and y1 <= box[3] + 3):
            continue
        out.append(((x0 + x1) / 2, (y0 + y1) / 2, p))
    # dedupe (fill + stroke drawn twice)
    ded = []
    for x, y, p in out:
        if all(abs(x - u) > 0.3 or abs(y - v) > 0.3 for u, v, _ in ded):
            ded.append((x, y, p))
    return ded


def error_bars(paths, box, color):
    """Two-point stroked segments of a colour inside the box: returns vertical
    and horizontal bars as (x, y_lo, y_hi) and (y, x_lo, x_hi)."""
    vert, horiz = [], []
    for p in paths:
        if not p.stroke or p.fill or len(p.subpaths) != 1 or len(p.subpaths[0]) != 2:
            continue
        if tuple(round(v, 2) for v in p.stroke_color) != color:
            continue
        a, b = p.subpaths[0]
        if not (box[0] - 1 <= min(a[0], b[0]) and max(a[0], b[0]) <= box[2] + 1 and box[1] - 1 <= min(a[1], b[1]) and max(a[1], b[1]) <= box[3] + 1):
            continue
        if abs(a[0] - b[0]) < 1e-3:
            vert.append((a[0], min(a[1], b[1]), max(a[1], b[1])))
        elif abs(a[1] - b[1]) < 1e-3:
            horiz.append((a[1], min(a[0], b[0]), max(a[0], b[0])))
    return vert, horiz
