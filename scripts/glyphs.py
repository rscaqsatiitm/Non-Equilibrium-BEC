"""Cluster filled black glyph outlines (text converted to paths) by shape so
that tick labels of vector figures can be read back as numbers."""
import numpy as np
from pdfvec import extract, Path


def signature(p: Path, n=24):
    """Shape signature: resampled, bbox-normalised outline of each subpath."""
    x0, y0, x1, y1 = p.bbox
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    sig = [len(p.subpaths), round(w / h, 1)]
    for sp in p.subpaths:
        q = (sp - [x0, y0]) / [w, h]
        # resample along the polyline
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
        if d[-1] < 1e-9:
            samp = np.repeat(q[:1], n, axis=0)
        else:
            t = np.linspace(0, d[-1], n)
            samp = np.c_[np.interp(t, d, q[:, 0]), np.interp(t, d, q[:, 1])]
        sig.append(samp)
    return sig


def dist(a, b):
    if a[0] != b[0] or abs(a[1] - b[1]) > 0.25:
        return 1e9
    return max(np.abs(x - y).max() for x, y in zip(a[2:], b[2:]))


def is_glyph(p: Path, smin=2.0, smax=12.0):
    if not p.fill or p.stroke:
        return False
    c = p.fill_color
    if not (len(c) == 1 and c[0] < 0.3 or len(c) == 3 and max(c) < 0.3 or len(c) == 4 and c[3] > 0.7):
        return False
    return smin <= p.size <= smax


class GlyphClusters:
    def __init__(self, tol=0.08):
        self.reps = []     # (signature, representative path)
        self.tol = tol

    def assign(self, p: Path):
        s = signature(p)
        for i, (rs, rp) in enumerate(self.reps):
            if dist(s, rs) < self.tol:
                return i
        self.reps.append((s, p))
        return len(self.reps) - 1


if __name__ == "__main__":
    import sys, json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdf, pages, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
    it = extract(pdf, pages)
    gc = GlyphClusters()
    labels = []
    for p in it.paths:
        if is_glyph(p):
            labels.append((gc.assign(p), p))
    print(len(labels), "glyphs in", len(gc.reps), "clusters")
    n = len(gc.reps)
    cols = 10
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    for i, (s, p) in enumerate(gc.reps):
        ax = axs.flat[i]
        for sp in p.subpaths:
            ax.fill(sp[:, 0], sp[:, 1], "k")
        ax.set_aspect("equal"); ax.set_title(str(i), fontsize=8); ax.axis("off")
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=110)
