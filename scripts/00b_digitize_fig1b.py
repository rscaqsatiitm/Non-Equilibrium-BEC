"""Digitize Fig. 1(b) of the PRL (raster image embedded in the PDF): K density
(red circles, left axis, 1e16 m^-3) and measured dissipation rate gamma_K
(blue diamonds, right axis, Hz) versus K-MOT loading time.

Axis calibration: the panel frame is located from the dark pixel rows/columns;
the axis ranges (0..2.5 s, 0..1.0, 0..0.5 Hz) are read from the tick labels of
the rendered figure.  Marker centres are the centroids of connected clusters of
saturated red / blue pixels.
"""
import sys, os, csv
import numpy as np
from PIL import Image
from scipy import ndimage

png, out = sys.argv[1], sys.argv[2]
im = np.asarray(Image.open(png).convert("RGB")).astype(int)
H, W, _ = im.shape
r, g, b = im[..., 0], im[..., 1], im[..., 2]
# Panel (b) occupies the right part of the image; restrict to x > 0.45 W
x_off = int(0.45 * W)
sub = im[:, x_off:]
rs, gs, bs = sub[..., 0], sub[..., 1], sub[..., 2]
dark = (sub < 200).any(axis=2)   # frame and ticks are coloured (red/blue axes)
# frame: long dark horizontal and vertical lines
rows = np.where(dark.sum(axis=1) > 0.4 * dark.shape[1])[0]
cols = np.where(dark.sum(axis=0) > 0.4 * dark.shape[0])[0]
# cluster consecutive indices
def groups(idx):
    g, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            g.append(cur); cur = [i]
    g.append(cur)
    return [np.mean(c) for c in g]
yrows, xcols = groups(rows), groups(cols)
print("frame rows", yrows, "cols", xcols)
y_top, y_bot = min(yrows), max(yrows)
x_left, x_right = min(xcols), max(xcols)
# tick marks: bottom axis ticks point outward (band just below the frame);
# the tick labels also intrude into that band, so keep the groups that form an
# arithmetic sequence between the first and the last group (0 and 2.5 s).
band = dark[int(y_bot) + 2:int(y_bot) + 14, :]
cand = groups(np.where(band.sum(axis=0) >= 3)[0])
target = np.linspace(cand[0], cand[-1], 6)
xticks = [min(cand, key=lambda c: abs(c - t)) for t in target]
assert max(abs(np.array(xticks) - target)) < 3, (cand, xticks)
print("x ticks (px)", xticks)
# right-axis ticks (gamma_K, 0..0.5 Hz, step 0.1) point outward to the right;
# the frame top coincides with the 0.5 Hz tick
band = dark[:, int(x_right) + 2:int(x_right) + 14]
cand = groups(np.where(band.sum(axis=1) >= 3)[0])
cand = [c for c in cand if c < y_bot - 5]
yticks = sorted(cand)[:6]
print("y ticks (px)", yticks)
assert len(yticks) == 6 and abs(np.std(np.diff(yticks)) ) < 2, yticks
# calibration: x ticks 0..2.5 step 0.5; left y ticks 0..1.0 step 0.2 (major) possibly with minors
xt = np.array(xticks)
xs, xi = np.polyfit(xt, np.linspace(0, 2.5, len(xt)), 1)
yt = np.array(yticks)
ys_l, yi_l = np.polyfit(yt, np.linspace(1.0, 0.0, len(yt)), 1)      # left axis n_K (1e16 m^-3): same frame, 0..1.0
ys_r, yi_r = np.polyfit(yt, np.linspace(0.5, 0.0, len(yt)), 1)      # right axis gamma_K (Hz)
print("x cal", xs, xi, " y cal", ys_l, yi_l)

def centroids(mask, minpix=15):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        yy, xx = np.where(lab == i)
        if len(yy) >= minpix:
            out.append((xx.mean(), yy.mean(), len(yy), xx.max() - xx.min() + 1, yy.max() - yy.min() + 1))
    return out

inside = np.zeros_like(dark)
inside[int(y_top) + 2:int(y_bot) - 2, int(x_left) + 2:int(x_right) - 2] = True
red = (rs > 180) & (gs < 90) & (bs < 90) & inside
blue = (bs > 150) & (rs < 90) & (gs < 90) & inside
rows_out = []
for name, mask, cal in (("nK_1e16_m-3", red, (ys_l, yi_l)), ("gammaK_Hz", blue, (ys_r, yi_r))):
    cs = centroids(mask)
    # markers: roughly square blobs; error bars are tall thin blobs (red only)
    pts = []
    for x, y, n, w, h in cs:
        if w < 6:
            continue
        if h <= 1.6 * w:
            pts.append((x, y, w, h))
        else:
            # marker merged with its error bar: the marker centre is the
            # widest part of the blob
            lab, _ = ndimage.label(mask)
            yy, xx = np.where(lab == lab[int(round(y)), int(round(x))])
            widths = {}
            for yv, xv in zip(yy, xx):
                widths.setdefault(yv, []).append(xv)
            wmax = max(len(v) for v in widths.values())
            rows_c = [yv for yv, v in widths.items() if len(v) >= 0.8 * wmax]
            pts.append((x, float(np.mean(rows_c)), wmax, h))
    pts.sort()
    for x, y, w, h in pts:
        rows_out.append((name, xs * (x + x_off) * 0 + xs * x + xi, cal[0] * y + cal[1], w, h))
    print(name, [(round(xs * x + xi, 3), round(cal[0] * y + cal[1], 4)) for x, y, w, h in pts])
# red error bars: thin tall blobs -> half-length in data units
errs = []
for x, y, n, w, h in centroids(red, minpix=10):
    if h > 1.6 * w:
        errs.append((xs * x + xi, ys_l * (y - h / 2) + yi_l, ys_l * (y + h / 2) + yi_l))
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series", "t_s", "value", "blob_w_px", "blob_h_px"])
    for r_ in rows_out:
        w.writerow([r_[0], f"{r_[1]:.4f}", f"{r_[2]:.4f}", r_[3], r_[4]])
    for e in errs:
        w.writerow(["nK_errorbar", f"{e[0]:.4f}", f"{e[1]:.4f}", f"{e[2]:.4f}", ""])
print("wrote", out)
