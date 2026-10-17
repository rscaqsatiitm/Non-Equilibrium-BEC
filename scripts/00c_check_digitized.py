"""Plot every digitized data set for visual comparison with the PRL figures."""
import os, glob, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rd = csv.reader(f); hdr = next(rd)
        rows = [[float(v) if v not in ("", "nan") and v.replace(".", "").replace("-", "").replace("e", "").isdigit() else v for v in r] for r in rd]
    return hdr, rows
def col(rows, i): return np.array([r[i] for r in rows], dtype=float)

fig, axs = plt.subplots(3, 4, figsize=(16, 11))
# Fig 2
for tag, c, m in (("nodiss", "b", "v"), ("diss", "r", "o")):
    h, r = load(f"fig2_{tag}.csv"); t = col(r, 0)
    for k, (name, ei) in enumerate((("N", 2), ("T_nK", 4), ("F", 6), ("mu_nK", 8))):
        axs[0, k].errorbar(t, col(r, h.index(name)), col(r, ei), fmt=m, color=c, ms=4)
        axs[0, k].set_title("Fig2 " + name)
    h, r = load(f"fig2c_Feq_{tag}.csv"); axs[0, 2].plot(col(r, 0), col(r, 1), "--", color=c)
    h, r = load(f"fig2c_Feq_band_{tag}.csv"); axs[0, 2].fill(col(r, 0), col(r, 1), color=c, alpha=0.2)
    h, r = load(f"fig2d_muBEC_{tag}.csv"); axs[0, 3].plot(col(r, 0), col(r, 1), "--", color=c)
# Fig 3
ax = axs[1, 0]
h, r = load("fig3_open_nodiss.csv"); ax.errorbar(col(r, 0), col(r, 1), col(r, 3), col(r, 2), fmt="o", mfc="none", color="b", ms=4)
h, r = load("fig3_filled_diss.csv")
for row in r:
    rgb = [float(v) for v in row[5].split(",")]
    ax.errorbar(row[0], row[1], row[3], row[2], fmt="s" if row[4] == "square" else "o", color=rgb, ms=5)
h, r = load("fig3_eq1_curve.csv"); ax.plot(col(r, 0), col(r, 1), "b-")
for f in glob.glob(os.path.join(D, "fig3_model_dotted_*.csv")):
    h, r = load(os.path.basename(f)); ax.plot(col(r, 0), col(r, 1), ".", ms=2)
ax.set_xlim(0.5, 1.5); ax.set_ylim(0, 0.8); ax.set_title("Fig3")
# Fig 4
ax = axs[1, 1]
h, r = load("fig4_deltaF.csv"); ax.errorbar(col(r, 0), col(r, 1), col(r, 2), fmt="ro", ms=4)
h, r = load("fig4_fit_curve.csv"); ax.plot(col(r, 0), col(r, 1), "-"); ax.set_title("Fig4 dF"); ax.set_ylim(0, 0.3)
# Fig 1b
ax = axs[1, 2]
h, r = load("fig1b_gammaK.csv")
for s, c in (("nK_1e16_m-3", "r"), ("gammaK_Hz", "b")):
    rr = [x for x in r if x[0] == s]
    ax.plot([x[1] for x in rr], [x[2] for x in rr], "o", color=c)
ax.set_title("Fig1b nK (red), gammaK (blue)")
# S1
ax = axs[1, 3]
h, r = load("figS1_omega_calculated.csv"); ax.plot(col(r, 0), col(r, 1), "ro")
h, r = load("figS1_omega_measured.csv"); ax.plot(col(r, 0), col(r, 1), "bv")
tt = np.linspace(0, 2, 5); ax.plot(tt, 192 - 56 * tt, "r--"); ax.plot(tt, 153 - 44 * tt, "b-"); ax.set_title("FigS1")
# S2 i
for tag, c, m in (("nodiss", "b", "v"), ("diss", "r", "o")):
    h, r = load(f"figS2i_{tag}.csv"); t = col(r, 0)
    for k, name in enumerate(("N", "T_nK", "F", "mu_nK")):
        axs[2, k].errorbar(t, col(r, h.index(name)), col(r, h.index(name) + 1), fmt=m, color=c, ms=4); axs[2, k].set_title("S2i " + name)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "check_digitized_1.png"), dpi=90)

fig, axs = plt.subplots(2, 4, figsize=(16, 7))
for tag, c, m in (("nodiss", "b", "v"), ("diss", "r", "o")):
    h, r = load(f"figS2ii_{tag}.csv"); t = col(r, 0)
    for k, name in enumerate(("N", "tau", "F", "T_nK")):
        axs[0, k].errorbar(t, col(r, h.index(name)), col(r, h.index(name) + 1), fmt=m, color=c, ms=4); axs[0, k].set_title("S2ii " + name)
    for k, p in ((0, "a"), (2, "c")):
        h, r = load(f"figS2ii{p}_curve_{tag}_0.csv"); axs[0, k].plot(col(r, 0), col(r, 1), "-", color=c)
for k, panel in enumerate(("bec", "thermal")):
    ax = axs[1, k]
    for tag, c in (("withK", "g"), ("withoutK", "b")):
        h, r = load(f"figS5_{panel}_{tag}.csv"); ax.plot(col(r, 0), col(r, 1), "o", color=c, ms=3)
        for f in sorted(glob.glob(os.path.join(D, f"figS5_{panel}_{tag}_fit_*.csv"))):
            h, r = load(os.path.basename(f)); ax.plot(col(r, 0), col(r, 1), "-", color=c)
    ax.set_title("S5 " + panel)
for k, name in enumerate(("Nth", "N0", "F")):
    ax = axs[1, 2] if k < 2 else axs[1, 3]
    h, r = load(f"figS7_{name}_data.csv"); ax.plot(col(r, 0), col(r, 1), "o", ms=3, label=name)
    h, r = load(f"figS7_{name}_model.csv"); ax.plot(col(r, 0), col(r, 1), "-")
    ax.legend(); ax.set_title("S7")
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "check_digitized_2.png"), dpi=90)
print("ok")
