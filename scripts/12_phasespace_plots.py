"""Overlay phase-space results (data/ps_*.json) on the Fig. 2 data.
Usage: python scripts/12_phasespace_plots.py <tag_noK> [tag_K] [tag_K4]"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
data = {"noK": load("fig2_nodiss.csv"), "K": load("fig2_diss.csv"), "K4": load("figS2i_diss.csv")}
tags = dict(zip(["noK", "K", "K4"], sys.argv[1:4]))
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for key, tag, c, m in zip(tags, tags.values(), ("b", "r", "m"), ("v", "o", "s")):
    rows = json.load(open(os.path.join(ROOT, "data", f"ps_{tag}.json")))
    t = np.array([r["t_lab"] for r in rows]); d = data[key]
    g = lambda k: np.array([r[k] for r in rows])
    axs[0, 0].errorbar(d["t_s"], d["N"], d["N_err"], fmt=m, color=c, ms=4); axs[0, 0].plot(t, g("N"), "-", color=c, label=key)
    axs[0, 1].errorbar(d["t_s"], d["T_nK"], d["T_err"], fmt=m, color=c, ms=4)
    axs[0, 1].plot(t, g("Tmz") * 1e9, "-", color=c, label=key + " T_kin z (vertical)"); axs[0, 1].plot(t, g("Tmx") * 1e9, "--", color=c, label=key + " T_kin x"); axs[0, 1].plot(t, g("Tmy") * 1e9, ":", color=c, label=key + " T_kin y")
    axs[0, 2].errorbar(d["t_s"], d["F"], d["F_err"], fmt=m, color=c, ms=4); axs[0, 2].plot(t, g("F"), "-", color=c)
    axs[1, 0].plot(t, g("T_gauss_x") * 1e9, "--", color=c, label=key + " Gaussian-fit x"); axs[1, 0].plot(t, g("T_gauss_z") * 1e9, "-", color=c, label=key + " Gaussian-fit z"); axs[1, 0].errorbar(d["t_s"], d["T_nK"], d["T_err"], fmt=m, color=c, ms=4)
    axs[1, 1].plot(t, g("Tmx") / g("Tmz"), "--", color=c); axs[1, 1].plot(t, g("Tmy") / g("Tmz"), ":", color=c)
    axs[1, 2].plot(t, g("depth") / 1.380649e-23 * 1e9, "-", color=c)
axs[0, 0].set_title("N"); axs[0, 0].legend(fontsize=8); axs[0, 1].set_title("kinetic T per axis (nK)"); axs[0, 1].legend(fontsize=7); axs[0, 2].set_title("F")
axs[1, 0].set_title("TOF Gaussian-fit T (nK)"); axs[1, 0].legend(fontsize=7); axs[1, 1].set_title("T_x/T_z (dashed), T_y/T_z (dotted)"); axs[1, 2].set_title("saddle depth (nK)")
for ax in axs.flat: ax.set_xlabel("t (s)")
fig.suptitle("Phase-space model: " + ", ".join(f"{k}: {v}" for k, v in tags.items())); fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "phasespace_" + "_".join(tags.values()) + ".png"), dpi=110)
print("saved")
