"""Overlay phase-space hold results (data/ps_hold_*.json) on Fig. S2(ii) / Fig. 4.
Usage: python scripts/14_phasespace_hold_plots.py <tag_nodiss> <tag_diss>"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nebec import constants as C
ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
data = {"nodiss": load("figS2ii_nodiss.csv"), "diss": load("figS2ii_diss.csv")}
dF = load("fig4_deltaF.csv")
tags = dict(zip(["nodiss", "diss"], sys.argv[1:3]))
om = 2 * np.pi * 99.2
def F_eq(N, T):
    Tc = 0.94 * C.hbar * om * N ** (1 / 3) / C.kB
    return np.clip(1 - (T / Tc) ** 3, 0, 1)
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
out = {}
for key, tag, c, m in zip(tags, tags.values(), ("b", "r"), ("v", "o")):
    rows = json.load(open(os.path.join(ROOT, "data", f"ps_{tag}.json")))
    t = np.array([r["t"] for r in rows]); d = data[key]
    g = lambda k: np.array([r[k] for r in rows])
    Tkin = (g("Tmx") + g("Tmz")) / 2; Tfit = (g("T_fit_x") + g("T_fit_z")) / 2
    axs[0, 0].errorbar(d["t_s"], d["N"], d["N_err"], fmt=m, color=c, ms=4, label=key + " data"); axs[0, 0].plot(t, g("N"), "-", color=c, label=key + " model")
    axs[0, 1].errorbar(d["t_s"], d["T_nK"], d["T_err"], fmt=m, color=c, ms=4); axs[0, 1].plot(t, Tkin * 1e9, "-", color=c, label=key + " T_kin"); axs[0, 1].plot(t, Tfit * 1e9, "--", color=c, label=key + " bimodal fit")
    axs[0, 2].errorbar(d["t_s"], d["F"], d["F_err"], fmt=m, color=c, ms=4); axs[0, 2].plot(t, g("F"), "-", color=c, label=key + " model F"); axs[0, 2].plot(t, g("F_fit"), "--", color=c, label=key + " F_fit")
    # delta F relative to the equilibrium value at the model's own (N, T)
    dFm = g("F") - F_eq(g("N"), Tkin); dFf = g("F_fit") - F_eq(g("N"), Tfit)
    axs[1, 0].plot(t, dFm, "-", color=c, label=key + " F - F_eq(N, T_kin)"); axs[1, 0].plot(t, dFf, "--", color=c, label=key + " F_fit - F_eq(N, T_fit)")
    axs[1, 1].plot(t, g("Tmx") / g("Tmz"), "--", color=c, label=key + " T_x/T_z"); axs[1, 1].plot(t, g("Tmy") / g("Tmz"), ":", color=c, label=key + " T_y/T_z")
    axs[1, 2].plot(t, g("Nc"), "-", color=c, label=key + " N_c"); axs[1, 2].plot(t, g("Nth"), ":", color=c, label=key + " N_th")
    out[key] = dict(t=t.tolist(), N=g("N").tolist(), Nc=g("Nc").tolist(), F=g("F").tolist(), F_fit=g("F_fit").tolist(), T_kin=Tkin.tolist(), T_fit=Tfit.tolist(), dF=dFm.tolist(), dF_fit=dFf.tolist())
axs[1, 0].errorbar(dF["t_s"], dF["dF"], dF["dF_err"], fmt="ks", ms=4, label="Fig. 4 data (diss)")
axs[0, 0].set_title("N"); axs[0, 1].set_title("T (nK)"); axs[0, 2].set_title("F"); axs[1, 0].set_title("F - F_eq"); axs[1, 1].set_title("T_x/T_z, T_y/T_z"); axs[1, 2].set_title("N_c, N_th")
for ax in axs.flat: ax.set_xlabel("hold time (s)"); ax.legend(fontsize=7)
fig.suptitle("Phase-space model, hold phase: " + ", ".join(f"{k}: {v}" for k, v in tags.items())); fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "phasespace_hold.png"), dpi=110)
json.dump(out, open(os.path.join(ROOT, "data", "results_phasespace_hold.json"), "w"))
print("saved")
