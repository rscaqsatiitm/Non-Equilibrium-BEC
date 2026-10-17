"""Sensitivity of the dissipation prediction to the model inputs.
Usage: python scripts/05_sensitivity.py <calibration json> [n_grid]
Output: data/sensitivity.json, figures/sensitivity.png
"""
import sys, os, json, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nebec.experiment import run_ramp
from nebec import evaporation as EV

ROOT = os.path.join(os.path.dirname(__file__), "..")
cal = json.load(open(sys.argv[1]))["best"]
n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 240
gk = json.load(open(os.path.join(ROOT, "data", "gammaK_fit.json")))
base = dict(eta1=cal["eta1"], p=cal["p"], gamma_bg=cal["gamma_bg"], s_omega=cal["s_omega"], evap_mode=cal.get("evap_mode", "3d"), gK_a=gk["c_1x"], gK_k=gk["k"])
cases = {
    "reference": dict(),
    "C12 x0.5": dict(c12_scale=0.5), "C12 x2": dict(c12_scale=2.0),
    "C22 x0.5": dict(c22_scale=0.5), "C22 x2": dict(c22_scale=2.0),
    "eta1 -10%": dict(eta1=0.9 * base["eta1"]), "eta1 +10%": dict(eta1=1.1 * base["eta1"]),
    "s_omega 0.9": dict(s_omega=0.9), "L3 = 0": dict(L3_scale=0.0), "L3 x2": dict(L3_scale=2.0),
    "T1 +10%": dict(T1=1.1 * EV.T1_MEAS), "c +20%": "a+", "c -20%": "a-", "evap 3d": dict(evap_mode="3d"), "gamma_K = a t (Fig. 1b)": dict(gK_a=gk["a_fig1b"], gK_k=1.0),
}
res = {}
for name, mod in cases.items():
    P = dict(base)
    df = 1.0
    if mod == "a+": df = 1.2
    elif mod == "a-": df = 0.8
    else: P.update(mod)
    t0 = time.time()
    r = run_ramp(P, diss_factor=df, n_grid=n_grid)
    res[name] = dict(t=r["t"].tolist(), N=r["N"].tolist(), F=r["F"].tolist(), T=(r["T"] * 1e9).tolist(), tau=r["tau"].tolist(), F_eq1=r["F_eq1"].tolist())
    k = -1
    print(f"{name:12s}: t=2: N={r['N'][k]:.0f} F={r['F'][k]:.3f} T={r['T'][k]*1e9:.1f} nK  tau={r['tau'][k]:.3f}  F-F_eq1={r['F'][k]-r['F_eq1'][k]:.3f}  [{time.time()-t0:.0f} s]", flush=True)
json.dump(res, open(os.path.join(ROOT, "data", f"sensitivity_{base['evap_mode']}.json"), "w"))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for name, r in res.items():
    lw = 2.5 if name == "reference" else 1
    axs[0].plot(r["t"], r["F"], lw=lw, label=name); axs[1].plot(r["tau"], r["F"], lw=lw)
axs[0].set_xlabel("t (s)"); axs[0].set_ylabel("F (dissipation 1x)"); axs[0].legend(fontsize=7)
axs[1].set_xlabel("tau"); axs[1].set_ylabel("F"); axs[1].set_xlim(0.7, 1.2)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", f"sensitivity_{base['evap_mode']}.png"), dpi=110)
