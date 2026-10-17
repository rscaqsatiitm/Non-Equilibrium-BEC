"""Equilibrium at fixed atom number and energy for every measured state of
Figs. 2 (with K, t >= 1.5 s) and S2(ii) (hold, with and without K).
For each digitized (N, T, F): thermal energy of an ideal Bose gas with the
fugacity that gives the measured thermal number at the measured T, condensate
energy (5/7) mu_TF N_c; then the temperature T' at which a saturated cloud plus
condensate has the same N and E, with the condensed fraction from Eq. (1) of
the PRL (interaction and finite-size shifts) or from the ideal gas.
Sensitivity: T +- 3% (digitization and fit error) and trap frequency x 0.85
(measured line instead of calibrated line, Fig. S1).
Outputs: data/results_ne_equilibrium.json, figures/ne_equilibrium.png."""
import sys, os, csv, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nebec import constants as C
from nebec.observables import F_eq_eq1
ROOT = os.path.join(os.path.dirname(__file__), "..")
z3, z4 = 1.2020569, 1.0823232
def g(s, z): return sum(z**k / k**s for k in range(1, 2000))
def load(name):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "digitized", name))))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
def F_ideal(T, N, w):
    return max(0.0, 1 - min(z3 * (C.kB * T / (C.hbar * w)) ** 3, N) / N)
def energy(N, T, F, w):
    Nth, Nc = N * (1 - F), N * F
    Nsat = z3 * (C.kB * T / (C.hbar * w)) ** 3
    z = 1.0 if Nth >= Nsat else brentq(lambda z: g(3, z) / z3 - Nth / Nsat, 1e-9, 1.0)
    return 3 * C.kB * T * Nth * g(4, z) / g(3, z) + (5 / 7) * C.mu_TF(Nc, w) * Nc, z
def ne_equilibrium(N, T, F, w, Ffun):
    E, z = energy(N, T, F, w)
    def resid(Tp):
        Fp = Ffun(Tp, N, w); Nth = N * (1 - Fp)
        return 3 * C.kB * Tp * Nth * z4 / z3 + (5 / 7) * C.mu_TF(N * Fp, w) * N * Fp - E
    Tp = brentq(resid, 1e-9, 5 * T)
    return Ffun(Tp, N, w), Tp, z
series = [("K, ramp", "fig2_diss.csv", None, 1.5), ("K, hold", "figS2ii_diss.csv", 2 * np.pi * 99.2, 0.0), ("no K, hold", "figS2ii_nodiss.csv", 2 * np.pi * 99.2, 0.0)]
out = {}
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, (label, name, wb, tmin) in zip(axs, series):
    d = load(name); rows = []
    for i in range(len(d["t_s"])):
        t, N, T, F = d["t_s"][i], d["N"][i], d["T_nK"][i] * 1e-9, d["F"][i]
        if t < tmin: continue
        w = wb if wb else 2 * np.pi * (192 - 56 * t)
        r = dict(t=t, N=N, T=T * 1e9, F=F)
        r["F_eq1_T"] = F_eq_eq1(T, N, w); r["F_ideal_T"] = F_ideal(T, N, w)
        r["F_NE_eq1"], Tp, r["z"] = ne_equilibrium(N, T, F, w, F_eq_eq1); r["T_NE_eq1"] = Tp * 1e9
        r["F_NE_ideal"], Tp2, _ = ne_equilibrium(N, T, F, w, F_ideal); r["T_NE_ideal"] = Tp2 * 1e9
        r["F_NE_eq1_Tp3"] = ne_equilibrium(N, 1.03 * T, F, w, F_eq_eq1)[0]; r["F_NE_eq1_Tm3"] = ne_equilibrium(N, 0.97 * T, F, w, F_eq_eq1)[0]
        r["F_NE_eq1_w085"] = ne_equilibrium(N, T, F, 0.85 * w, F_eq_eq1)[0]
        rows.append(r)
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    stat = lambda a: f"{a.mean():+.3f} +- {a.std():.3f}"
    print(f"\n== {label} ({len(rows)} points)")
    print("  t      N    T_meas  F   |  z   F_eq1(T)  F_NE(eq1)  T'  | F_NE(ideal)  T'  | dF_NE at T+3%, T-3%, w*0.85")
    for r in rows:
        print(f" {r['t']:5.2f} {r['N']:6.0f} {r['T']:5.0f} {r['F']:5.2f} | {r['z']:4.2f}  {r['F_eq1_T']:5.2f}    {r['F_NE_eq1']:5.2f}   {r['T_NE_eq1']:4.0f} |   {r['F_NE_ideal']:5.2f}    {r['T_NE_ideal']:4.0f} | {r['F']-r['F_NE_eq1_Tp3']:+.2f} {r['F']-r['F_NE_eq1_Tm3']:+.2f} {r['F']-r['F_NE_eq1_w085']:+.2f}")
    print(f"  F - F_eq1(T_meas): {stat(R['F'] - R['F_eq1_T'])};  F - F_NE(eq1): {stat(R['F'] - R['F_NE_eq1'])};  F - F_NE(ideal): {stat(R['F'] - R['F_NE_ideal'])}")
    print(f"  T_meas/T'(eq1): {(R['T']/R['T_NE_eq1']).min():.2f} to {(R['T']/R['T_NE_eq1']).max():.2f};  sensitivity of F - F_NE: T+3% {stat(R['F']-R['F_NE_eq1_Tp3'])}, T-3% {stat(R['F']-R['F_NE_eq1_Tm3'])}, w x0.85 {stat(R['F']-R['F_NE_eq1_w085'])}")
    out[label] = {k: v.tolist() for k, v in R.items()}
    ax.plot(R["t"], R["F"], "ko", label="F measured"); ax.plot(R["t"], R["F_eq1_T"], "rs--", label="F_eq Eq. (1) at T_meas"); ax.plot(R["t"], R["F_NE_eq1"], "b^-", label="F at fixed (N, E), Eq. (1)"); ax.plot(R["t"], R["F_NE_ideal"], "c:", label="F at fixed (N, E), ideal")
    ax.set_title(label); ax.set_xlabel("t (s)"); ax.set_ylabel("condensed fraction"); ax.legend(fontsize=7); ax.set_ylim(-0.02, 0.9)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "ne_equilibrium.png"), dpi=130)
json.dump(out, open(os.path.join(ROOT, "data", "results_ne_equilibrium.json"), "w"), indent=1)
print("\nsaved figures/ne_equilibrium.png, data/results_ne_equilibrium.json")
