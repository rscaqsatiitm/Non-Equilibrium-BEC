"""Tabulate phase-space results (data/ps_<tag>.json) against the Fig. 2 / S2(i) data.
Usage: python scripts/13_phasespace_compare.py <series: noK|K|K4> <tag> [tag ...]"""
import sys, os, json, csv
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
series = sys.argv[1]
d = load({"noK": "fig2_nodiss.csv", "K": "fig2_diss.csv", "K4": "figS2i_diss.csv"}[series])
sel = d["t_s"] >= 1.0
td, Nd, Td, Fd = d["t_s"][sel], d["N"][sel], d["T_nK"][sel], d["F"][sel]
print(f"series {series}: data at t = {td}")
for tag in sys.argv[2:]:
    rows = json.load(open(os.path.join(ROOT, "data", f"ps_{tag}.json")))
    t = np.array([r["t_lab"] for r in rows]); g = lambda k: np.array([r[k] for r in rows])
    ip = lambda k: np.interp(td, t, g(k))
    N, F, Ff = ip("N"), ip("F"), ip("F_fit")
    Tk = 1e9 * (ip("Tmx") + ip("Tmz")) / 2; Tf = 1e9 * (ip("T_fit_x") + ip("T_fit_z")) / 2
    print(f"\n== {tag}")
    print(" t     N_data  N_mod |  T_data T_kin(xz) T_fit |  F_data F_mod F_fit")
    for i in range(len(td)):
        print(f" {td[i]:.2f} {Nd[i]:7.0f} {N[i]:6.0f} | {Td[i]:6.0f} {Tk[i]:6.0f} {Tf[i]:6.0f} | {Fd[i]:5.2f} {F[i]:5.2f} {Ff[i]:5.2f}")
    rN = np.sqrt(np.mean(np.log(N / Nd) ** 2)); rT = np.sqrt(np.mean(np.log(Tk / Td) ** 2)); rF = np.sqrt(np.mean((F - Fd) ** 2))
    print(f" rms log(N/Ndata) {rN:.3f}, rms log(Tkin/Tdata) {rT:.3f}, rms (F-Fdata) {rF:.3f}")
