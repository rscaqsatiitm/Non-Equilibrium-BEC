"""Calibrate the evaporation parameters (eta1, p, gamma_bg[, s_omega]) on the
no-dissipation data of Fig. 2 (N, T, F versus time).  Writes
data/calibration_ramp.json and figures/calibration_ramp.png.
Usage: python scripts/02_calibrate_evaporation.py [s_omega] [n_grid] [maxiter]
"""
import sys, os, json, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.optimize import minimize
from nebec.experiment import run_ramp

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
blue = load("fig2_nodiss.csv")
s_omega = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 200
maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 80
evap_mode = sys.argv[4] if len(sys.argv) > 4 else "3d"

# error floors: 3% on N and T, 0.02 on F (digitized SEMs are smaller than the
# systematic differences between runs)
sN = np.maximum(blue["N_err"], 0.03 * blue["N"])
sT = np.maximum(blue["T_err"], 0.03 * blue["T_nK"])
sF = np.maximum(blue["F_err"], 0.02)
sel = blue["t_s"] > 1.0 + 1e-6      # t = 1.0 is the initial condition

def chi2_of(res):
    N = np.interp(blue["t_s"], res["t"], res["N"]); T = np.interp(blue["t_s"], res["t"], res["T"] * 1e9); F = np.interp(blue["t_s"], res["t"], res["F"])
    cN = (((N - blue["N"]) / sN) ** 2)[sel].sum(); cT = (((T - blue["T_nK"]) / sT) ** 2)[sel].sum(); cF = (((F - blue["F"]) / sF) ** 2)[sel].sum()
    return cN + cT + cF, (cN, cT, cF)

history = []
def cost(theta):
    eta1, p, gbg = theta
    if eta1 < 2 or eta1 > 20 or p < 0.5 or p > 6 or gbg < 0 or gbg > 1.0:
        return 1e6
    t0 = time.time()
    res = run_ramp(dict(eta1=eta1, p=p, gamma_bg=gbg, s_omega=s_omega, evap_mode=evap_mode), diss_factor=0.0, n_grid=n_grid)
    c, parts = chi2_of(res)
    history.append((list(map(float, theta)), float(c)))
    print(f"eta1={eta1:.3f} p={p:.3f} gbg={gbg:.4f} -> chi2={c:.2f} (N {parts[0]:.1f}, T {parts[1]:.1f}, F {parts[2]:.1f})  [{time.time()-t0:.1f} s]", flush=True)
    return c

x0 = np.array([5.66, 2.8, 0.32]) if evap_mode == "3d" else np.array([3.58, 2.78, 0.21])
simplex = np.array([x0, x0 + [1.5, 0, 0], x0 + [0, 1.0, 0], x0 + [0, 0, 0.1]])
r = minimize(cost, x0, method="Nelder-Mead", options=dict(initial_simplex=simplex, maxiter=maxiter, xatol=0.02, fatol=0.5))
best = dict(eta1=float(r.x[0]), p=float(r.x[1]), gamma_bg=float(r.x[2]), s_omega=s_omega, evap_mode=evap_mode, chi2=float(r.fun), n_grid=n_grid)
print("BEST", best)
with open(os.path.join(ROOT, "data", f"calibration_ramp_s{s_omega:.2f}_{evap_mode}.json"), "w") as f:
    json.dump(dict(best=best, history=history), f, indent=1)
