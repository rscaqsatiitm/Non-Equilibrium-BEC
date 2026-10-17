"""Hold phase (Fig. 4 / S2(ii)): calibrate the frozen trap (eta_hold, omega_hold
scale, gamma_bg) on the blue no-dissipation hold data, then predict the red
(post-dissipation) hold from its measured initial state.  Output:
data/calibration_hold.json, figures/hold.png
Usage: python scripts/04_hold_calibration.py [n_grid] [maxiter]
"""
import sys, os, json, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.optimize import minimize
from nebec.experiment import run_hold
from nebec import constants as C

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
blue, red = load("figS2ii_nodiss.csv"), load("figS2ii_diss.csv")
n_grid = int(sys.argv[1]) if len(sys.argv) > 1 else 200
maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 60
evap_mode = sys.argv[3] if len(sys.argv) > 3 else "3d"

# trap at the end of the Fig.-4 ramp: the supplement's Fig. S2 calibration gives
# a constant omega during the hold; the measured tau and T at t=0 (blue) give
# omega_cal = k_B T/(0.94 hbar N^{1/3} tau)
def omega_from_tau(d, i=0):
    return C.kB * d["T_nK"][i] * 1e-9 / (0.94 * C.hbar * d["N"][i] ** (1 / 3) * d["tau"][i])
om_cal_blue = omega_from_tau(blue); om_cal_red = omega_from_tau(red)
print("omega_cal/2pi from (T,N,tau) at t=0: blue %.1f Hz, red %.1f Hz" % (om_cal_blue / 2 / np.pi, om_cal_red / 2 / np.pi))

def series(d):
    return d["t_s"], d["N"], np.maximum(d["N_err"], 0.03 * d["N"]), d["T_nK"], np.maximum(d["T_err"], 0.03 * d["T_nK"]), d["F"], np.maximum(d["F_err"], 0.02)

def chi2_of(res, d):
    t, N, sN, T, sT, F, sF = series(d)
    sel = t > 1e-3
    Nm = np.interp(t, res["t"], res["N"]); Tm = np.interp(t, res["t"], res["T"] * 1e9); Fm = np.interp(t, res["t"], res["F"])
    parts = ((((Nm - N) / sN) ** 2)[sel].sum(), (((Tm - T) / sT) ** 2)[sel].sum(), (((Fm - F) / sF) ** 2)[sel].sum())
    return sum(parts), parts

def run(theta, d, om_cal):
    eta, s_om, gbg = theta
    return run_hold(d["N"][0], d["T_nK"][0] * 1e-9, d["F"][0], s_om * om_cal, eta, params=dict(gamma_bg=gbg, evap_mode=evap_mode), n_grid=n_grid)

history = []
def cost(theta):
    eta, s_om, gbg = theta
    if eta < 2 or eta > 25 or s_om < 0.6 or s_om > 1.3 or gbg < 0 or gbg > 1:
        return 1e6
    t0 = time.time()
    res = run(theta, blue, om_cal_blue)
    c, parts = chi2_of(res, blue)
    history.append((list(map(float, theta)), float(c)))
    print(f"eta={eta:.3f} s_om={s_om:.3f} gbg={gbg:.4f} -> chi2={c:.2f} {tuple(round(p,1) for p in parts)} [{time.time()-t0:.1f} s]", flush=True)
    return c

x0 = np.array([8.0, 1.0, 0.05]) if evap_mode == "3d" else np.array([5.0, 1.05, 0.1])
simplex = np.array([x0, x0 + [2.0, 0, 0], x0 + [0, 0.15, 0], x0 + [0, 0, 0.1]])
r = minimize(cost, x0, method="Nelder-Mead", options=dict(initial_simplex=simplex, maxiter=maxiter, xatol=0.02, fatol=0.5))
best = dict(eta_hold=float(r.x[0]), s_omega=float(r.x[1]), gamma_bg=float(r.x[2]), chi2=float(r.fun), n_grid=n_grid, evap_mode=evap_mode,
            omega_cal_blue_Hz=float(om_cal_blue / 2 / np.pi), omega_cal_red_Hz=float(om_cal_red / 2 / np.pi))
print("BEST", best)
with open(os.path.join(ROOT, "data", f"calibration_hold_{evap_mode}.json"), "w") as f:
    json.dump(dict(best=best, history=history), f, indent=1)
