"""Three-temperature moment model: calibrate on the no-K data, predict the K
series, and report the axis temperatures.  Usage: python scripts/08_three_temperature.py
"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nebec import constants as C
from nebec.twotemp import ThreeTemperatureModel
from nebec.evaporation import make_omega, make_trap_depth, make_gamma_K, integrated_gamma_K, T1_MEAS
from nebec.observables import F_eq_eq1, mu_thermal
from nebec.experiment import omega_cal

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
blue, red, red4 = load("fig2_nodiss.csv"), load("fig2_diss.csv"), load("figS2i_diss.csv")
gk = json.load(open(os.path.join(ROOT, "data", "gammaK_fit.json")))
RATIOS = (1.2, 0.7, 1.2)      # (x, y, z)/w_bar from the quoted final frequencies (70,120,120) Hz; z = gravity

def run(theta, gamma=lambda t: 0.0, N1=55993.0, aniso=(1, 1, 1), t_eval=np.linspace(1, 2, 41), Tmeas="mean"):
    eta1, p, gbg, c_ev = theta
    om = make_omega(1.0); U = make_trap_depth(om, eta1, p=p)
    M = ThreeTemperatureModel(om, RATIOS, U, gamma=lambda t: gamma(t) + gbg, c_ev=c_ev)
    y0 = M.initial(1.0, N1 - 1.0, T1_MEAS, Nc=1.0, aniso=aniso)
    r = M.run(y0, 1.0, 2.0, t_eval)
    return r

def chi2(theta, dat=blue, key="T_mean"):
    if theta[0] < 1.5 or theta[0] > 20 or theta[1] < 0.5 or theta[1] > 6 or theta[2] < 0 or theta[3] < 0.05 or theta[3] > 50:
        return 1e6
    r = run(theta)
    sel = dat["t_s"] > 1.0 + 1e-6
    N = np.interp(dat["t_s"], r["t"], r["N"]); T = np.interp(dat["t_s"], r["t"], r[key] * 1e9); F = np.interp(dat["t_s"], r["t"], r["F"])
    sN = np.maximum(dat["N_err"], 0.03 * dat["N"]); sT = np.maximum(dat["T_err"], 0.03 * dat["T_nK"]); sF = np.maximum(dat["F_err"], 0.02)
    return ((((N - dat["N"]) / sN) ** 2)[sel].sum() + (((T - dat["T_nK"]) / sT) ** 2)[sel].sum() + (((F - dat["F"]) / sF) ** 2)[sel].sum())

best = None
for x0 in ([5.0, 2.5, 0.2, 1.0], [4.0, 3.0, 0.1, 3.0]):
    res = minimize(chi2, x0, method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-3, fatol=0.5))
    print("start", x0, "->", res.x, "chi2", res.fun, flush=True)
    print("start", x0, "->", res.x, "chi2", res.fun)
    if best is None or res.fun < best.fun:
        best = res
theta = best.x
print("BEST eta1=%.2f p=%.2f gamma_bg=%.3f c_ev=%.2f chi2=%.1f" % (*theta, best.fun))
out = {"calibration": dict(eta1=theta[0], p=theta[1], gamma_bg=theta[2], c_ev=theta[3], chi2=best.fun)}
series = {"nodiss": (lambda t: 0.0, 55993.0, blue), "diss": (make_gamma_K(a=gk["c_1x"], k=2.0), 55993.0 * np.exp(-integrated_gamma_K(1.0, gk["c_1x"], 1.0, 2.0)), red),
          "diss4": (make_gamma_K(a=gk["c_4x"], k=2.0), 55993.0 * np.exp(-integrated_gamma_K(1.0, gk["c_4x"], 1.0, 2.0)), red4)}
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for tag, (gam, N1, dat), c, m in zip(series, series.values(), ("b", "r", "m"), ("v", "o", "s")):
    r = run(theta, gamma=gam, N1=N1)
    r["tau_cal"] = r["T_mean"] / np.array([C.Tc_ideal(N, omega_cal(t)) for N, t in zip(r["N"], r["t"])])
    r["F_eq1"] = np.array([F_eq_eq1(T, N, omega_cal(t)) for T, N, t in zip(r["T_mean"], r["N"], r["t"])])
    out[tag] = {k: v.tolist() for k, v in r.items()}
    k = -1
    print(f"{tag}: t=2: N={r['N'][k]:.0f} F={r['F'][k]:.3f} Tx={r['Tx'][k]*1e9:.0f} Ty={r['Ty'][k]*1e9:.0f} Tz={r['Tz'][k]*1e9:.0f} nK  T_mean={r['T_mean'][k]*1e9:.0f}  gamma_coll={r['gamma_coll'][k]:.1f} Hz  z={r['z'][k]:.3f}  data: N={dat['N'][-1]:.0f} T={dat['T_nK'][-1]:.0f} F={dat['F'][-1]:.2f}")
    print("   anisotropy (Tx/Tz, Ty/Tz) along the ramp:", [(round(a / b, 2), round(c_ / b, 2)) for a, b, c_ in zip(r["Tx"][::10], r["Tz"][::10], r["Ty"][::10])])
    axs[0, 0].errorbar(dat["t_s"], dat["N"], dat["N_err"], fmt=m, color=c, ms=4); axs[0, 0].plot(r["t"], r["N"], "-", color=c)
    axs[0, 1].errorbar(dat["t_s"], dat["T_nK"], dat["T_err"], fmt=m, color=c, ms=4)
    axs[0, 1].plot(r["t"], r["Tz"] * 1e9, "-", color=c, label=tag + " T_z (evap.)"); axs[0, 1].plot(r["t"], r["Tx"] * 1e9, "--", color=c, label=tag + " T_x"); axs[0, 1].plot(r["t"], r["Ty"] * 1e9, ":", color=c, label=tag + " T_y")
    axs[0, 2].errorbar(dat["t_s"], dat["F"], dat["F_err"], fmt=m, color=c, ms=4); axs[0, 2].plot(r["t"], r["F"], "-", color=c); axs[0, 2].plot(r["t"], r["F_eq1"], "--", color=c)
    axs[1, 0].plot(r["t"], r["Tx"] / r["Tz"], "--", color=c); axs[1, 0].plot(r["t"], r["Ty"] / r["Tz"], ":", color=c)
    axs[1, 1].plot(r["t"], r["gamma_coll"], "-", color=c); axs[1, 1].plot(r["t"], r["gamma_coll"] / 2.7, "--", color=c)
    axs[1, 2].plot(r["t"], r["z"], "-", color=c)
axs[0, 0].set_title("N"); axs[0, 1].set_title("T per axis (nK)"); axs[0, 1].legend(fontsize=7); axs[0, 2].set_title("F (dashed: Eq.1 at T_mean)")
axs[1, 0].set_title("T_x/T_z (dashed), T_y/T_z (dotted)"); axs[1, 1].set_title("gamma_coll (solid), gamma_coll/2.7 (dashed) [Hz]"); axs[1, 2].set_title("fugacity z")
for ax in axs.flat: ax.set_xlabel("t (s)")
fig.suptitle("Three-temperature moment model (calibrated on blue; T_mean fitted to the data)"); fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "three_temperature_ramp.png"), dpi=110)
json.dump(out, open(os.path.join(ROOT, "data", "results_three_temperature.json"), "w"))
