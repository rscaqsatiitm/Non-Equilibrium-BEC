"""Hold phase: with the trap calibrated on the blue (no-dissipation) hold data,
predict the red (post-dissipation) hold from its measured initial state and
compare delta F = F - F_eq1(tau) with Fig. 4.
Usage: python scripts/04b_hold_predict.py <calibration_hold json> [n_grid]
Outputs: figures/hold_model.png, figures/fig4_model.png, data/results_hold.json
"""
import sys, os, json, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nebec.experiment import run_hold
from nebec.observables import F_eq_eq1
from nebec.equilibrium import closed_system_equilibrium
from nebec import constants as C

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
blue, red, fig4 = load("figS2ii_nodiss.csv"), load("figS2ii_diss.csv"), load("fig4_deltaF.csv")
cal = json.load(open(sys.argv[1]))["best"]
n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 320
om_cal = C.kB * blue["T_nK"][0] * 1e-9 / (0.94 * C.hbar * blue["N"][0] ** (1 / 3) * blue["tau"][0])
U_hold = cal["eta_hold"] * C.kB * blue["T_nK"][0] * 1e-9        # same physical trap for both series
print("omega_cal/2pi = %.1f Hz, U_hold/k_B = %.0f nK, s_omega = %.3f" % (om_cal / 2 / np.pi, U_hold / C.nK, cal["s_omega"]))

runs = {}
for tag, d in (("nodiss", blue), ("diss", red)):
    eta_eff = U_hold / (C.kB * d["T_nK"][0] * 1e-9)
    t0 = time.time()
    r = run_hold(d["N"][0], d["T_nK"][0] * 1e-9, d["F"][0], cal["s_omega"] * om_cal, eta_eff, params=dict(gamma_bg=cal["gamma_bg"], evap_mode=cal.get("evap_mode", "3d")), n_grid=n_grid, gaussian_T=True)
    # paper-style analysis with the constant calibrated omega
    r["tau_cal"] = r["T"] / C.Tc_ideal(r["N"], om_cal)
    r["F_eq1_cal"] = np.array([F_eq_eq1(T, N, om_cal) for T, N in zip(r["T"], r["N"])])
    r["dF"] = r["F"] - r["F_eq1_cal"]
    FNE = []
    for k in range(len(r["t"])):
        try:
            FNE.append(closed_system_equilibrium(r["N"][k], r["E_nK"][k] * C.nK, r["omega"][k])[2])
        except Exception:
            FNE.append(np.nan)
    r["F_NE"] = np.array(FNE)
    runs[tag] = r
    print(tag, "%.0f s" % (time.time() - t0), flush=True)
    for k in range(0, len(r["t"]), 5):
        print(f"  t={r['t'][k]:.1f} N={r['N'][k]:.0f} F={r['F'][k]:.3f} T={r['T'][k]*1e9:.1f} tau_cal={r['tau_cal'][k]:.3f} dF={r['dF'][k]:.3f} F_NE={r['F_NE'][k]:.3f} mu_th={r['mu_th_nK'][k]:.1f} mu_c={r['mu_c_nK'][k]:.1f}")

# experimental delta F for the red series (paper's definition) from the digitized S2(ii) data
dF_exp = red["F"] - np.array([F_eq_eq1(T * 1e-9, N, om_cal) for T, N in zip(red["T_nK"], red["N"])])
# exponential fit of the model dF after 1.5 s (as in Fig. 4)
r = runs["diss"]; sel = r["t"] >= 1.5
try:
    popt, pcov = curve_fit(lambda t, A, tau, c: c + A * np.exp(-(t - 1.5) / tau), r["t"][sel], r["dF"][sel], p0=[0.1, 3.0, 0.1])
    print("model dF relaxation after 1.5 s: tau = %.2f s (offset %.3f)" % (popt[1], popt[2]))
except Exception as e:
    popt = [np.nan] * 3; print("fit failed", e)
fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
for tag, d, c, m in (("nodiss", blue, "b", "v"), ("diss", red, "r", "o")):
    r = runs[tag]
    axs[0, 0].errorbar(d["t_s"], d["N"], d["N_err"], fmt=m, color=c, ms=4); axs[0, 0].plot(r["t"], r["N"], "-", color=c)
    axs[0, 1].errorbar(d["t_s"], d["T_nK"], d["T_err"], fmt=m, color=c, ms=4); axs[0, 1].plot(r["t"], r["T"] * 1e9, "-", color=c); axs[0, 1].plot(r["t"], r["T_gauss"] * 1e9, ":", color=c)
    axs[0, 2].errorbar(d["t_s"], d["tau"], d["tau_err"], fmt=m, color=c, ms=4); axs[0, 2].plot(r["t"], r["tau_cal"], "-", color=c)
    axs[1, 0].errorbar(d["t_s"], d["F"], d["F_err"], fmt=m, color=c, ms=4); axs[1, 0].plot(r["t"], r["F"], "-", color=c); axs[1, 0].plot(r["t"], r["F_eq1_cal"], "--", color=c); axs[1, 0].plot(r["t"], r["F_NE"], "-.", color=c, lw=1)
    axs[1, 1].plot(r["t"], r["mu_th_nK"], "-", color=c); axs[1, 1].plot(r["t"], r["mu_c_nK"], "--", color=c)
axs[1, 2].errorbar(fig4["t_s"], fig4["dF"], fig4["dF_err"], fmt="o", color="r", ms=4, label="Fig. 4 data")
axs[1, 2].plot(red["t_s"], dF_exp, "x", color="k", ms=5, label="from digitized S2(ii)")
axs[1, 2].plot(runs["diss"]["t"], runs["diss"]["dF"], "r-", label="model (red)"); axs[1, 2].plot(runs["nodiss"]["t"], runs["nodiss"]["dF"], "b-", label="model (blue)")
axs[1, 2].legend(fontsize=8)
for ax, lab in zip(axs.flat, ("N", "T (nK) [dotted: Gaussian-fit T]", "tau (calibrated omega)", "F [dashed Eq.(1), dash-dot (N,E) eq.]", "mu (nK) thermal / BEC", "delta F")):
    ax.set_title(lab); ax.set_xlabel("t (s)")
fig.suptitle("Hold phase: trap calibrated on blue; red predicted from its initial state"); fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", f"hold_model_{cal.get('evap_mode', '3d')}.png"), dpi=110)
out = {tag: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if not k.startswith("_")} for tag, r in runs.items()}
out["calibration"] = cal; out["dF_exp_red"] = dF_exp.tolist(); out["model_dF_tau_s"] = float(popt[1]); out["n_grid"] = n_grid
json.dump(out, open(os.path.join(ROOT, "data", f"results_hold_{cal.get('evap_mode', '3d')}.json"), "w"))
print("done")
