"""Trial run of the ramp protocol with guessed parameters, overlaid on the
digitized Fig. 2 data."""
import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nebec.experiment import run_ramp
from nebec import constants as C

D = os.path.join(os.path.dirname(__file__), "..", "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rd = csv.DictReader(f); rows = list(rd)
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
blue, red = load("fig2_nodiss.csv"), load("fig2_diss.csv")

params = dict(eta1=float(sys.argv[1]) if len(sys.argv) > 1 else 8.0, p=float(sys.argv[2]) if len(sys.argv) > 2 else 2.0,
              s_omega=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0, gamma_bg=float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
t0 = time.time()
res = {}
for tag, df in (("nodiss", 0.0), ("diss", 1.0)):
    r = run_ramp(params, diss_factor=df, gaussian_T=True)
    res[tag] = r
    print(tag, "done in %.1f s, nfev=%d" % (time.time() - t0, r["_sol"].nfev)); t0 = time.time()
    for k in range(0, len(r["t"]), 4):
        print(f"  t={r['t'][k]:.2f} N={r['N'][k]:.0f} Nc={r['Nc'][k]:.0f} F={r['F'][k]:.3f} T={r['T'][k]*1e9:.1f} T_G={r['T_gauss'][k]*1e9:.1f} mu_c={r['mu_c_nK'][k]:.1f} mu_th={r['mu_th_nK'][k]:.1f} tau={r['tau'][k]:.3f} Feq1={r['F_eq1'][k]:.3f}")
fig, axs = plt.subplots(2, 2, figsize=(10, 7))
for tag, c, m, dat in (("nodiss", "b", "v", blue), ("diss", "r", "o", red)):
    r = res[tag]
    axs[0, 0].errorbar(dat["t_s"], dat["N"], dat["N_err"], fmt=m, color=c, ms=4); axs[0, 0].plot(r["t"], r["N"], "-", color=c)
    axs[0, 1].errorbar(dat["t_s"], dat["T_nK"], dat["T_err"], fmt=m, color=c, ms=4); axs[0, 1].plot(r["t"], r["T"] * 1e9, "-", color=c); axs[0, 1].plot(r["t"], r["T_gauss"] * 1e9, ":", color=c)
    axs[1, 0].errorbar(dat["t_s"], dat["F"], dat["F_err"], fmt=m, color=c, ms=4); axs[1, 0].plot(r["t"], r["F"], "-", color=c); axs[1, 0].plot(r["t"], r["F_eq1"], "--", color=c)
    axs[1, 1].errorbar(dat["t_s"], dat["mu_nK"], dat["mu_err"], fmt=m, color=c, ms=4); axs[1, 1].plot(r["t"], r["mu_th_nK"], "-", color=c); axs[1, 1].plot(r["t"], r["mu_c_nK"], "--", color=c)
for ax, lab in zip(axs.flat, ("N", "T (nK): solid T_E, dotted T_Gauss", "F (dashed: Eq.1 at model T)", "mu (nK): solid thermal, dashed BEC")):
    ax.set_title(lab); ax.set_xlabel("t (s)")
fig.suptitle("trial: " + str(params)); fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "trial_run.png"), dpi=90)
