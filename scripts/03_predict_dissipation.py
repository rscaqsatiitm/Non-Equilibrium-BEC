"""With the calibrated evaporation parameters, predict the dissipation data
(Fig. 2 red, Fig. S2(i) red at 4x) with no new parameters, and compare in the
F-tau plane (Fig. 3).  Also computes the (N,E)-conserving equilibrium
reference along the dissipation trajectory.
Usage: python scripts/03_predict_dissipation.py <calibration json> [n_grid] [gammaK variant: fig1b|empirical]
Outputs: figures/fig2_model.png, figures/fig3_model.png, figures/figS2i_model.png,
         data/results_ramp.json
"""
import sys, os, json, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nebec.experiment import run_ramp, omega_cal
from nebec.equilibrium import closed_system_equilibrium
from nebec.observables import F_eq_eq1
from nebec import constants as C

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
cal = json.load(open(sys.argv[1]))["best"]
n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 320
variant = sys.argv[3] if len(sys.argv) > 3 else "fig1b"
params = dict(eta1=cal["eta1"], p=cal["p"], gamma_bg=cal["gamma_bg"], s_omega=cal["s_omega"], evap_mode=cal.get("evap_mode", "3d"))
gk = json.load(open(os.path.join(ROOT, "data", "gammaK_fit.json")))
if variant == "empirical":
    # gamma_K = c t^2 with c fitted on the atom numbers of each data set
    series = (("nodiss", 0.0, dict()), ("diss", 1.0, dict(gK_a=gk["c_1x"], gK_k=gk["k"])), ("diss4", 1.0, dict(gK_a=gk["c_4x"], gK_k=gk["k"])))
else:
    series = (("nodiss", 0.0, dict()), ("diss", 1.0, dict()), ("diss4", 4.0, dict()))
suffix = f"_{params['evap_mode']}_{variant}"
print("calibration:", params, "variant:", variant)

runs = {}
for tag, df, extra in series:
    t0 = time.time()
    P = dict(params); P.update(extra)
    runs[tag] = run_ramp(P, diss_factor=df, n_grid=n_grid, gaussian_T=True, dt_out=0.05)
    print(tag, "%.0f s" % (time.time() - t0), flush=True)

# (N,E)-equilibrium reference along each trajectory (closed system with the
# model's own DoS at the instantaneous trap frequency)
for tag, r in runs.items():
    Feq, Teq = [], []
    for k in range(len(r["t"])):
        try:
            T, Nc, F = closed_system_equilibrium(r["N"][k], r["E_nK"][k] * C.nK, r["omega"][k])
        except Exception:
            T, Nc, F = np.nan, np.nan, np.nan
        Feq.append(F); Teq.append(T)
    r["F_NE"] = np.array(Feq); r["T_NE"] = np.array(Teq)

def chi2(r, dat, keys=("N", "T_nK", "F")):
    out = {}
    for k in keys:
        m = {"N": r["N"], "T_nK": r["T"] * 1e9, "F": r["F"]}[k]
        mi = np.interp(dat["t_s"], r["t"], m)
        s = np.maximum(dat[k.split("_")[0] + "_err"], {"N": 0.03 * dat["N"], "T_nK": 0.03 * dat["T_nK"], "F": 0.02}[k])
        sel = dat["t_s"] > 1.0 + 1e-6
        out[k] = float((((mi - dat[k]) / s) ** 2)[sel].sum())
        out[k + "_npts"] = int(sel.sum())
    return out

data = {"nodiss": load("fig2_nodiss.csv"), "diss": load("fig2_diss.csv"), "diss4": load("figS2i_diss.csv")}
summary = {tag: chi2(runs[tag], data[tag]) for tag in runs}
print(json.dumps(summary, indent=1))

# ---- Fig. 2-like figure
def fig2like(tags, dats, fname, title):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    for tag, c in zip(tags, ("b", "r")):
        r, dat = runs[tag], dats[tag]
        m = "v" if tag == "nodiss" else "o"
        axs[0, 0].errorbar(dat["t_s"], dat["N"], dat["N_err"], fmt=m, color=c, ms=4, label=tag + " data"); axs[0, 0].plot(r["t"], r["N"], "-", color=c, label=tag + " model")
        axs[0, 1].errorbar(dat["t_s"], dat["T_nK"], dat["T_err"], fmt=m, color=c, ms=4); axs[0, 1].plot(r["t"], r["T"] * 1e9, "-", color=c); axs[0, 1].plot(r["t"], r["T_gauss"] * 1e9, ":", color=c)
        axs[1, 0].errorbar(dat["t_s"], dat["F"], dat["F_err"], fmt=m, color=c, ms=4); axs[1, 0].plot(r["t"], r["F"], "-", color=c); axs[1, 0].plot(r["t"], r["F_eq1"], "--", color=c); axs[1, 0].plot(r["t"], r["F_NE"], "-.", color=c, lw=1)
        axs[1, 1].errorbar(dat["t_s"], dat["mu_nK"], dat["mu_err"], fmt=m, color=c, ms=4); axs[1, 1].plot(r["t"], r["mu_th_nK"], "-", color=c); axs[1, 1].plot(r["t"], r["mu_c_nK"], "--", color=c)
    axs[0, 0].set_ylabel("N"); axs[0, 1].set_ylabel("T (nK)  [solid: model T, dotted: Gaussian-fit T]")
    axs[1, 0].set_ylabel("F  [solid: model; dashed: Eq.(1) at model T; dash-dot: (N,E) equilibrium]"); axs[1, 1].set_ylabel("mu (nK) [solid: thermal (Eq. S7); dashed: BEC]")
    for ax in axs.flat: ax.set_xlabel("t (s)")
    axs[0, 0].legend(fontsize=8); fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", fname), dpi=110)

fig2like(("nodiss", "diss"), data, f"fig2_model{suffix}.png", f"Fig. 2 of the PRL versus the kinetic model (calibrated on blue only; evaporation {params['evap_mode']}, gamma_K {variant})")
fig2like(("nodiss", "diss4"), data, f"figS2i_model{suffix}.png", f"Fig. S2(i) (4x dissipation) versus the kinetic model ({params['evap_mode']}, gamma_K {variant})")

# ---- Fig. 3-like figure (F vs tau with the paper's calibrated omega)
fig, ax = plt.subplots(figsize=(6.5, 5))
op = load("fig3_open_nodiss.csv"); ax.errorbar(op["tau"], op["F"], op["F_err"], op["tau_err"], fmt="o", mfc="none", color="b", ms=4, alpha=0.5, label="no dissipation (data, open)")

with open(os.path.join(D, "fig3_filled_diss.csv")) as f:
    rows = list(csv.DictReader(f))
for row in rows:
    rgb = [float(v) for v in row["rgb"].split(",")]
    ax.errorbar(float(row["tau"]), float(row["F"]), float(row["F_err"]), float(row["tau_err"]), fmt="s" if row["marker"] == "square" else "o", color=rgb, ms=5)
tt = np.linspace(0.5, 1.5, 200)
ax.plot(tt, [max(1 - t**3, 0) for t in tt], "k:", lw=1, label="1 - tau^3")
for tag, c, lab in (("nodiss", "b", "model, no dissipation"), ("diss", "r", "model, dissipation"), ("diss4", "m", "model, 4x dissipation")):
    r = runs[tag]; ax.plot(r["tau"], r["F"], "-", color=c, lw=2, label=lab)
ax.set_xlim(0.5, 1.5); ax.set_ylim(0, 0.8); ax.set_xlabel("tau = T/T_c (paper's calibrated omega)"); ax.set_ylabel("F"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", f"fig3_model{suffix}.png"), dpi=110)

out = {tag: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if not k.startswith("_")} for tag, r in runs.items()}
out["calibration"] = params; out["chi2"] = summary; out["n_grid"] = n_grid; out["variant"] = variant
json.dump(out, open(os.path.join(ROOT, "data", f"results_ramp{suffix}.json"), "w"))
print("done")
