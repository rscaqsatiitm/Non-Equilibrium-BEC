"""Collect the numbers used in the manuscript from the result files and write
manuscript/numbers.json plus a markdown ledger table (manuscript/ledger.md).
Usage: python scripts/06_summary.py <results_ramp json> <results_hold json> [sensitivity json]
"""
import sys, os, json, csv
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "digitized")
def load(name):
    with open(os.path.join(D, name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
ramp = json.load(open(sys.argv[1])); hold = json.load(open(sys.argv[2]))
sens = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
num = {"ramp_file": os.path.basename(sys.argv[1]), "hold_file": os.path.basename(sys.argv[2]), "calibration": ramp["calibration"], "chi2": ramp["chi2"], "variant": ramp.get("variant")}
def at(r, t, key):
    return float(np.interp(t, r["t"], r[key]))
data = {"nodiss": load("fig2_nodiss.csv"), "diss": load("fig2_diss.csv"), "diss4": load("figS2i_diss.csv")}
for tag in ("nodiss", "diss", "diss4"):
    r = ramp[tag]; d = data[tag]
    tlast = 2.0
    i = int(np.argmin(np.abs(d["t_s"] - tlast)))
    num[tag] = dict(t=tlast, N_model=at(r, tlast, "N"), N_data=float(d["N"][i]), T_model_nK=at(r, tlast, "T") * 1e9, T_gauss_model_nK=at(r, tlast, "T_gauss") * 1e9,
                    T_data_nK=float(d["T_nK"][i]), F_model=at(r, tlast, "F"), F_data=float(d["F"][i]), F_eq1_model=at(r, tlast, "F_eq1"), F_NE_model=at(r, tlast, "F_NE"),
                    tau_model=at(r, tlast, "tau"), mu_th_model_nK=at(r, tlast, "mu_th_nK"), mu_c_model_nK=at(r, tlast, "mu_c_nK"),
                    maxabs_dF_model_vs_data=float(np.max(np.abs(np.interp(d["t_s"], r["t"], r["F"]) - d["F"]))),
                    rms_dF_model_vs_data=float(np.sqrt(np.mean((np.interp(d["t_s"], r["t"], r["F"]) - d["F"]) ** 2))),
                    rms_relT=float(np.sqrt(np.mean(((np.interp(d["t_s"], r["t"], r["T"]) * 1e9 - d["T_nK"]) / d["T_nK"]) ** 2))),
                    rms_relN=float(np.sqrt(np.mean(((np.interp(d["t_s"], r["t"], r["N"]) - d["N"]) / d["N"]) ** 2))))
    # paper-style delta F of the model at t = 2 and of the data (Eq.1 at the measured T,N with omega_cal)
    from nebec.observables import F_eq_eq1
    from nebec.experiment import omega_cal
    num[tag]["dF_data_t2"] = float(d["F"][i] - F_eq_eq1(d["T_nK"][i] * 1e-9, d["N"][i], omega_cal(d["t_s"][i])))
    num[tag]["dF_model_t2"] = num[tag]["F_model"] - num[tag]["F_eq1_model"]
    num[tag]["F_NE_minus_Feq1_t2"] = num[tag]["F_NE_model"] - num[tag]["F_eq1_model"]
# hold
fig4 = load("fig4_deltaF.csv")
for tag in ("nodiss", "diss"):
    r = hold[tag]
    num["hold_" + tag] = dict(F0=r["F"][0], F_end=r["F"][-1], T0_nK=r["T"][0] * 1e9, T_end_nK=r["T"][-1] * 1e9, N0=r["N"][0], N_end=r["N"][-1],
                              dF_min=float(np.min(r["dF"])), dF_end=float(r["dF"][-1]), dF_at_1p5=at(r, 1.5, "dF"), F_NE0=r["F_NE"][0], F_eq1_0=r["F_eq1_cal"][0], tau_end=r["tau_cal"][-1])
num["hold_model_dF_tau_s"] = hold.get("model_dF_tau_s"); num["hold_calibration"] = hold["calibration"]
num["fig4_dF_data_range"] = [float(fig4["dF"].min()), float(fig4["dF"].max())]
if sens:
    num["sensitivity_F_t2"] = {k: v["F"][-1] for k, v in sens.items()}
    num["sensitivity_T_t2"] = {k: v["T"][-1] for k, v in sens.items()}
json.dump(num, open(os.path.join(ROOT, "manuscript", "numbers.json"), "w"), indent=1)
lines = ["| quantity | value | type | pointer |", "|---|---|---|---|"]
def row(q, v, typ, ptr): lines.append(f"| {q} | {v} | {typ} | {ptr} |")
c = ramp["calibration"]
row("eta_1 (trap depth / k_B T_1 at t=1 s)", f"{c['eta1']:.2f}", "SIM (fit)", num["ramp_file"])
row("p (depth exponent)", f"{c['p']:.2f}", "SIM (fit)", num["ramp_file"])
row("gamma_bg (1/s)", f"{c['gamma_bg']:.3f}", "SIM (fit)", num["ramp_file"])
for tag in ("nodiss", "diss", "diss4"):
    n = num[tag]
    row(f"{tag}: N(2 s) model / data", f"{n['N_model']:.0f} / {n['N_data']:.0f}", "SIM / digitized", "results + data/digitized")
    row(f"{tag}: T(2 s) model / data (nK)", f"{n['T_model_nK']:.0f} / {n['T_data_nK']:.0f}", "SIM / digitized", "")
    row(f"{tag}: F(2 s) model / data", f"{n['F_model']:.3f} / {n['F_data']:.3f}", "SIM / digitized", "")
    row(f"{tag}: rms |F_model - F_data|", f"{n['rms_dF_model_vs_data']:.3f}", "SIM", "")
    row(f"{tag}: F_eq1(model T) and F_NE at 2 s", f"{n['F_eq1_model']:.3f}, {n['F_NE_model']:.3f}", "SIM", "")
    row(f"{tag}: delta F data / model at 2 s", f"{n['dF_data_t2']:.3f} / {n['dF_model_t2']:.3f}", "digitized / SIM", "")
for tag in ("nodiss", "diss"):
    h = num["hold_" + tag]
    row(f"hold {tag}: F 0 -> 4 s", f"{h['F0']:.3f} -> {h['F_end']:.3f}", "SIM", num["hold_file"])
    row(f"hold {tag}: T 0 -> 4 s (nK)", f"{h['T0_nK']:.0f} -> {h['T_end_nK']:.0f}", "SIM", "")
    row(f"hold {tag}: delta F at 1.5 s, 4 s", f"{h['dF_at_1p5']:.3f}, {h['dF_end']:.3f}", "SIM", "")
row("Fig. 4 delta F range (data)", f"{num['fig4_dF_data_range'][0]:.2f} to {num['fig4_dF_data_range'][1]:.2f}", "digitized", "fig4_deltaF.csv")
open(os.path.join(ROOT, "manuscript", "ledger.md"), "w").write("\n".join(lines) + "\n")
# LaTeX macros for the manuscript
def mac(name, val):
    return "\\newcommand{\\%s}{%s}" % (name, val)
m = [mac("etaone", f"{c['eta1']:.1f}"), mac("pexp", f"{c['p']:.1f}"), mac("gbg", f"{c['gamma_bg']:.2f}")]
for tag, pre in (("nodiss", "nd"), ("diss", "ds"), ("diss4", "dq")):
    n = num[tag]
    m += [mac(pre + "Nmod", f"{n['N_model']/1e3:.1f}"), mac(pre + "Ndat", f"{n['N_data']/1e3:.1f}"), mac(pre + "Tmod", f"{n['T_model_nK']:.0f}"), mac(pre + "TGmod", f"{n['T_gauss_model_nK']:.0f}"),
          mac(pre + "Tdat", f"{n['T_data_nK']:.0f}"), mac(pre + "Fmod", f"{n['F_model']:.2f}"), mac(pre + "Fdat", f"{n['F_data']:.2f}"), mac(pre + "Frms", f"{n['rms_dF_model_vs_data']:.2f}"),
          mac(pre + "Trms", f"{100*n['rms_relT']:.0f}"), mac(pre + "Nrms", f"{100*n['rms_relN']:.0f}"), mac(pre + "Feqone", f"{n['F_eq1_model']:.2f}"), mac(pre + "FNE", f"{n['F_NE_model']:.2f}"),
          mac(pre + "dFdat", f"{n['dF_data_t2']:.2f}"), mac(pre + "dFmod", f"{n['dF_model_t2']:.2f}"), mac(pre + "taumod", f"{n['tau_model']:.2f}"), mac(pre + "muth", f"{n['mu_th_model_nK']:.0f}"), mac(pre + "muc", f"{n['mu_c_model_nK']:.0f}")]
for tag, pre in (("nodiss", "hnd"), ("diss", "hds")):
    h = num["hold_" + tag]
    m += [mac(pre + "Fzero", f"{h['F0']:.2f}"), mac(pre + "Fend", f"{h['F_end']:.2f}"), mac(pre + "Tzero", f"{h['T0_nK']:.0f}"), mac(pre + "Tend", f"{h['T_end_nK']:.0f}"), mac(pre + "dFmid", f"{h['dF_at_1p5']:.2f}"), mac(pre + "dFend", f"{h['dF_end']:.2f}"), mac(pre + "FNEzero", f"{h['F_NE0']:.2f}"), mac(pre + "Feqzero", f"{h['F_eq1_0']:.2f}"), mac(pre + "Nend", f"{h['N_end']/1e3:.1f}")]
hc = hold["calibration"]
m += [mac("etahold", f"{hc['eta_hold']:.1f}"), mac("somhold", f"{hc['s_omega']:.2f}"), mac("gbghold", f"{hc['gamma_bg']:.2f}")]
SENSNAMES = {"reference": "reference", "C12 x0.5": "CxiiHalf", "C12 x2": "CxiiTwo", "C22 x0.5": "CxxiiHalf", "C22 x2": "CxxiiTwo",
             "eta1 -10%": "etaMinus", "eta1 +10%": "etaPlus", "s_omega 0.9": "somega", "L3 = 0": "Lthreezero", "L3 x2": "Lthreetwo",
             "T1 +10%": "Tone", "c +20%": "cPlus", "c -20%": "cMinus", "a +20%": "cPlus", "a -20%": "cMinus", "evap 3d": "evapthree", "gamma_K = a t (Fig. 1b)": "gammaFig"}
if sens:
    for k, v in sens.items():
        m.append(mac("sens" + SENSNAMES.get(k, "".join(ch for ch in k if ch.isalpha())), f"{v['F'][-1]:.2f}"))
        m.append(mac("sensT" + SENSNAMES.get(k, "".join(ch for ch in k if ch.isalpha())), f"{v['T'][-1]:.0f}"))
open(os.path.join(ROOT, "manuscript", "numbers.tex"), "w").write("\n".join(m) + "\n")
print("\n".join(lines))
