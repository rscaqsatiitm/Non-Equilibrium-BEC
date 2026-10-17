"""Phase-space simulation of the hold phase (Fig. 4 / S2(ii)): constant trap at
f_bar (the paper's calibrated 99 Hz times a scale), initial state from the
measured (N, T, F) with a Bose cloud at mu = mu_c - delta (undersaturated if
the measured thermal number is below saturation), no K.
Usage: python scripts/11_phasespace_hold.py <g_eff/g> <gamma_bg> <series: nodiss|diss> [n_particles] [t_hold] [tag]
"""
import sys, os, json, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.phasespace.trap import default_trap
from nebec.phasespace.particles import Cloud
from nebec.phasespace.run import Simulation
from nebec.phasespace.condensate import TFCondensate
from nebec.phasespace.analysis import momentum_temperatures, experiment_observables

g_fac = float(sys.argv[1]); gamma_bg = float(sys.argv[2]); series = sys.argv[3]
Np = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
t_hold = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
tag = sys.argv[6] if len(sys.argv) > 6 else f"hold_{series}_g{g_fac:.2f}"
ROOT = os.path.join(os.path.dirname(__file__), "..")
def load(name):
    with open(os.path.join(ROOT, "data", "digitized", name)) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
d = load("figS2ii_nodiss.csv" if series == "nodiss" else "figS2ii_diss.csv")
N0, T0, F0 = d["N"][0], d["T_nK"][0] * 1e-9, d["F"][0]
om_cal = 2 * np.pi * 99.2
tr = default_trap(1.0, g_eff=g_fac * 9.81)
rng = np.random.default_rng(7)
sim0 = Simulation(tr, lambda t: om_cal, Cloud(np.zeros((1, 3)), np.zeros((1, 3))), Nc0=N0 * F0, bose=True, dt=40e-6)
w, r0, axes = sim0.w, sim0.r0, sim0._axes
wbar = np.prod(w) ** (1 / 3)
Nth0 = N0 * (1 - F0); Nc0 = N0 * F0
Nsat = 1.202 * (C.kB * T0 / (C.hbar * wbar)) ** 3
mu_c = C.mu_TF(Nc0, wbar)
# fugacity of the thermal cloud from N_th/N_sat (ideal gas, g_3(z)/zeta(3))
from scipy.optimize import brentq
g3 = lambda z: sum(z**k / k**3 for k in range(1, 400))
z0 = 1.0 if Nth0 >= Nsat else brentq(lambda z: g3(z) / 1.202 - Nth0 / Nsat, 1e-6, 1.0)
mu_over_kT = np.log(z0) + mu_c / (C.kB * T0)          # relative to the trap bottom, band bottom at mu_c
cond = TFCondensate(Nc0, w, r0, axes)
cl = Cloud.thermal(Np, T0, tr, tr.s, alpha=Nth0 / Np, rng=rng, mu_over_kT=mu_over_kT, cond=cond)
cl.alpha = Nth0 / cl.r.shape[0]
sim = Simulation(tr, lambda t: om_cal, cl, Nc0=Nc0, bose=True, dt=40e-6, seed=9, T_ref=T0, smooth=True, gamma=lambda t: gamma_bg)
print(f"{series}: N0={N0:.0f} T0={T0*1e9:.0f} nK F0={F0:.3f} -> Nth={Nth0:.0f} (N_sat={Nsat:.0f}, z={z0:.3f}), Nc={Nc0:.0f}, mu_c={mu_c/C.nK:.1f} nK; trap f={np.round(w/2/np.pi,1)} Hz depth {sim.depth/C.nK:.0f} nK", flush=True)
rows = []; T0w = time.time()
def cb(S):
    dd = S.summary(); Tm = momentum_temperatures(S.trapped_cloud()); ob = experiment_observables(S.trapped_cloud(), S.cond, axes=(0, 2))
    dd.update(Tmx=Tm[0], Tmy=Tm[1], Tmz=Tm[2], T_fit_x=ob["T_fit_x"], T_fit_z=ob["T_fit_y"], F_fit=ob["F_fit"], Nth_fit=ob["N_th_fit"], Nc_fit=ob["N_c_fit"])
    rows.append(dd)
    print(f"t={dd['t']:.2f} N={dd['N']:.0f} Nth={dd['Nth']:.0f} Nc={dd['Nc']:.0f} F={dd['F']:.3f} T_kin x,y,z={Tm[0]*1e9:.0f},{Tm[1]*1e9:.0f},{Tm[2]*1e9:.0f} | fit T x,z={ob['T_fit_x']*1e9:.0f},{ob['T_fit_y']*1e9:.0f} F_fit={ob['F_fit']:.3f} | out={dd['n_out']} in={dd['n_in']} c22={dd['n_c22']} evap={dd['n_evap']:.0f} [{time.time()-T0w:.0f} s]", flush=True)
sim.run(t_hold, callback=cb, every=0.05)
json.dump([{k: (float(v) if np.isscalar(v) else v) for k, v in r.items()} for r in rows], open(os.path.join(ROOT, "data", f"ps_{tag}.json"), "w"), indent=1)
print("done")
