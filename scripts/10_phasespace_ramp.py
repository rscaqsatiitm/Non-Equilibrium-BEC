"""Phase-space simulation of the last second of the ramp (t = 1 to 2 s).
Usage: python scripts/10_phasespace_ramp.py <g_eff/g> <gamma_bg> <K factor: 0|1|4> [n_particles] [tag] [fx/g]
Initial state: Bose cloud with N = 55993 (reduced by the K loss accumulated
before t = 1 s when K is on), T = 240 nK, seeded condensate of 10 atoms, in the
trap at f_bar = 136 Hz.  Ramp f_bar = 192 - 56 t Hz.  Output: data/ps_<tag>.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.phasespace.trap import default_trap
from nebec.phasespace.particles import Cloud
from nebec.phasespace.run import Simulation
from nebec.phasespace.analysis import gaussian_fit_T, momentum_temperatures, experiment_observables
from nebec.evaporation import make_omega, make_gamma_K, integrated_gamma_K, T1_MEAS

g_fac = float(sys.argv[1]); gamma_bg = float(sys.argv[2]); kfac = float(sys.argv[3])
Np = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
fx_fac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
tag = sys.argv[5] if len(sys.argv) > 5 else f"g{g_fac:.2f}_fx{fx_fac:.2f}_K{kfac:.0f}"
ROOT = os.path.join(os.path.dirname(__file__), "..")
gk = json.load(open(os.path.join(ROOT, "data", "gammaK_fit.json")))
c = {0: 0.0, 1: gk["c_1x"], 4: gk["c_4x"]}[int(kfac)]
gK = make_gamma_K(a=c, k=2.0) if c > 0 else (lambda t: 0.0)
gamma = lambda t: gK(t) + gamma_bg
om = make_omega(1.0)
N1 = 55993.0 * (np.exp(-integrated_gamma_K(1.0, c, 1.0, 2.0)) if c > 0 else 1.0)

tr = default_trap(1.0, g_eff=g_fac * 9.81, fx=fx_fac * 9.81)
rng = np.random.default_rng(11)
sim = Simulation(tr, lambda t: om(t + 1.0), Cloud(np.zeros((1, 3)), np.zeros((1, 3))), Nc0=10.0, bose=True, dt=40e-6, seed=5, T_ref=120e-9, smooth=True)
sim._set_trap(0.0)
cl = Cloud.thermal(Np, T1_MEAS, tr, tr.s, alpha=N1 / Np, rng=rng, mu_over_kT=-0.05)
cl.alpha = N1 / cl.r.shape[0]
sim = Simulation(tr, lambda t: om(t + 1.0), cl, Nc0=10.0, bose=True, dt=40e-6, seed=5, T_ref=120e-9, smooth=True, gamma=gamma)
print(f"start: N={cl.N:.0f}, particles {cl.r.shape[0]}, alpha {cl.alpha:.2f}, f_bar={sim.summary()['f_bar']:.1f} Hz, depth {sim.depth/C.nK:.0f} nK, g_eff={g_fac} g, fx={fx_fac} g", flush=True)
rows = []
def cb(S):
    d = S.summary(); Tm = momentum_temperatures(S.trapped_cloud())
    ob = experiment_observables(S.trapped_cloud(), S.cond, axes=(0, 2))
    d.update(Tmx=Tm[0], Tmy=Tm[1], Tmz=Tm[2], T_fit_x=ob["T_fit_x"], T_fit_z=ob["T_fit_y"], F_fit=ob["F_fit"], Nth_fit=ob["N_th_fit"], Nc_fit=ob["N_c_fit"], t_lab=S.t + 1.0)
    rows.append(d)
    print(f"t={d['t_lab']:.2f} N={d['N']:.0f} Nth={d['Nth']:.0f} Nc={d['Nc']:.0f} F={d['F']:.3f} T_kin x,y,z={Tm[0]*1e9:.0f},{Tm[1]*1e9:.0f},{Tm[2]*1e9:.0f} nK | bimodal fit: T x,z={ob['T_fit_x']*1e9:.0f},{ob['T_fit_y']*1e9:.0f} nK F_fit={ob['F_fit']:.3f} | f_bar={d['f_bar']:.0f} depth={d['depth']/C.nK:.0f} nK c22={d['n_c22']} out={d['n_out']} in={d['n_in']} evap={d['n_evap']:.0f}  [{time.time()-T0:.0f} s]", flush=True)
T0 = time.time()
sim.run(1.0, callback=cb, every=0.05)
json.dump([{k: (float(v) if np.isscalar(v) else v) for k, v in d.items()} for d in rows], open(os.path.join(ROOT, "data", f"ps_{tag}.json"), "w"), indent=1)
print("done")
