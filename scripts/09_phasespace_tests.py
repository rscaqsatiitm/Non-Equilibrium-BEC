"""Tests of the phase-space model: classical collision rate, cross-axis
thermalization, detailed balance of the condensate exchange, refill time of an
undersaturated cloud.  Usage: python scripts/09_phasespace_tests.py [n_particles]"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.phasespace.trap import default_trap
from nebec.phasespace.particles import Cloud
from nebec.phasespace.run import Simulation
from nebec.phasespace.condensate import TFCondensate
Np = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
rng = np.random.default_rng(2)
tr = default_trap(1.0, g_eff=0.5 * 9.81); om = lambda t: 2 * np.pi * 108.0
T = 100e-9; Nc0 = 5000.0
sim0 = Simulation(tr, om, Cloud(np.zeros((1, 3)), np.zeros((1, 3))), Nc0=Nc0, bose=True, dt=50e-6)
w = sim0.w; r0 = sim0.r0; wbar = np.prod(w) ** (1 / 3)
Nsat = 1.202 * (C.kB * T / (C.hbar * wbar)) ** 3
print('trap f_bar=%.1f Hz freqs %s, N_sat(100 nK)=%.0f, mu_c(Nc=5000)=%.1f nK, depth %.0f nK' % (wbar / 2 / np.pi, np.round(w / 2 / np.pi, 1), Nsat, C.mu_TF(Nc0, wbar) / C.nK, sim0.depth / C.nK))
mu_c_kT = C.mu_TF(Nc0, wbar) / (C.kB * T)
for label, Nth, mu in (('saturated (mu = mu_c)', Nsat, mu_c_kT), ('undersaturated 26%', 0.74 * Nsat, mu_c_kT - 0.3)):
    cond = TFCondensate(Nc0, w, r0, sim0._axes)
    cl = Cloud.thermal(Np, T, tr, tr.s, alpha=Nth / Np, rng=rng, mu_over_kT=mu, cond=cond)
    cl.alpha = Nth / cl.r.shape[0]
    sim = Simulation(tr, om, cl, Nc0=Nc0, bose=True, dt=50e-6, seed=3, T_ref=T, smooth=True)
    rows = []; t0 = time.time()
    sim.run(0.2, callback=lambda S: rows.append(S.summary()), every=0.01)
    print(f"--- {label}: Nth0={Nth:.0f}, particles {cl.r.shape[0]}, alpha={cl.alpha:.2f}, wall {time.time()-t0:.0f} s")
    for d in rows[::2]:
        print(f"  t={d['t']:.3f} Nth={d['Nth']:.0f} Nc={d['Nc']:.0f} F={d['F']:.3f} out={d['n_out']} in={d['n_in']} c22={d['n_c22']} T_kin={d['T_kin']*1e9:.1f} nK")
