"""Condensate onset latency in the phase-space model: Bose cloud at T = 150 nK
in the trap at f_bar = 108 Hz with N = s x N_sat (s = 1.3, 1.6), seed N_c = 1,
C22 + ergodic exchange on; print N_c(t)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.phasespace.trap import default_trap
from nebec.phasespace.particles import Cloud
from nebec.phasespace.run import Simulation
sat = float(sys.argv[1]); Np = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
T, fb = 150e-9, 108.0
tr = default_trap(1.0, g_eff=0.85 * 9.81, fx=0.2 * 9.81)
sim = Simulation(tr, lambda t: 2 * np.pi * fb, Cloud(np.zeros((1, 3)), np.zeros((1, 3))), Nc0=1.0, bose=True, dt=40e-6)
sim._set_trap(0.0)
wbar = np.prod(sim.w) ** (1 / 3)
Nsat = 1.202 * (C.kB * T / (C.hbar * wbar)) ** 3
N = sat * Nsat
rng = np.random.default_rng(3)
cl = Cloud.thermal(Np, T, tr, tr.s, alpha=N / Np, rng=rng, mu_over_kT=-0.01, U_max=sim.depth, in_basin=tr.basin(tr.s, 0.999 * sim.depth))
sim = Simulation(tr, lambda t: 2 * np.pi * fb, cl, Nc0=1.0, bose=True, dt=40e-6, scale_table=sim, seed=4)
print(f"N = {N:.0f} = {sat} N_sat, depth {sim.depth/C.nK:.0f} nK, f_bar {fb} Hz, T {T*1e9:.0f} nK")
T0 = time.time()
def cb(S):
    d = S.summary()
    print(f"t = {S.t:.3f}  Nc = {d['Nc']:8.1f}  Nth = {d['Nth']:.0f}  T_kin = {d['T_kin']*1e9:.0f} nK  in = {d['n_in']} out = {d['n_out']}  [{time.time()-T0:.0f} s]", flush=True)
sim.run(0.4, callback=cb, every=0.02)
