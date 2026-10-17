"""Does the DSMC C22 kernel preserve a saturated Bose energy distribution?
Sample a Bose cloud (z -> 1) in the trap at f_bar = 108 Hz, T = 150 nK, N = 47000
(1.6 x saturation), evolve with C22 only (no condensate exchange) and compare
the low-energy occupation g(x) with the Bose and Maxwell-Boltzmann forms."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.phasespace.trap import default_trap
from nebec.phasespace.particles import Cloud
from nebec.phasespace.run import Simulation
from nebec.dos import rho_hat
g_fac, fx_fac, Np = 0.85, 0.2, int(sys.argv[1]) if len(sys.argv) > 1 else 40000
T, N, fb = 150e-9, 47000.0, 108.0
tr = default_trap(1.0, g_eff=g_fac * 9.81, fx=fx_fac * 9.81)
sim = Simulation(tr, lambda t: 2 * np.pi * fb, Cloud(np.zeros((1, 3)), np.zeros((1, 3))), Nc0=1.0, bose=True, dt=40e-6, c12=False)
sim._set_trap(0.0)
rng = np.random.default_rng(3)
cl = Cloud.thermal(Np, T, tr, tr.s, alpha=N / Np, rng=rng, mu_over_kT=-0.01, U_max=sim.depth, in_basin=tr.basin(tr.s, 0.999 * sim.depth))
sim = Simulation(tr, lambda t: 2 * np.pi * fb, cl, Nc0=1.0, bose=True, dt=40e-6, c12=False, scale_table=sim, seed=4)
wbar = np.prod(sim.w) ** (1 / 3)
def hist(S):
    m = S.trapped_mask(); r = S.cloud.r[m]; v = S.cloud.v[m]
    eps = 0.5 * C.m_Rb * np.sum(v * v, axis=1) + S.trap.potential(r) - S.U0_now
    x = eps / (C.hbar * wbar)
    edges = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]) * 1.0
    cnt, _ = np.histogram(x, edges); cnt = cnt * S.cloud.alpha
    xm = 0.5 * (edges[1:] + edges[:-1])
    rho = rho_hat(xm, 0.0) * (edges[1:] - edges[:-1])
    g = cnt / rho
    Tk = S.trapped_cloud().axis_temperatures().mean(); kT = C.kB * Tk / (C.hbar * wbar)
    return xm, g, kT, S.trapped_cloud().N, Tk
xm, g0, kT, N0, Tk0 = hist(sim)
print(f"depth {sim.depth/C.nK:.0f} nK, kT/hw = {kT:.1f}, N = {N0:.0f}, Nsat = {1.202*kT**3:.0f}")
print(" x      g(t=0)   Bose(z=1)   MB(same N)")
Nsat = 1.202 * kT ** 3
for i in range(len(xm)):
    print(f"{xm[i]:6.1f} {g0[i]:9.2f} {1/np.expm1(xm[i]/kT):9.2f} {N0/kT**3*np.exp(-xm[i]/kT):9.2f}")
T0 = time.time()
for k in range(3):
    sim.run(sim.t + 0.1)
    xm, g, kT, Nn, Tk = hist(sim)
    print(f"t = {sim.t:.2f} s, N = {Nn:.0f}, T_kin = {Tk*1e9:.0f} nK, c22 = {sim.stats['n_c22']}  [{time.time()-T0:.0f} s]")
    print(" x      g        Bose(z=1)  MB(same N)  g/g0")
    for i in range(len(xm)):
        print(f"{xm[i]:6.1f} {g[i]:9.2f} {1/np.expm1(xm[i]/kT):9.2f} {Nn/kT**3*np.exp(-xm[i]/kT):9.2f} {g[i]/g0[i]:6.2f}")
