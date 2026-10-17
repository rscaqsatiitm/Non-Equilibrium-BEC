"""Compare the condensate growth rate of the ergodic C12 integral with the
constant-temperature, infinite-reservoir growth equation of Gardiner et al.,
PRL 81, 5266 (1998):  dn0/dt = 2 W+ { [1 - exp((mu(n0) - mu)/kT)] n0 + 1 },
with W+ from their Eq. (10) (time unit 8 m a^2 w^2/(pi hbar) = our Gamma0).
We take a thermal cloud that is a Bose distribution at T with chemical
potential mu = mu_c + delta (slightly oversaturated), no truncation, and
evaluate dN_c/dt of our model for several N_c, T.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.kinetic import KineticModel, Protocol
from nebec.collisions import c12
from nebec.dos import rho_hat, rho_c_table

def lerch(z, r, kmax=4000):
    k = np.arange(kmax)
    return np.sum(z**k / (k + r + 1.0))

def Wplus_gardiner(T, mu, mu_n0, omega, emax):
    kT = C.kB * T
    z = np.exp((mu - emax) / kT); zn = np.exp((mu_n0 - emax) / kT)
    s = sum((z * zn) ** r * lerch(z, r) ** 2 for r in range(1, 400))
    return 0.5 * (kT / (C.hbar * omega)) ** 2 * (np.log(1 - z) ** 2 + z**2 * s)

om = 2 * np.pi * 100.0
P = Protocol(omega=lambda t: om, trap_depth=lambda t: 5e-6 * C.kB, L3=0.0)
M = KineticModel(P, n_grid=1500, open_top=False); M.setup_grid(0.0, 5e-6 * C.kB / (C.hbar * om))
print(f"{'T(nK)':>6} {'Nc':>7} {'mu_c(nK)':>9} {'dmu/kT':>7} {'model dNc/dt':>13} {'Gardiner dn0/dt':>16} {'ratio':>6}  {'W+ model(Hz)':>12} {'W+ G98 (Hz)':>11}")
for T in (80e-9, 120e-9, 200e-9):
    for Nc in (1e3, 5e3, 2e4):
        for dmu in (0.005, 0.02):
            mc = M.mc(Nc, om); rho, Phi = M.tables(mc)
            kT = C.kB * T / (C.hbar * om)
            g = np.zeros(M.n); g[1:] = 1.0 / np.expm1(M.x[1:] / kT - dmu)     # mu - mu_c = dmu kT
            d12, dNc = c12(g, Phi, M.Delta, C.Gamma0(om), Nc, False)
            mu_c = C.mu_TF(Nc, om); mu = mu_c + dmu * C.kB * T
            emax = 2 * mu_c
            Wp = Wplus_gardiner(T, mu, mu_c, om, emax) * C.Gamma0(om)
            gard = 2 * Wp * ((1 - np.exp(-dmu)) * Nc + 1)
            Wmodel = dNc / (2 * ((1 - np.exp(-dmu)) * Nc + 1))
            print(f"{T*1e9:6.0f} {Nc:7.0f} {mu_c/C.nK:9.1f} {dmu:7.2f} {dNc:13.1f} {gard:16.1f} {dNc/gard:6.2f}  {Wmodel:12.2f} {Wp:11.2f}")
