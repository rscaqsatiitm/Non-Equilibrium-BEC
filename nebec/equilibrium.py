"""Equilibrium references.

* `closed_system_equilibrium(N, E, omega)`: the (N, E)-conserving equilibrium of
  the same ergodic Hartree-Fock model (saturated Bose cloud with mu = mu_c(N_c),
  Thomas-Fermi condensate), i.e. the state a closed system relaxes to.  This is
  the reference that replaces "equilibrium at the measured T" when the loss is
  uniform in energy.
* `saturated_cloud(T, Nc, omega, x, rho)`: N_th and E_th of the saturated
  thermal cloud at temperature T in the presence of N_c condensate atoms.
"""
import numpy as np
from scipy.optimize import brentq

from . import constants as C
from .dos import rho_hat


def saturated_cloud(T, Nc, omega, x_max=None, n=2000):
    mc = C.mu_TF(Nc, omega) / (C.hbar * omega)
    kT = C.kB * T / (C.hbar * omega)
    x_max = x_max if x_max is not None else 40 * kT + 10
    x = np.linspace(0.0, x_max, n)
    rho = rho_hat(x, mc); rho[0] = 0.0
    g = np.zeros(n); g[1:] = 1.0 / np.expm1(x[1:] / kT)
    Nth = np.trapezoid(rho * g, x)
    Eth = C.hbar * omega * np.trapezoid((x + mc) * rho * g, x)
    Ec = (5.0 / 7.0) * C.mu_TF(Nc, omega) * Nc
    return Nth, Eth, Ec


def closed_system_equilibrium(N, E, omega, x_max=None):
    """Return (T, Nc, F) of the equilibrium state with total number N and
    total energy E (J).  If no condensate is needed, Nc = 0 and the thermal
    cloud is unsaturated (fugacity < 1)."""
    # condensed branch: find Nc such that Nth(T*(Nc)) + Nc = N with E matched
    def energy_mismatch(Nc):
        # T such that Nth(T, Nc) = N - Nc
        f = lambda T: saturated_cloud(T, Nc, omega, x_max)[0] - (N - Nc)
        T = brentq(f, 1e-9, 5e-6)
        Nth, Eth, Ec = saturated_cloud(T, Nc, omega, x_max)
        return Eth + Ec - E, T
    lo, hi = 1.0, N - 1.0
    mlo, Tlo = energy_mismatch(lo)
    if mlo < 0:
        # even with ~no condensate the saturated cloud has too little energy:
        # the gas is above threshold -> unsaturated cloud, Nc = 0
        kT = lambda T: C.kB * T / (C.hbar * omega)
        def E_of_T_mu(T):
            x = np.linspace(0, 40 * kT(T) + 10, 2000); rho = 0.5 * x**2
            def Nfun(mu):
                g = np.zeros_like(x); g[1:] = 1.0 / np.expm1(np.maximum(x[1:] / kT(T) - mu, 1e-12))
                return np.trapezoid(rho * g, x), np.trapezoid(x * rho * g, x) * C.hbar * omega
            mu = brentq(lambda m: Nfun(m)[0] - N, -60.0, -1e-9)
            return Nfun(mu)[1]
        T = brentq(lambda T: E_of_T_mu(T) - E, 1e-9, 5e-6)
        return T, 0.0, 0.0
    Nc = brentq(lambda Nc: energy_mismatch(Nc)[0], lo, hi)
    T = energy_mismatch(Nc)[1]
    return T, Nc, Nc / N
