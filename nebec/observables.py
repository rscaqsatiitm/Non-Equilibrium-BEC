"""Observables computed the way the experiment measures them.

* Thermal momentum distribution in the ergodic approximation and the
  temperature obtained from a Gaussian fit to it (the PRL fits a Gaussian plus
  Thomas-Fermi profile to time-of-flight images; after 20 ms the thermal
  image is the momentum distribution up to a <1% size correction).
* Chemical potential of the thermal cloud from (N_th, T) via the semiclassical
  relation N_th = g_3(z) (k_B T / hbar omega)^3 (supplement Eqs. S7-S8).
* Equilibrium condensate fraction of Eq. (1) of the PRL.
"""
import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.special import zeta

from . import constants as C
from .dos import rho_hat


def polylog3(z, nmax=200):
    k = np.arange(1, nmax + 1)
    return np.sum(z ** k / k**3)


def mu_thermal(Nth, T, omega):
    """mu (J) of an ideal Bose gas with Nth atoms at T in a trap omega; the
    paper's definition (fugacity z from N_th = g_3(z)(k_B T/hbar omega)^3).
    Returns nan if N_th exceeds the saturated number."""
    Nsat = zeta(3) * (C.kB * T / (C.hbar * omega)) ** 3
    if Nth >= Nsat:
        return np.nan
    f = lambda lnz: polylog3(np.exp(lnz)) * (C.kB * T / (C.hbar * omega)) ** 3 - Nth
    lnz = brentq(f, -60.0, -1e-12)
    return lnz * C.kB * T


def F_eq_eq1(T, N, omega):
    """Eq. (1) of the PRL: F = 1 - tau^3 - eta tau^2 (1 - tau^3)^{2/5}."""
    Tc = C.Tc_ideal(N, omega)
    tau = T / Tc
    eta = 2.15 * (C.a_s * N ** (1.0 / 6.0) / C.aho(omega)) ** 0.4
    if tau >= 1:
        return 0.0
    return max(1 - tau**3 - eta * tau**2 * (1 - tau**3) ** 0.4, 0.0)


def momentum_distribution(x, g, mc, omega, npts=400, pmax_kT=None):
    """Radial momentum distribution n(p) (per d^3p/(2 pi hbar)^3 ... up to a
    constant) of the thermal cloud in the ergodic approximation:
    n(p) = int d^3r g(eps(r,p)), eps = p^2/2m + U_eff(r) - mu_c.
    Returns (p_grid [in units of sqrt(m hbar omega)], n(p))."""
    # effective potential in units of hbar*omega, radial coordinate y = r/l
    ymax = np.sqrt(2.0 * (x[-1] + 2 * mc)) + 1.0
    y = np.linspace(0.0, ymax, 600)
    V = 0.5 * y * y
    U = np.where(V < mc, 2 * mc - V, V) - mc        # shifted effective potential
    p = np.linspace(0.0, np.sqrt(2.0 * x[-1]), npts)
    out = np.zeros_like(p)
    gi = lambda e: np.interp(e, x, g, left=0.0, right=0.0)
    for i, pv in enumerate(p):
        e = 0.5 * pv * pv + U
        out[i] = np.trapezoid(y * y * gi(e), y)
    return p, out


def gaussian_fit_temperature(x, g, mc, omega):
    """Temperature from a least-squares Gaussian fit to the 2D column density
    of the thermal momentum distribution (what a Gaussian fit to a TOF image
    returns), weighting each 2D pixel equally."""
    p, n3 = momentum_distribution(x, g, mc, omega)
    # 2D column density from the radial 3D distribution (Abel projection)
    q = p[:-1]
    n2 = np.array([2.0 * np.trapezoid(np.interp(np.sqrt(qv**2 + np.linspace(0, np.sqrt(max(p[-1]**2 - qv**2, 0)), 200)**2), p, n3), np.linspace(0, np.sqrt(max(p[-1]**2 - qv**2, 0)), 200)) for qv in q])
    w = q  # area weight of a ring in 2D
    def model(qq, A, s2):
        return A * np.exp(-qq**2 / (2 * s2))
    A0 = n2[0]; s20 = max(np.sum(w * q**2 * n2) / np.sum(w * n2) / 2, 1e-6)
    try:
        popt, _ = curve_fit(model, q, n2, p0=[A0, s20], sigma=1.0 / np.sqrt(w + 1e-9), maxfev=20000)
    except RuntimeError:
        return np.nan
    # p is in units sqrt(m hbar omega): p^2/(2m) = (p_dimless^2/2) hbar omega
    # Gaussian width sigma_p^2 = m k_B T  ->  s2 (dimless) = k_B T/(hbar omega)
    return popt[1] * C.hbar * omega / C.kB


def effective_temperature(x, g, rho, mc, omega):
    """Temperature from the mean kinetic+potential energy per thermal atom via
    the equilibrium relation of the same ergodic model (E/N for a saturated Bose
    gas with the current DoS), solved for T.  Used as the "true" T."""
    n = rho * g
    E = np.sum((x + mc) * n) / max(np.sum(n), 1e-300)   # in hbar*omega
    def EoverN(T):
        beta = C.hbar * omega / (C.kB * T)
        gg = 1.0 / np.expm1(np.maximum(beta * x, 1e-12)); gg[0] = 0.0
        return np.sum((x + mc) * rho * gg) / np.sum(rho * gg)
    try:
        return brentq(lambda T: EoverN(T) - E, 1e-9, 5e-6)
    except ValueError:
        return np.nan
