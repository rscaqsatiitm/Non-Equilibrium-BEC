"""Semiclassical Hartree-Fock density of states of the thermal atoms in an
isotropic harmonic trap containing a Thomas-Fermi condensate.

Energies are measured in units of hbar*omega from the bottom of the thermal
band, x = (eps - mu_c)/(hbar omega); the condensate chemical potential is
m_c = mu_c/(hbar omega).  The effective potential seen by thermal atoms is
U = 2 mu_c - V(r) inside the condensate and V(r) outside (BZS Eq. 48), so the
band bottom is at mu_c (on the condensate surface).

rho_hat(x, m_c) = hbar*omega * rho(eps) follows BZS Eqs. (49)-(50):
    rho_hat = (2/pi) [I_-(x) + I_+(x)],
with a_+- = 2 (x +- m_c) (shifted energies) and the integrals over y = r/l.
"""
import numpy as np


def _Iminus(x, mc):
    """I_-(x): integral over the condensate region, y^2 < 2 m_c, of
    y^2 sqrt(2(x - m_c) + y^2) where the root is real: y^2 > 2(m_c - x)."""
    x = np.asarray(x, dtype=float)
    am = 2.0 * (x - mc)                       # a_- (shifted: eps_bar - mu_c -> x - m_c)
    lo = np.sqrt(np.maximum(0.0, -am))       # sqrt(max(0, -a_-))
    hi = np.sqrt(2.0 * mc)
    def F(y):
        u = np.sqrt(np.maximum(am + y * y, 0.0))
        # antiderivative of y^2 sqrt(a + y^2):
        # y u^3/4 - a y u/8 - a^2/8 log(y + u)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.where(y + u > 0, np.log(np.maximum(y + u, 1e-300)), 0.0)
        return y * u**3 / 4.0 - am * y * u / 8.0 - am**2 / 8.0 * L
    val = F(hi) - F(lo)
    return np.where(hi > lo, val, 0.0)


def _Iplus(x, mc):
    """I_+(x): integral over y^2 > 2 m_c of y^2 sqrt(2(x + m_c) - y^2)
    up to y^2 = 2(x + m_c)."""
    x = np.asarray(x, dtype=float)
    ap = 2.0 * (x + mc)
    lo = np.sqrt(2.0 * mc)
    hi = np.sqrt(np.maximum(ap, 0.0))
    def F(y):
        u = np.sqrt(np.maximum(ap - y * y, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.where(ap > 0, np.arcsin(np.clip(y / np.sqrt(np.maximum(ap, 1e-300)), -1, 1)), 0.0)
        return -y * u**3 / 4.0 + ap * y * u / 8.0 + ap**2 / 8.0 * A
    val = F(hi) - F(lo)
    return np.where(hi > lo, val, 0.0)


def rho_hat(x, mc):
    """Dimensionless density of states hbar*omega*rho(eps) at shifted energy x."""
    x = np.asarray(x, dtype=float)
    if mc <= 0:
        return 0.5 * np.maximum(x, 0.0) ** 2
    return (2.0 / np.pi) * (_Iminus(x, mc) + _Iplus(x, mc))


def Gamma_hat(x, mc, nsub=64):
    """Cumulative number of states below x (per unit occupation), by
    composite trapezoid on a fine sub-grid (rho_hat ~ x^{3/2} near 0)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    u = np.linspace(0.0, 1.0, nsub + 1)
    yy = x[:, None] * u[None, :]
    r = rho_hat(yy, mc)
    return np.trapezoid(r, yy, axis=1)


def rho_w_hat(xhalf, mc, dmc_dt, delta=None):
    """Weighted density of states (BZS Eq. 16) in shifted dimensionless units
    at the half-grid points: rho_w = dm_c/dt * W(x), W = -dGamma_hat/dm_c.
    It is the flux of states through x per unit time caused by the change of
    the condensate mean field."""
    if dmc_dt == 0.0 or mc <= 0:
        return np.zeros_like(np.asarray(xhalf, dtype=float))
    d = delta if delta is not None else max(1e-3 * mc, 1e-4)
    mp, mm = mc + d, max(mc - d, 1e-6)
    W = -(Gamma_hat(xhalf, mp) - Gamma_hat(xhalf, mm)) / (mp - mm)
    return dmc_dt * W


def rho_c_table(x, mc):
    """Condensate-weighted factor Phi(U_-/mu_c) of BZS Eqs. (57)-(60) for all
    pairs (x3, x4) of the grid: the fraction of the condensate that can take
    part in a collision producing thermal atoms at x3 and x4 (or the inverse)."""
    x = np.asarray(x, dtype=float)
    X3, X4 = np.meshgrid(x, x, indexing="ij")
    Um = (2.0 / 3.0) * ((X3 + X4) - np.sqrt(np.maximum(X3 * X3 - X3 * X4 + X4 * X4, 0.0)))
    if mc <= 0:
        return np.ones_like(Um)
    u = np.clip(Um / mc, 0.0, 1.0)
    Phi = 2.5 * (1.0 - (1.0 - u) ** 1.5) - 1.5 * (1.0 - (1.0 - u) ** 2.5)
    return np.where(Um >= mc, 1.0, Phi)
