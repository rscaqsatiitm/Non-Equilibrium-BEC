"""Anisotropic (three-temperature) moment model of the thermal cloud.

Motivation: at the end of the ramp the thermal cloud makes only 5 to 7
collisions per atom per second, so the exchange of energy between the trap
axes (about 2.7 collisions per atom, Monroe et al. 1993 / Wu and Foot 1996,
recalled values) takes 0.3 to 0.5 s, while evaporation in a crossed dipole trap
removes energy along gravity only.  The ergodic model cannot represent this.

State: N_th, E_x, E_y, E_z (thermal energies per axis, kinetic + potential),
N_c.  Closure: ideal Bose gas with a product distribution
f = 1/(exp(sum_i eps_i/k T_i - alpha) - 1), for which
    N_th = g_3(z) prod_i (k T_i / hbar w_i),   E_i = k T_i N_th g_4(z)/g_3(z).
Processes:
  * adiabatic ramp:        dE_i/dt = E_i (dw_i/dt)/w_i
  * cross-axis relaxation: dE_i/dt = -gamma_x (E_i - E/3), gamma_x = gamma_coll/2.7
  * evaporation along z:   dN/dt = -c_ev gamma_coll N_th exp(-eta_z) (1D escape),
                           removing (U + k T_z) along z and k T_x, k T_y radially
  * uniform loss gamma(t) on N_th, E_i, N_c; three-body on N_c
  * condensate exchange:   dN_c/dt = Gamma_12 (N_th - N_sat), Gamma_12 = c_12 gamma_coll,
                           with E_i unchanged (energies measured from the band bottom)
Here c_ev is calibrated on the data without K and c_12 = 3 is the rate found in the
ergodic simulation (50 ms refill at gamma_coll = 7 Hz).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import zeta
from . import constants as C

Z3, Z4 = zeta(3), zeta(4)


_ZT = np.concatenate((np.linspace(1e-9, 0.9, 2000, endpoint=False), 1 - np.logspace(-1, -9, 2000)))
_K = np.arange(1, 2001)
_G3 = np.array([np.sum(z**_K / _K**3) for z in _ZT])
_G4 = np.array([np.sum(z**_K / _K**4) for z in _ZT])


def polylog(s, z, n=None):
    """g_s(z) for s = 3, 4 by interpolation of a fine table (z in (0, 1])."""
    tab = _G3 if s == 3 else _G4
    return float(np.interp(z, _ZT, tab))


def thermo(Nth, E, omegas):
    """Return (z, T_i) from N_th, E = (E_x, E_y, E_z) and omegas."""
    E = np.maximum(E, 1e-40)
    # N^4 g4^3/g3^4 = prod E_i /(hbar^3 prod w_i)
    rhs = np.prod(E) / (C.hbar**3 * np.prod(omegas)) / Nth**4
    def f(z):
        g3, g4 = polylog(3, z), polylog(4, z)
        return g4**3 / g3**4 - rhs
    if f(1.0 - 1e-12) >= 0:          # oversaturated: z = 1
        z = 1.0
    elif f(1e-6) <= 0:
        z = 1e-6
    else:
        z = brentq(f, 1e-6, 1.0 - 1e-12)
    g3, g4 = polylog(3, z), polylog(4, z)
    T = E * g3 / (Nth * g4 * C.kB)
    return z, T


def Nsat(T, omegas):
    return Z3 * np.prod(C.kB * T / (C.hbar * np.asarray(omegas)))


class ThreeTemperatureModel:
    def __init__(self, omega_bar, ratios=(1.2, 0.7, 1.2), trap_depth=None, gamma=lambda t: 0.0,
                 c_ev=1.0, c_12=2.0, L3=C.L3_cond, chi=2.7, evap_axis=2, Nc_floor=1.0):
        self.omega_bar = omega_bar              # function of t (rad/s), geometric mean
        self.ratios = np.array(ratios)          # (w_x, w_y, w_z)/w_bar; z = gravity axis
        self.U = trap_depth                     # function of t (J)
        self.gamma = gamma
        self.c_ev, self.c_12, self.L3, self.chi = c_ev, c_12, L3, chi
        self.az = evap_axis
        self.Nc_floor = Nc_floor

    def omegas(self, t):
        return self.ratios * self.omega_bar(t)

    def domega(self, t, h=1e-4):
        return (self.omega_bar(t + h) - self.omega_bar(t - h)) / (2 * h) / self.omega_bar(t)

    def gamma_coll(self, Nth, T, omegas):
        nbar = Nth * np.prod(np.sqrt(C.m_Rb * omegas**2 / (4 * np.pi * C.kB * T)))
        vrel = np.sqrt(16 * C.kB * T.mean() / (np.pi * C.m_Rb))
        return nbar * C.sigma_el * vrel

    def rhs(self, t, y):
        Nth, Ex, Ey, Ez, Nc = y
        Nth = max(Nth, 1.0); Nc = max(Nc, 0.0)
        E = np.array([Ex, Ey, Ez]); om = self.omegas(t)
        z, T = thermo(Nth, E, om)
        gc = self.gamma_coll(Nth, T, om)
        dN, dE, dNc = 0.0, np.zeros(3), 0.0
        # ramp
        dE += E * self.domega(t)
        # cross-axis relaxation
        dE += -(gc / self.chi) * (E - E.sum() / 3.0)
        # evaporation along z
        Tz = T[self.az]; eta_z = self.U(t) / (C.kB * Tz)
        rate = self.c_ev * gc * Nth * np.exp(-eta_z)
        dN -= rate
        for i in range(3):
            dE[i] -= rate * ((self.U(t) + C.kB * Tz) if i == self.az else C.kB * T[i])
        # uniform losses
        g = self.gamma(t)
        dN -= g * Nth; dE -= g * E; dNc -= g * Nc
        # three-body
        if Nc > 0:
            npk = C.mu_TF(Nc, self.omega_bar(t)) / C.g_int
            dNc -= self.L3 * (8.0 / 21.0) * npk**2 * Nc
        # condensate exchange
        # bosonic-stimulation form (Gardiner et al.): rate proportional to the
        # relative over/undersaturation and to (N_c + 1); c_12 = 2 reproduces the
        # 50 ms refill of a 26% undersaturated cloud found in the ergodic model
        Ns = Nsat(T, om)
        ex = self.c_12 * gc * (Nth - Ns) / Ns * (Nc + 1.0)
        if ex < 0 and Nc <= 0:
            ex = 0.0
        dNc += ex; dN -= ex
        return [dN, dE[0], dE[1], dE[2], dNc]

    def initial(self, t0, Nth, T, Nc=1.0, aniso=(1.0, 1.0, 1.0)):
        om = self.omegas(t0)
        Ti = T * np.array(aniso)
        # choose z such that N_th matches (z<=1); E_i = kT_i N g4/g3
        Ns = Nsat(Ti, om)
        if Nth >= Ns:
            z = 1.0
            Nth_eff = Nth
        else:
            z = brentq(lambda zz: polylog(3, zz) / Z3 * Ns - Nth, 1e-9, 1.0)
            Nth_eff = Nth
        g3, g4 = polylog(3, z), polylog(4, z)
        E = C.kB * Ti * Nth_eff * g4 / g3
        return np.array([Nth_eff, E[0], E[1], E[2], Nc])

    def run(self, y0, t0, t1, t_eval):
        sol = solve_ivp(self.rhs, (t0, t1), y0, t_eval=t_eval, rtol=1e-6, atol=1e-3, method="LSODA", max_step=2e-3)
        out = dict(t=sol.t, Nth=sol.y[0], Nc=sol.y[4])
        Tx, Ty, Tz, zz, gcs = [], [], [], [], []
        for k in range(len(sol.t)):
            om = self.omegas(sol.t[k]); E = sol.y[1:4, k]
            z, T = thermo(max(sol.y[0, k], 1.0), E, om)
            Tx.append(T[0]); Ty.append(T[1]); Tz.append(T[2]); zz.append(z); gcs.append(self.gamma_coll(max(sol.y[0, k], 1), T, om))
        out.update(Tx=np.array(Tx), Ty=np.array(Ty), Tz=np.array(Tz), z=np.array(zz), gamma_coll=np.array(gcs))
        out["N"] = out["Nth"] + out["Nc"]; out["F"] = out["Nc"] / out["N"]
        out["T_geo"] = (out["Tx"] * out["Ty"] * out["Tz"]) ** (1 / 3)
        out["T_mean"] = (out["Tx"] + out["Ty"] + out["Tz"]) / 3
        return out
