"""Time evolution of the thermal energy distribution and the condensate number.

State vector y = [n_0, ..., n_{n-1}, N_c] with n_i = rho_hat_i g_i the number of
thermal atoms per unit shifted energy x (units of hbar*omega(t)).

Because the grid is measured in units of hbar*omega(t), an adiabatic change
of the trap frequency leaves n_i unchanged (harmonic-oscillator adiabatic
invariant); the only explicit "advection" in x comes from the change of the
condensate mean field, through rho_w (BZS Eqs. 16-19, 52), here evaluated
from the m_c dependence of the cumulative density of states.

Loss channels: atoms scattered above the truncation energy x_t(t) (evaporation,
inside the collision integrals), spilling of atoms above x_t when the trap depth
drops (fast relaxation), uniform one-body loss (K collisions + background),
three-body recombination.
"""
import numpy as np
from scipy.integrate import solve_ivp

from . import constants as C
from .dos import rho_hat, rho_w_hat, rho_c_table
from .collisions import c22, c12


class Protocol:
    """Time-dependent experimental parameters."""

    def __init__(self, omega, trap_depth, gamma_K=None, gamma_bg=0.0, L3=None):
        self.omega = omega                  # omega(t) [rad/s]
        self.trap_depth = trap_depth        # U(t) [J]
        self.gamma_K = gamma_K or (lambda t: 0.0)
        self.gamma_bg = gamma_bg
        self.L3 = C.L3_cond if L3 is None else L3

    def domega_dt(self, t, h=1e-4):
        return (self.omega(t + h) - self.omega(t - h)) / (2 * h)


class KineticModel:
    def __init__(self, protocol: Protocol, n_grid=320, x_max=None, spill_rate=None,
                 include_rho_w=True, include_c12=True, include_c22=True, c12_scale=1.0,
                 c22_scale=1.0, Nc_floor=1.0, open_top=True, evap_mode='3d', omega_z_ratio=1.0, kick_ratio=0.0):
        self.p = protocol
        self.n = n_grid
        self.x_max = x_max
        self.spill_rate = spill_rate
        self.include_rho_w = include_rho_w
        self.include_c12 = include_c12
        self.include_c22 = include_c22
        self.c12_scale = c12_scale
        self.c22_scale = c22_scale
        self.Nc_floor = Nc_floor
        self.open_top = open_top
        self.kick_ratio = kick_ratio        # rate of glancing K-Rb collisions (energy transfer uniform in [0, U_t]) relative to gamma_K
        self.evap_mode = evap_mode          # '3d': all atoms above U_t leave; '1d': only if the energy in one direction exceeds U_t
        self.omega_z_ratio = omega_z_ratio  # omega_z/omega_bar for the escape attempt rate in '1d' mode
        self._cache = {}

    # ----- grid helpers
    def setup_grid(self, t0, x_max):
        self.x_max = x_max
        self.Delta = x_max / (self.n - 1)
        self.x = np.arange(self.n) * self.Delta
        self.xhalf = (np.arange(self.n - 1) + 0.5) * self.Delta

    def mc(self, Nc, omega):
        return C.mu_TF(Nc, omega) / (C.hbar * omega)

    def dmc_dt(self, Nc, dNc_dt, omega, domega_dt):
        # m_c = c Nc^{2/5} omega^{1/5} / omega  ->  m_c ∝ Nc^{2/5} omega^{-4/5}
        if Nc <= 0:
            return 0.0
        return self.mc(Nc, omega) * (0.4 * dNc_dt / Nc - 0.8 * domega_dt / omega)

    def tables(self, mc):
        key = round(mc, 4)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            rho = rho_hat(self.x, mc)
            rho[0] = 0.0
            Phi = rho_c_table(self.x, mc)
            self._cache[key] = (rho, Phi)
        return self._cache[key]

    # ----- right-hand side
    def rhs(self, t, y, diag=None):
        n = self.n
        nvec = np.maximum(y[:n], 0.0)
        Nc = max(y[n], 0.0)
        omega = self.p.omega(t)
        mc = self.mc(max(Nc, self.Nc_floor), omega)
        rho, Phi = self.tables(mc)
        g = np.zeros(n)
        g[1:] = nvec[1:] / rho[1:]
        # truncation
        xt = self.p.trap_depth(t) / (C.hbar * omega) - mc
        n_act = int(np.clip(np.floor(xt / self.Delta), 1, n))
        g_act = g.copy()
        if self.evap_mode == '3d':
            g_act[n_act:] = 0.0
        G0 = C.Gamma0(omega)
        dn = np.zeros(n)
        dNc = 0.0
        if self.include_c22:
            dn += self.c22_scale * c22(g_act, rho, self.Delta, G0, n_act, self.open_top)
        if self.include_c12:
            d12, dNc12 = c12(g_act, Phi, self.Delta, G0 * self.c12_scale, max(Nc, self.Nc_floor), self.open_top)
            dn += d12
            dNc += dNc12
        # atoms above the truncation energy leave within ~ a trap period
        sr = self.spill_rate if self.spill_rate is not None else omega / (2 * np.pi)
        if self.open_top:
            if self.evap_mode == '3d':
                dn[n_act:] -= sr * nvec[n_act:]
            else:
                # ergodic fraction of the energy shell with energy in the
                # escape direction above the trap depth: (1 - U/eps)^2 for a
                # 3D harmonic trap; escape attempts at the trap frequency
                eps = self.x + mc                       # total energy above the trap bottom, units of hbar w
                Ut = xt + mc
                frac = np.where(eps > Ut, (1.0 - Ut / np.maximum(eps, 1e-12)) ** 2, 0.0)
                dn -= sr * self.omega_z_ratio * frac * nvec
        # glancing K-Rb collisions: an atom receives an energy kick E_t
        # distributed uniformly in [0, U_t] (flat differential cross section
        # in the quantum regime, supplement Eq. S2); kicks above U_t are losses
        if self.kick_ratio > 0 and self.p.gamma_K(t) > 0:
            gk = self.kick_ratio * self.p.gamma_K(t)
            Ut = xt + mc                      # trap depth above the trap bottom, units of hbar w
            nk = int(max(1, round(Ut / self.Delta)))   # number of grid steps spanned by the kick distribution
            # thermal atoms: leave x_i, arrive uniformly at x_i + j Delta, j = 1..nk
            cum = np.concatenate(([0.0], np.cumsum(nvec)))
            arrive = np.zeros(n)
            # arrive[i] = (1/nk) sum_{j=1..nk} nvec[i-j] = (cum[i] - cum[max(i-nk,0)])/nk
            idx = np.arange(n)
            arrive = (cum[idx] - cum[np.maximum(idx - nk, 0)]) / nk
            dn += gk * (arrive - nvec)
            # condensate atoms: kicked to energy E_t - mu_c above the band bottom if E_t > mu_c
            if Nc > 0:
                nkc = int(max(0, round((Ut - mc) / self.Delta)))
                if nkc > 0:
                    dn[1:nkc + 1] += gk * Nc / (nk * self.Delta)
                dNc -= gk * Nc * (nkc / nk)
        # one-body losses (uniform in energy) and three-body losses
        gam = self.p.gamma_K(t) + self.p.gamma_bg
        dn -= gam * nvec
        dNc -= gam * Nc
        if self.p.L3 > 0 and Nc > 0:
            npk = C.mu_TF(Nc, omega) / C.g_int
            dNc -= self.p.L3 * (8.0 / 21.0) * npk**2 * Nc
        # adiabatic shift of the thermal energies by the changing mean field
        if self.include_rho_w and Nc > self.Nc_floor:
            dmc = self.dmc_dt(Nc, dNc, omega, self.p.domega_dt(t))
            rw = rho_w_hat(self.xhalf, mc, dmc)
            flux = np.where(rw > 0, rw * g[:-1], rw * g[1:])
            flux[0] = 0.0 if self.x[0] == 0.0 else flux[0]
            dn[:-1] -= flux / self.Delta
            dn[1:] += flux / self.Delta
        out = np.empty(n + 1)
        out[:n] = dn
        out[n] = dNc
        if diag is not None:
            diag.update(dict(mc=mc, n_act=n_act, xt=xt, omega=omega))
        return out

    # ----- observables
    def unpack(self, t, y):
        n = self.n
        nvec = np.maximum(y[:n], 0.0)
        Nc = max(y[n], 0.0)
        omega = self.p.omega(t)
        mc = self.mc(max(Nc, self.Nc_floor), omega)
        rho, _ = self.tables(mc)
        g = np.zeros(n); g[1:] = nvec[1:] / rho[1:]
        Nth = nvec.sum() * self.Delta
        Eth = C.hbar * omega * ((self.x + mc) * nvec).sum() * self.Delta
        Ec = (5.0 / 7.0) * C.mu_TF(Nc, omega) * Nc
        return dict(t=t, omega=omega, mc=mc, mu_c=C.mu_TF(Nc, omega), Nc=Nc, Nth=Nth, N=Nth + Nc,
                    F=Nc / max(Nth + Nc, 1e-300), Eth=Eth, Ec=Ec, E=Eth + Ec, g=g, nvec=nvec, rho=rho)

    # ----- integration
    def run(self, y0, t0, t1, t_eval=None, rtol=1e-6, atol=1e-3, max_step=np.inf, method="RK45"):
        sol = solve_ivp(self.rhs, (t0, t1), y0, t_eval=t_eval, rtol=rtol, atol=atol,
                        max_step=max_step, method=method)
        return sol


def truncated_bose(model: KineticModel, T, mu_over_kT, N_target=None, omega=None, t=0.0, Nc0=0.0):
    """Initial thermal distribution: Bose occupation at temperature T (K) with
    fugacity exp(mu_over_kT) (mu measured from the band bottom), truncated at the
    trap depth.  If N_target is given, the fugacity is adjusted to give that
    number of thermal atoms.  Returns the state vector."""
    omega = omega if omega is not None else model.p.omega(t)
    mc = model.mc(max(Nc0, model.Nc_floor), omega)
    rho, _ = model.tables(mc)
    xt = model.p.trap_depth(t) / (C.hbar * omega) - mc
    if getattr(model, "evap_mode", "3d") == "1d":
        # with one-direction escape the steady state is not sharply truncated;
        # start from the untruncated distribution on the whole grid and let
        # the model remove the atoms above the trap depth at its own rate
        xt = model.x[-1] + 1.0
    beta = C.hbar * omega / (C.kB * T)

    def dist(mu):
        g = np.zeros(model.n)
        arg = beta * model.x[1:] - mu
        g[1:] = 1.0 / np.expm1(np.maximum(arg, 1e-12))
        g[model.x > xt] = 0.0
        return g
    mu = mu_over_kT
    if N_target is not None:
        lo, hi = -40.0, -1e-9
        for _ in range(200):
            mu = 0.5 * (lo + hi)
            Nn = (rho * dist(mu)).sum() * model.Delta
            if Nn > N_target:
                hi = mu
            else:
                lo = mu
    g = dist(mu)
    y0 = np.empty(model.n + 1)
    y0[:model.n] = rho * g
    y0[model.n] = Nc0
    return y0, mu
