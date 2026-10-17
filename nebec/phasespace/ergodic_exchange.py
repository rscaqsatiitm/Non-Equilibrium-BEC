"""Hybrid condensate exchange: the ergodic C12 collision integral (validated in
nebec.collisions / Bijlsma-Zaremba-Stoof) evaluated on the energy distribution
of the test particles, and realised by removing or creating test particles in
the corresponding energy bins.

Energies: eps = K + U_trap(r) - U0 + 2 g n_c(r); shifted variable
x = (eps - mu_c)/(hbar w_bar) on a uniform grid; occupation g_i = N_i/(rho_i Delta)
with the Hartree-Fock density of states rho_hat(x, m_c) of the harmonic
approximation.  Created particles are placed on the energy shell: position
from p(r) ~ sqrt(eps - U_eff(r)) by rejection in a box, isotropic velocity.
"""
import numpy as np
from .. import constants as C
from ..dos import rho_hat, rho_c_table
from ..collisions import c12


class ErgodicExchange:
    def __init__(self, n_grid=200, x_max=400.0, rng=None, c12_scale=1.0):
        self.n = n_grid
        self.Delta = x_max / (n_grid - 1)
        self.x = np.arange(n_grid) * self.Delta
        self.rng = np.random.default_rng() if rng is None else rng
        self.c12_scale = c12_scale
        self._cache = {}
        self.n_removed = 0; self.n_created = 0

    def tables(self, mc):
        key = round(mc, 3)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            rho = rho_hat(self.x, mc); rho[0] = 0.0
            self._cache[key] = (rho, rho_c_table(self.x, mc))
        return self._cache[key]

    def energies(self, sim, mask):
        cl = sim.cloud
        r = cl.r[mask]; v = cl.v[mask]
        U = sim.trap.potential(r) - sim.U0_now + 2 * C.g_int * sim.cond.density(r)
        return 0.5 * C.m_Rb * np.sum(v * v, axis=1) + U

    def apply(self, sim, dt):
        cl = sim.cloud
        wbar = np.prod(sim.w) ** (1 / 3)
        Nc = max(sim.Nc, sim.Nc_floor)
        mc = C.mu_TF(Nc, wbar) / (C.hbar * wbar)
        rho, Phi = self.tables(mc)
        mask = sim.trapped_mask()
        idx_all = np.flatnonzero(mask)
        eps = self.energies(sim, mask)
        x = (eps - mc * C.hbar * wbar) / (C.hbar * wbar)
        b = np.floor(x / self.Delta + 0.5).astype(int)
        valid = (b >= 1) & (b < self.n)
        counts = np.bincount(b[valid], minlength=self.n).astype(float) * cl.alpha
        g = np.zeros(self.n); g[1:] = counts[1:] / (rho[1:] * self.Delta)
        G0 = C.Gamma0(wbar) * self.c12_scale
        dn, dNc = c12(g, Phi, self.Delta, G0, Nc, False)      # d(rho g)_i/dt per unit x, dN_c/dt
        dN_bins = dn * self.Delta * dt                         # atoms per bin in dt
        # realise: remove from bins with dN<0, create in bins with dN>0
        n_rem_tot = 0; new_r = []; new_v = []
        for i in np.flatnonzero(np.abs(dN_bins) > 1e-12):
            m_atoms = dN_bins[i]
            m = int(abs(m_atoms) / cl.alpha); frac = abs(m_atoms) / cl.alpha - m
            if self.rng.random() < frac:
                m += 1
            if m == 0:
                continue
            if m_atoms < 0:
                cand = idx_all[valid & (b == i)]
                if cand.size == 0:
                    continue
                m = min(m, cand.size)
                pick = self.rng.choice(cand, size=m, replace=False)
                mask_keep = np.ones(cl.r.shape[0], dtype=bool); mask_keep[pick] = False
                cl.r = cl.r[mask_keep]; cl.v = cl.v[mask_keep]
                # indices shift after removal: recompute mapping lazily by restarting the loop bookkeeping
                idx_all = np.flatnonzero(sim.trapped_mask())
                eps = self.energies(sim, sim.trapped_mask()); x = (eps - mc * C.hbar * wbar) / (C.hbar * wbar)
                b = np.floor(x / self.Delta + 0.5).astype(int); valid = (b >= 1) & (b < self.n)
                n_rem_tot += m
            else:
                e_target = (self.x[i] + self.rng.uniform(-0.5, 0.5, size=m) * self.Delta + mc) * C.hbar * wbar
                rr, vv = self._sample_shell(sim, e_target, mc)
                new_r.append(rr); new_v.append(vv)
        if new_r:
            cl.r = np.vstack([cl.r] + new_r); cl.v = np.vstack([cl.v] + new_v)
            self.n_created += sum(len(a) for a in new_r)
        self.n_removed += n_rem_tot
        sim.Nc += dNc * dt
        return dNc * dt

    def _sample_shell(self, sim, e_target, mc):
        """Positions on the energy shell: p(r) ~ sqrt(E - U_eff(r)); box = 4 sigma."""
        w = sim.w; wbar = np.prod(w) ** (1 / 3)
        out_r = np.zeros((len(e_target), 3)); out_v = np.zeros((len(e_target), 3))
        for k, E in enumerate(e_target):
            # rejection sampling in an ellipsoidal box sized by the classical turning points
            amp = np.sqrt(2 * max(E, 1e-32) / (C.m_Rb * w**2))
            for _ in range(200):
                q = self.rng.uniform(-1, 1, size=3) * amp
                r = sim.r0 + q @ sim._axes.T
                U = sim.trap.potential(r[None, :])[0] - sim.U0_now + 2 * C.g_int * sim.cond.density(r[None, :])[0]
                if U < E and self.rng.random() < np.sqrt((E - U) / E):
                    K = E - U
                    n = self.rng.normal(size=3); n /= np.linalg.norm(n)
                    out_r[k] = r; out_v[k] = n * np.sqrt(2 * K / C.m_Rb)
                    break
            else:
                out_r[k] = sim.r0; out_v[k] = self.rng.normal(size=3) * np.sqrt(C.kB * 50e-9 / C.m_Rb)
        return out_r, out_v
