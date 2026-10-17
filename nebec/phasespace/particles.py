"""Test-particle representation of the thermal cloud and its free streaming.

Each test particle carries a weight alpha (atoms per particle).  Positions and
velocities are advanced with velocity-Verlet in the trap potential plus the
condensate mean field.  Particles that leave the trap region (potential energy
above the asymptotic value and position beyond the escape radius, or simply
beyond r_max) are removed and counted as evaporated.
"""
import numpy as np
from .. import constants as C


class Cloud:
    def __init__(self, r, v, alpha=1.0):
        self.r = np.array(r, dtype=float)
        self.v = np.array(v, dtype=float)
        self.alpha = alpha
        self.n_evap = 0.0

    @property
    def N(self):
        return self.alpha * self.r.shape[0]

    @classmethod
    def thermal(cls, n_particles, T, trap, s, alpha=1.0, rng=None, mu_over_kT=None, cond=None, U_max=None, r_max=70e-6, in_basin=None):
        """Sample a (classical or Bose, via rejection on energy) distribution in the
        trap at temperature T: positions from exp(-U/kT) by rejection within a
        box, velocities Maxwellian.  Bose statistics are imposed approximately by
        rejection with the factor 1/(z^-1 exp(E/kT) - 1) / exp(-E/kT) when
        mu_over_kT is given.  Candidates outside the bound region (farther than
        r_max from the minimum, or above U_max relative to the minimum) get zero
        weight: with gravity or a horizontal force the potential falls without
        bound beyond the saddle and would otherwise dominate the rejection
        normalisation."""
        rng = np.random.default_rng() if rng is None else rng
        r0, U0 = trap.minimum(s)
        def bound(cand, U):
            ok = np.sum((cand - r0) ** 2, axis=1) < r_max ** 2
            if U_max is not None:
                ok &= U < U_max
            if in_basin is not None:
                ok &= in_basin(cand)
            return ok
        w, _, _ = trap.frequencies(s)
        sig = np.sqrt(C.kB * T / (C.m_Rb * w**2))
        R, V = [], []
        need = n_particles
        while need > 0:
            m = int(need * 3) + 100
            cand = r0 + rng.normal(size=(m, 3)) * (4 * sig)
            U = trap.potential(cand, s) - U0
            if cond is not None:
                U = U + 2 * C.g_int * cond.density(cand)
            # importance sampling: proposal gaussian q(r) ~ exp(-r^2/(2 (4 sig)^2)); accept with exp(-U/kT)/q * const
            q = np.exp(-np.sum(((cand - r0) / (4 * sig)) ** 2, axis=1) / 2)
            acc = np.where(bound(cand, U), np.exp(-U / (C.kB * T)) / q, 0.0)
            acc = acc / acc.max()
            keep = rng.random(m) < acc
            R.append(cand[keep]); need -= keep.sum()
        r = np.vstack(R)[:n_particles]
        v = rng.normal(size=(n_particles, 3)) * np.sqrt(C.kB * T / C.m_Rb)
        if mu_over_kT is not None:
            # Bose correction on the total energy measured from the trap bottom:
            # accept with [1/(e^{E/kT - mu} - 1)] / e^{-E/kT}, normalised by its
            # maximum; repeat the proposal until n_particles are accepted
            Racc, Vacc = [], []
            got = 0
            while got < n_particles:
                m = max(int((n_particles - got) * 4), 2000)
                cand = r0 + rng.normal(size=(m, 3)) * (4 * sig)
                U = trap.potential(cand, s) - U0
                if cond is not None:
                    U = U + 2 * C.g_int * cond.density(cand)
                q = np.exp(-np.sum(((cand - r0) / (4 * sig)) ** 2, axis=1) / 2)
                vv = rng.normal(size=(m, 3)) * np.sqrt(C.kB * T / C.m_Rb)
                E = 0.5 * C.m_Rb * np.sum(vv * vv, axis=1) + U
                x = E / (C.kB * T)
                wgt = (1.0 / np.expm1(np.maximum(x - mu_over_kT, 1e-9))) / q      # target / proposal (position part); velocity proposal cancels e^{-K/kT}
                wgt = wgt * np.exp(0.5 * C.m_Rb * np.sum(vv * vv, axis=1) / (C.kB * T))   # undo the Maxwell proposal, leaving the Bose factor
                wgt = np.where(bound(cand, U), wgt, 0.0)
                wmax = (1.0 / np.expm1(max(-mu_over_kT, 1e-9)))                      # bound at E = 0, q <= 1
                keep = rng.random(m) < wgt / (wmax / q.min() if q.min() > 0 else wmax) * 0 + np.minimum(wgt / wgt.max(), 1.0)
                Racc.append(cand[keep]); Vacc.append(vv[keep]); got += keep.sum()
            r = np.vstack(Racc)[:n_particles]; v = np.vstack(Vacc)[:n_particles]
        return cls(r, v, alpha)

    def step(self, dt, trap, s, cond=None):
        """Velocity-Verlet step with trap + condensate mean-field force."""
        f = trap.force(self.r, s)
        if cond is not None:
            f = f + cond.mean_field_force(self.r)
        a = f / C.m_Rb
        self.v += 0.5 * dt * a
        self.r += dt * self.v
        f = trap.force(self.r, s)
        if cond is not None:
            f = f + cond.mean_field_force(self.r)
        self.v += 0.5 * dt * f / C.m_Rb

    def remove_escaped(self, trap, s, r_max=150e-6):
        """Remove particles far outside the trap (beyond r_max from the minimum
        or below it by more than r_max)."""
        r0, _ = trap.minimum(s)
        d = self.r - r0
        gone = (np.abs(d[:, 0]) > r_max) | (np.abs(d[:, 1]) > r_max) | (d[:, 2] < -r_max) | (d[:, 2] > r_max)
        n = gone.sum()
        if n:
            self.n_evap += n * self.alpha
            self.r = self.r[~gone]; self.v = self.v[~gone]
        return n

    # ---- diagnostics
    def energies(self, trap, s, cond=None):
        U = trap.potential(self.r, s)
        if cond is not None:
            U = U + 2 * C.g_int * cond.density(self.r)
        return 0.5 * C.m_Rb * np.sum(self.v * self.v, axis=1), U

    def axis_temperatures(self):
        """k_B T_i = m <v_i^2> (kinetic temperatures per axis)."""
        return C.m_Rb * np.mean(self.v * self.v, axis=0) / C.kB
