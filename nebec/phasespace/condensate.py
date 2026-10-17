"""Thomas-Fermi condensate in the local harmonic approximation of the trap.

The condensate radius (a few um) is small compared with the beam waists
(35-45 um), so the trap is expanded to second order around its minimum:
V = V0 + (m/2) sum_i w_i^2 q_i^2 in the principal-axis frame (q = R^T (r - r0)).
Then n_c(r) = (mu_c - (m/2) sum w_i^2 q_i^2)/g for positive values,
mu_c = (hbar w_bar/2)(15 N_c a/l)^{2/5}, and the mean-field force on a thermal
atom is -2 g grad n_c = +2 m w_i^2 q_i inside the condensate (the thermal atoms
are pushed out of the condensate).
"""
import numpy as np
from .. import constants as C


class TFCondensate:
    def __init__(self, Nc, omegas, r0, axes):
        self.omegas = np.asarray(omegas, dtype=float)   # principal frequencies (rad/s)
        self.r0 = np.asarray(r0, dtype=float)
        self.axes = np.asarray(axes, dtype=float)       # columns = principal axes (3x3)
        self.set_number(Nc)

    def set_number(self, Nc):
        self.Nc = max(float(Nc), 0.0)
        wbar = np.prod(self.omegas) ** (1 / 3)
        self.mu = C.mu_TF(self.Nc, wbar) if self.Nc > 0 else 0.0
        self.wbar = wbar

    def update_trap(self, omegas, r0, axes):
        self.omegas = np.asarray(omegas, dtype=float); self.r0 = np.asarray(r0, dtype=float); self.axes = np.asarray(axes, dtype=float)
        self.set_number(self.Nc)

    def _q(self, r):
        return (np.atleast_2d(r) - self.r0) @ self.axes

    def density(self, r):
        if self.Nc <= 0:
            return np.zeros(np.atleast_2d(r).shape[0])
        q = self._q(r)
        V = 0.5 * C.m_Rb * np.sum(self.omegas**2 * q * q, axis=1)
        return np.maximum(self.mu - V, 0.0) / C.g_int

    def mean_field_force(self, r):
        """-2 g grad n_c = 2 m w_i^2 q_i (inside), in the lab frame."""
        if self.Nc <= 0:
            return np.zeros_like(np.atleast_2d(r))
        q = self._q(r)
        V = 0.5 * C.m_Rb * np.sum(self.omegas**2 * q * q, axis=1)
        inside = (V < self.mu)[:, None]
        fq = 2 * C.m_Rb * (self.omegas**2 * q) * inside
        return fq @ self.axes.T

    def peak_density(self):
        return self.mu / C.g_int if self.Nc > 0 else 0.0

    def three_body_loss_rate(self):
        return C.L3_cond * (8.0 / 21.0) * self.peak_density() ** 2 if self.Nc > 0 else 0.0
