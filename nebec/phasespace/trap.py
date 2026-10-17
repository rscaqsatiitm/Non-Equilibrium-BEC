"""Crossed-beam optical dipole trap with gravity.

Geometry from the PRL: two beams (1070 nm, waist 35 um; 1550 nm, waist 45 um)
crossing at 40 degrees; we take both beams in the horizontal plane, symmetric
about x (half-angle theta = 20 deg), gravity along -z.  Beam divergence is
neglected (Rayleigh ranges of mm versus a cloud of 10 um).

U(r) = -U1 exp(-2 rho1^2/w1^2) - U2 exp(-2 rho2^2/w2^2) + m g z,
rho_i = distance from beam axis i.  The depths scale with the beam powers,
U_i(t) = U_i0 s(t), and s(t) is chosen so that the geometric-mean frequency at
the trap minimum (with gravity) follows the calibrated ramp.

Without gravity the curvatures are k_i = 4 U_i/(m w_i^2) and
w_z^2 = k1 + k2, w_y^2 = (k1 + k2) cos^2 theta, w_x^2 = (k1 + k2) sin^2 theta,
so w_x : w_y : w_z = sin theta : cos theta : 1 (0.34 : 0.94 : 1 at 40 degrees),
whose geometric mean at w_z = 2 pi 120 Hz is 82 Hz, close to the 80 Hz of the
calibrated ramp at t = 2 s.
"""
import numpy as np
from scipy.optimize import minimize
from .. import constants as C

G = 9.81


class CrossedTrap:
    def __init__(self, U10, U20, w1=35e-6, w2=45e-6, half_angle_deg=20.0, gravity=True, g_eff=None):
        self.U10, self.U20, self.w1, self.w2 = U10, U20, w1, w2
        th = np.deg2rad(half_angle_deg)
        self.d1 = np.array([np.cos(th), np.sin(th), 0.0])
        self.d2 = np.array([np.cos(th), -np.sin(th), 0.0])
        # effective gravity: the MOT quadrupole gradient (on during the last 6 s)
        # partly levitates |F=1, mF=-1> atoms; g_eff = g - mu_B B'/(2 m) is a parameter
        self.g = (G if gravity else 0.0) if g_eff is None else g_eff
        self.s = 1.0                       # power scale factor

    # ---- potential and force, vectorised over an (N,3) array of positions
    def potential(self, r, s=None):
        s = self.s if s is None else s
        r = np.atleast_2d(r)
        a1 = r @ self.d1; a2 = r @ self.d2
        rho1 = np.sum(r * r, axis=1) - a1 * a1
        rho2 = np.sum(r * r, axis=1) - a2 * a2
        return (-s * self.U10 * np.exp(-2 * rho1 / self.w1**2) - s * self.U20 * np.exp(-2 * rho2 / self.w2**2)
                + C.m_Rb * self.g * r[:, 2])

    def force(self, r, s=None):
        s = self.s if s is None else s
        r = np.atleast_2d(r)
        a1 = r @ self.d1; a2 = r @ self.d2
        perp1 = r - np.outer(a1, self.d1); perp2 = r - np.outer(a2, self.d2)
        rho1 = np.sum(perp1 * perp1, axis=1); rho2 = np.sum(perp2 * perp2, axis=1)
        e1 = np.exp(-2 * rho1 / self.w1**2); e2 = np.exp(-2 * rho2 / self.w2**2)
        # dU/dr = U_i s (4/w^2) perp * exp(...)  -> force = -grad U
        f = -(s * self.U10 * 4 / self.w1**2 * e1)[:, None] * perp1 - (s * self.U20 * 4 / self.w2**2 * e2)[:, None] * perp2
        f[:, 2] -= C.m_Rb * self.g
        return f

    # ---- trap characterisation
    def minimum(self, s=None):
        s = self.s if s is None else s
        res = minimize(lambda x: self.potential(x[None, :], s)[0], np.array([0.0, 0.0, -1e-6]), method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-34))
        return res.x, res.fun

    def frequencies(self, s=None, h=0.2e-6):
        """Eigenfrequencies (rad/s) of the Hessian at the minimum, and the minimum."""
        s = self.s if s is None else s
        r0, U0 = self.minimum(s)
        H = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                e_i = np.zeros(3); e_j = np.zeros(3); e_i[i] = h; e_j[j] = h
                H[i, j] = (self.potential((r0 + e_i + e_j)[None, :], s)[0] - self.potential((r0 + e_i - e_j)[None, :], s)[0]
                           - self.potential((r0 - e_i + e_j)[None, :], s)[0] + self.potential((r0 - e_i - e_j)[None, :], s)[0]) / (4 * h * h)
        w2 = np.linalg.eigvalsh(H) / C.m_Rb
        return np.sqrt(np.maximum(w2, 0.0)), r0, U0

    def saddle_depth(self, s=None, n=4001):
        """Effective depth: lowest barrier for escape.  With gravity the escape
        is downward along z through the saddle below the minimum; we scan the
        potential along the vertical line through the minimum and along the
        two beam-perpendicular directions and return the smallest barrier."""
        s = self.s if s is None else s
        r0, U0 = self.minimum(s)
        best = np.inf
        for d in (np.array([0, 0, -1.0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, -1.0, 0])):
            lam = np.linspace(0, 200e-6, n)
            U = self.potential(r0[None, :] + lam[:, None] * d[None, :], s)
            # barrier = max of U along the ray before it drops below U0 again (or the overall max)
            best = min(best, U.max() - U0)
        return best

    def set_scale_for_omega(self, omega_bar_target, s_lo=0.01, s_hi=100.0):
        """Choose the power scale s such that the geometric-mean frequency at the
        (gravity-shifted) minimum equals omega_bar_target."""
        from scipy.optimize import brentq
        def f(ls):
            w, _, _ = self.frequencies(np.exp(ls))
            return np.log(np.prod(w) ** (1 / 3) / omega_bar_target)
        ls = brentq(f, np.log(s_lo), np.log(s_hi))
        self.s = np.exp(ls)
        return self.s


def default_trap(power_ratio=1.0, g_eff=None):
    """Beam depths at the reference scale s = 1: U10 = 1 uK, U20 = U10/power_ratio
    in units of k_B.  The absolute scale is set later by set_scale_for_omega."""
    U10 = 1e-6 * C.kB
    return CrossedTrap(U10, U10 / power_ratio, g_eff=g_eff)
