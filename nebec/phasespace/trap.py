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
    def __init__(self, U10, U20, w1=35e-6, w2=45e-6, half_angle_deg=20.0, gravity=True, g_eff=None, fx=0.0):
        self.U10, self.U20, self.w1, self.w2 = U10, U20, w1, w2
        th = np.deg2rad(half_angle_deg)
        self.d1 = np.array([np.cos(th), np.sin(th), 0.0])
        self.d2 = np.array([np.cos(th), -np.sin(th), 0.0])
        # effective gravity: the MOT quadrupole gradient (on during the last 6 s)
        # partly levitates |F=1, mF=-1> atoms; g_eff = g - mu_B B'/(2 m) is a parameter
        self.g = (G if gravity else 0.0) if g_eff is None else g_eff
        self.fx = fx     # horizontal force per unit mass along x (the weak axis), e.g. from the quadrupole gradient
        self.s = 1.0                       # power scale factor

    # ---- potential and force, vectorised over an (N,3) array of positions
    def potential(self, r, s=None):
        s = self.s if s is None else s
        r = np.atleast_2d(r)
        a1 = r @ self.d1; a2 = r @ self.d2
        rho1 = np.sum(r * r, axis=1) - a1 * a1
        rho2 = np.sum(r * r, axis=1) - a2 * a2
        return (-s * self.U10 * np.exp(-2 * rho1 / self.w1**2) - s * self.U20 * np.exp(-2 * rho2 / self.w2**2)
                + C.m_Rb * self.g * r[:, 2] - C.m_Rb * self.fx * r[:, 0])

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
        f[:, 0] += C.m_Rb * self.fx
        return f

    # ---- trap characterisation
    def minimum(self, s=None):
        s = self.s if s is None else s
        best = None
        for x0 in (np.array([0.0, 0.0, -1e-6]), np.array([5e-6, 0.0, -5e-6]), np.array([-5e-6, 0.0, -5e-6]), np.array([10e-6, 0.0, -10e-6])):
            res = minimize(lambda x: self.potential(x[None, :], s)[0], x0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-34))
            if best is None or res.fun < best.fun:
                best = res
        res = best
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

    def saddle_depth(self, s=None, half=150e-6, n=96):
        """Effective depth by watershed: the lowest energy at which the basin of
        the trap minimum connects to the boundary of a box of half-size `half`
        (grid of n^3 points).  Robust for arbitrary force directions."""
        from scipy import ndimage
        s = self.s if s is None else s
        r0, U0 = self.minimum(s)
        ax = np.linspace(-half, half, n)
        X, Y, Z = np.meshgrid(ax + r0[0], ax + r0[1], ax + r0[2], indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        U = self.potential(pts, s).reshape(n, n, n) - U0
        i0 = (n // 2, n // 2, n // 2)
        def connected(E):
            lab, _ = ndimage.label(U < E)
            l0 = lab[i0]
            if l0 == 0:
                return False
            b = np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])
            return np.any(b == l0)
        lo, hi = 0.0, float(np.max(U[np.isfinite(U)]))
        if not connected(hi):
            return hi
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if connected(mid):
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def basin(self, s=None, E=None, half=150e-6, n=96):
        """Return a callable mask(points) that is True for points inside the
        connected region U - U0 < E containing the trap minimum (E defaults to
        0.999 of the watershed depth).  Used to confine the initial sampling to
        bound orbits when gravity or a horizontal force open the trap."""
        from scipy import ndimage
        s = self.s if s is None else s
        E = 0.999 * self.saddle_depth(s, half, n) if E is None else E
        r0, U0 = self.minimum(s)
        ax = np.linspace(-half, half, n)
        X, Y, Z = np.meshgrid(ax + r0[0], ax + r0[1], ax + r0[2], indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        U = self.potential(pts, s).reshape(n, n, n) - U0
        lab, _ = ndimage.label(U < E)
        inside = lab == lab[n // 2, n // 2, n // 2]
        d = ax[1] - ax[0]
        def mask(p):
            idx = np.rint((np.atleast_2d(p) - r0 + half) / d).astype(int)
            ok = np.all((idx >= 0) & (idx < n), axis=1)
            idx = np.clip(idx, 0, n - 1)
            return ok & inside[idx[:, 0], idx[:, 1], idx[:, 2]]
        return mask

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

    def min_search_start(self):
        return np.array([0.0, 0.0, -1e-6])


def default_trap(power_ratio=1.0, g_eff=None, fx=0.0):
    """Beam depths at the reference scale s = 1: U10 = 1 uK, U20 = U10/power_ratio
    in units of k_B.  The absolute scale is set later by set_scale_for_omega."""
    U10 = 1e-6 * C.kB
    return CrossedTrap(U10, U10 / power_ratio, g_eff=g_eff, fx=fx)
