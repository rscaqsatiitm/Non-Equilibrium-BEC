"""Direct-simulation Monte Carlo of thermal-thermal collisions (C22) with
optional Bose enhancement in the local-equilibrium approximation.

Bird's no-time-counter scheme on a Cartesian cell grid: in a cell with n
particles of weight alpha and volume V, the number of candidate pairs per
step is  n (n-1)/2 * alpha * sigma * v_rel,max * dt / V ; a candidate pair
collides with probability v_rel / v_rel,max.  The post-collision relative
velocity is isotropic (s-wave).  With Bose enhancement the acceptance is
multiplied by (1 + f(p3)) (1 + f(p4)) / B_max where f is the local Bose
occupation estimated from the cell's density and mean kinetic energy, and
v_rel,max is enlarged by B_max to keep the acceptance below one.
"""
import numpy as np
from scipy.optimize import brentq
from .. import constants as C


def _bose_params(n, ekin_mean):
    """Local (T, z) of an ideal Bose gas with density n and mean kinetic energy
    per particle ekin_mean: n lambda^3 = g_{3/2}(z), ekin = 3/2 kT g_{5/2}/g_{3/2}."""
    k = np.arange(1, 200)
    g32 = lambda z: np.sum(z**k / k**1.5); g52 = lambda z: np.sum(z**k / k**2.5)
    # ratio R = n lambda^3 /(ekin/(3/2 kT))... solve for z from the combination
    # n (2 pi hbar^2/(m kT))^{3/2} = g32(z) with kT = ekin / (1.5 g52/g32)
    def f(z):
        kT = ekin_mean / (1.5 * g52(z) / g32(z))
        lam3 = (2 * np.pi * C.hbar**2 / (C.m_Rb * kT)) ** 1.5
        return n * lam3 - g32(z)
    if f(1.0 - 1e-9) < 0:
        z = 1.0 - 1e-9
    elif f(1e-6) > 0:
        z = 1e-6
    else:
        z = brentq(f, 1e-6, 1.0 - 1e-9)
    kT = ekin_mean / (1.5 * g52(z) / g32(z))
    return kT, z


class DSMC:
    def __init__(self, cell=1.5e-6, bose=True, rng=None, bmax_cap=50.0):
        self.cell = cell
        self.bose = bose
        self.rng = np.random.default_rng() if rng is None else rng
        self.bmax_cap = bmax_cap
        self.n_coll = 0

    def step(self, cloud, dt, origin):
        r, v, alpha = cloud.r, cloud.v, cloud.alpha
        if r.shape[0] < 2:
            return 0
        idx = np.floor((r - origin) / self.cell).astype(np.int64)
        key = (idx[:, 0] * 100003 + idx[:, 1]) * 100003 + idx[:, 2]
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        bounds = np.flatnonzero(np.diff(key_s)) + 1
        starts = np.concatenate(([0], bounds)); ends = np.concatenate((bounds, [len(key_s)]))
        V = self.cell**3
        ncoll = 0
        for a, b in zip(starts, ends):
            n = b - a
            if n < 2:
                continue
            ids = order[a:b]
            vc = v[ids]
            # local Bose parameters
            if self.bose:
                ekin = 0.5 * C.m_Rb * np.mean(np.sum(vc * vc, axis=1))
                kT, z = _bose_params(n * alpha / V, ekin)
                # upper bound of the enhancement for the lowest momenta
                fmax = 1.0 / np.expm1(max(-np.log(z), 1e-6)) if z < 1 else 1.0 / np.expm1(1e-6)
                Bmax = min((1 + fmax) ** 2, self.bmax_cap)
            else:
                Bmax = 1.0
            vmax = 2.0 * np.sqrt(np.max(np.sum(vc * vc, axis=1))) + 1e-9
            ncand = 0.5 * n * (n - 1) * alpha * C.sigma_el * vmax * Bmax * dt / V
            m = int(ncand) + (self.rng.random() < ncand - int(ncand))
            if m == 0:
                continue
            i = self.rng.integers(0, n, m); j = self.rng.integers(0, n - 1, m); j = np.where(j >= i, j + 1, j)
            vi, vj = vc[i], vc[j]
            vrel = vi - vj; vr = np.linalg.norm(vrel, axis=1)
            # isotropic post-collision relative velocity
            cth = 2 * self.rng.random(m) - 1; sth = np.sqrt(1 - cth * cth); ph = 2 * np.pi * self.rng.random(m)
            nvec = np.stack([sth * np.cos(ph), sth * np.sin(ph), cth], axis=1)
            vcm = 0.5 * (vi + vj)
            v3 = vcm + 0.5 * vr[:, None] * nvec; v4 = vcm - 0.5 * vr[:, None] * nvec
            acc = vr / vmax
            if self.bose:
                e3 = 0.5 * C.m_Rb * np.sum(v3 * v3, axis=1); e4 = 0.5 * C.m_Rb * np.sum(v4 * v4, axis=1)
                f3 = 1.0 / np.expm1(np.maximum(e3 / kT - np.log(z), 1e-6)); f4 = 1.0 / np.expm1(np.maximum(e4 / kT - np.log(z), 1e-6))
                acc = acc * np.minimum((1 + f3) * (1 + f4), self.bmax_cap) / Bmax
            ok = self.rng.random(m) < acc
            # apply sequentially to avoid double use of a particle in one step
            used = np.zeros(n, dtype=bool)
            for k in np.flatnonzero(ok):
                if used[i[k]] or used[j[k]]:
                    continue
                used[i[k]] = used[j[k]] = True
                v[ids[i[k]]] = v3[k]; v[ids[j[k]]] = v4[k]
                ncoll += 1
        self.n_coll += ncoll
        return ncoll
