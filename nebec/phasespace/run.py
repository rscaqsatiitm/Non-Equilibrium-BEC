"""Driver for the phase-space simulation."""
import numpy as np
from .. import constants as C
from .trap import CrossedTrap, default_trap
from .particles import Cloud
from .condensate import TFCondensate
from .collisions_nb import collision_step
from .ergodic_exchange import ErgodicExchange


class Simulation:
    def __init__(self, trap, omega_bar, cloud, Nc0=0.0, cell=1.5e-6, bose=True, fcap=60.0, Av_factor=3.5,
                 dv_factor=0.5, T_ref=100e-9, gamma=lambda t: 0.0, L3=C.L3_cond, dt=20e-6, seed=0, c12=True, c22=True, smooth=True, vtol_factor=1.0, Nc_floor=1.0, exchange='ergodic', exchange_every=5, scale_table=None):
        self.trap, self.omega_bar, self.cloud = trap, omega_bar, cloud
        if scale_table is not None:      # reuse the (expensive) trap table of another Simulation on the same trap
            for k in ('_tab_f', '_tab_s', '_tab_w', '_tab_r0', '_tab_axes', '_tab_depth'):
                setattr(self, k, getattr(scale_table, k))
        self.cell, self.bose, self.fcap, self.Av_factor = cell, bose, fcap, Av_factor
        self.dv = dv_factor * np.sqrt(C.kB * T_ref / C.m_Rb)     # velocity-cell side for the occupation estimate
        self.smooth = smooth; self.vtol = vtol_factor * np.sqrt(C.kB * T_ref / C.m_Rb)
        self.Nc_floor = Nc_floor
        self.exchange = exchange          # 'ergodic' (hybrid) or 'montecarlo' (Jackson-Zaremba)
        self.exchange_every = exchange_every
        self.ergx = ErgodicExchange(rng=np.random.default_rng(seed + 1))
        self.gamma, self.L3, self.dt = gamma, L3, dt
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.c12_on, self.c22_on = c12, c22
        self.t = 0.0
        self.stats = dict(n_c22=0, n_out=0, n_in=0, n_evap=0.0, n_loss=0.0)
        self._set_trap(0.0)
        self.cond = TFCondensate(Nc0, self.w, self.r0, self._axes)
        self.Nc = float(Nc0)

    def _build_scale_table(self):
        """Tabulate the power scale s and the trap properties against the
        geometric-mean frequency, so that the ramp costs no root finding."""
        fb = np.linspace(40.0, 220.0, 61) * 2 * np.pi
        self._tab_f = fb; self._tab_s = []; self._tab_w = []; self._tab_r0 = []; self._tab_axes = []; self._tab_depth = []
        for f in fb:
            self.trap.set_scale_for_omega(f)
            w, r0, U0 = self.trap.frequencies()
            h = 0.2e-6; H = np.zeros((3, 3))
            p = lambda x: self.trap.potential(x[None, :])[0]
            for i in range(3):
                for j in range(3):
                    ei = np.zeros(3); ej = np.zeros(3); ei[i] = h; ej[j] = h
                    H[i, j] = (p(r0 + ei + ej) - p(r0 + ei - ej) - p(r0 - ei + ej) + p(r0 - ei - ej)) / (4 * h * h)
            wv, vec = np.linalg.eigh(H)
            self._tab_s.append(self.trap.s); self._tab_w.append(np.sqrt(np.maximum(wv, 0) / C.m_Rb)); self._tab_r0.append(r0); self._tab_axes.append(vec); self._tab_depth.append(self.trap.saddle_depth())
        self._tab_s = np.array(self._tab_s); self._tab_w = np.array(self._tab_w); self._tab_r0 = np.array(self._tab_r0); self._tab_depth = np.array(self._tab_depth)

    def _set_trap(self, t):
        if not hasattr(self, "_tab_s"):
            self._build_scale_table()
        f = self.omega_bar(t)
        self.trap.s = float(np.interp(f, self._tab_f, self._tab_s))
        self.w = np.array([np.interp(f, self._tab_f, self._tab_w[:, i]) for i in range(3)])
        self.r0 = np.array([np.interp(f, self._tab_f, self._tab_r0[:, i]) for i in range(3)])
        k = int(np.argmin(np.abs(self._tab_f - f))); self._axes = self._tab_axes[k]
        self.depth = float(np.interp(f, self._tab_f, self._tab_depth))
        self.U0_now = self.trap.potential(self.r0[None, :])[0]
        return

    def _bin(self):
        r = self.cloud.r
        origin = self.r0 - 60e-6
        idx = np.floor((r - origin) / self.cell).astype(np.int64)
        key = (idx[:, 0] * 4096 + idx[:, 1]) * 4096 + idx[:, 2]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        starts = np.concatenate(([0], bounds)).astype(np.int64); ends = np.concatenate((bounds, [len(ks)])).astype(np.int64)
        return order.astype(np.int64), starts, ends

    def step(self, trap_update_every=50, i_step=0):
        dt = self.dt
        if i_step % trap_update_every == 0:
            self._set_trap(self.t)
            self.cond.update_trap(self.w, self.r0, self._axes)
        self.cloud.step(dt, self.trap, self.trap.s, self.cond if self.Nc > self.Nc_floor else None)
        alive = np.ones(self.cloud.r.shape[0], dtype=np.bool_)
        order, starts, ends = self._bin()
        nc_at = self.cond.density(self.cloud.r) if self.c12_on else np.zeros(self.cloud.r.shape[0])
        max_new = 4000
        new_r = np.zeros((max_new, 3)); new_v = np.zeros((max_new, 3))
        n22, n_out, n_in, n_new = collision_step(self.cloud.r, self.cloud.v, alive, order, starts, ends, self.cloud.alpha, self.cell, self.dv, dt,
                                                 nc_at, self.bose, self.fcap, self.Av_factor, self.seed + i_step, new_r, new_v, max_new,
                                                 self.c22_on, self.c12_on and self.exchange == 'montecarlo', self.smooth, self.vtol)
        self.stats["n_c22"] += n22; self.stats["n_out"] += n_out; self.stats["n_in"] += n_in
        self.Nc += (n_in - n_out) * self.cloud.alpha
        if self.c12_on and self.Nc < self.Nc_floor:
            self.Nc = self.Nc_floor          # seed: the lowest mode is never empty (spontaneous growth)
        if n_new:
            self.cloud.r = np.vstack([self.cloud.r, new_r[:n_new]]); self.cloud.v = np.vstack([self.cloud.v, new_v[:n_new]])
            alive = np.concatenate([alive, np.ones(n_new, dtype=np.bool_)])
        if (~alive).any():
            self.cloud.r = self.cloud.r[alive]; self.cloud.v = self.cloud.v[alive]
        if self.c12_on and self.exchange == 'ergodic' and i_step % self.exchange_every == 0:
            self.ergx.apply(self, dt * self.exchange_every)
            if self.Nc < self.Nc_floor:
                self.Nc = self.Nc_floor
            self.stats["n_out"] = self.ergx.n_removed; self.stats["n_in"] = self.ergx.n_created
        # uniform loss (K + background) on thermal particles and condensate
        g = self.gamma(self.t)
        if g > 0:
            keep = self.rng.random(self.cloud.r.shape[0]) >= g * dt
            self.stats["n_loss"] += (~keep).sum() * self.cloud.alpha
            self.cloud.r = self.cloud.r[keep]; self.cloud.v = self.cloud.v[keep]
            self.Nc *= (1 - g * dt)
        if self.Nc > 0:
            self.Nc *= (1 - self.cond.three_body_loss_rate() * dt)
        self.cond.set_number(self.Nc)
        if i_step % 25 == 0:
            self._remove_escaped()
        self.t += dt

    def _remove_escaped(self, r_max=70e-6):
        d = self.cloud.r - self.r0
        gone = (np.abs(d[:, 0]) > r_max) | (np.abs(d[:, 1]) > r_max) | (np.abs(d[:, 2]) > r_max)
        n = gone.sum()
        if n:
            self.stats["n_evap"] += n * self.cloud.alpha
            self.cloud.r = self.cloud.r[~gone]; self.cloud.v = self.cloud.v[~gone]

    def run(self, t_end, callback=None, every=0.01, trap_update_every=50):
        i = 0; next_cb = self.t
        while self.t < t_end - 1e-12:
            self.step(trap_update_every, i); i += 1
            if callback is not None and self.t >= next_cb:
                callback(self); next_cb += every
        return self

    # ---- diagnostics
    def trapped_mask(self, radius=35e-6):
        """Particles still in the trap region (excludes atoms falling away
        after crossing the saddle, which would dominate the kinetic temperatures)."""
        d = self.cloud.r - self.r0
        return (np.abs(d[:, 0]) < radius) & (np.abs(d[:, 1]) < radius) & (np.abs(d[:, 2]) < radius)

    def trapped_cloud(self):
        from .particles import Cloud
        m = self.trapped_mask()
        return Cloud(self.cloud.r[m], self.cloud.v[m], self.cloud.alpha)

    def summary(self):
        cl = self.trapped_cloud()
        Tk = cl.axis_temperatures()
        Nth = cl.N
        return dict(t=self.t, Nth=Nth, Nc=self.Nc, N=Nth + self.Nc, F=self.Nc / max(Nth + self.Nc, 1e-9), Tx=Tk[0], Ty=Tk[1], Tz=Tk[2],
                    T_kin=Tk.mean(), mu_c=self.cond.mu, f_bar=np.prod(self.w) ** (1 / 3) / (2 * np.pi), depth=self.depth, **self.stats)
