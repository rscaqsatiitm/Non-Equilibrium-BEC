"""Numba kernels for the Monte Carlo collisions (Jackson & Zaremba, PRA 66,
033606 (2002), arXiv:cond-mat/0205421).

Occupation numbers f(r, v) are estimated by counting test particles in
phase-space cells: a spatial cell (side `cell`) times a velocity cell (side
`dv`), f = alpha N_6D (2 pi hbar)^3 / (m^3 cell^3 dv^3).  The same estimate is
used for the Bose enhancement of C22 and for both C12 processes, and the "in"
partner is a test particle taken from the velocity cell of v4, so that P_in
vanishes when no partner exists (as in JZ).

C22 (Eq. 31, NTC pair selection): P_ij = n sigma |v_i - v_j| (1+f3)(1+f4) dt
C12 out (Eq. 35): P_out = n_c sigma v_r^out (1 + f3 + f4) dt
C12 in (Eq. 37):  P_in = n_c sigma A_v f4 dt / (pi v_r)
"""
import numpy as np
from numba import njit

HBAR = 1.054571817e-34
M = 86.909180 * 1.66053906660e-27
A0 = 5.29177210903e-11
G_INT = 4 * np.pi * HBAR**2 * (100.4 * A0) / M
SIGMA = 8 * np.pi * (100.4 * A0) ** 2
H3 = (2 * np.pi * HBAR) ** 3


@njit(cache=True)
def _vkey(vx, vy, vz, dv):
    ix = int(np.floor(vx / dv)) + 512; iy = int(np.floor(vy / dv)) + 512; iz = int(np.floor(vz / dv)) + 512
    return (ix * 1024 + iy) * 1024 + iz


@njit(cache=True)
def _count(keys_sorted, a, b, key):
    """Number of entries equal to key in keys_sorted[a:b] (sorted), and the index of the first."""
    lo = a; hi = b
    while lo < hi:
        mid = (lo + hi) // 2
        if keys_sorted[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    first = lo
    hi = b
    while lo < hi:
        mid = (lo + hi) // 2
        if keys_sorted[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo - first, first


@njit(cache=True)
def _g32(z):
    s = 0.0
    for k in range(1, 80):
        s += z**k / k**1.5
    return s


@njit(cache=True)
def _g52(z):
    s = 0.0
    for k in range(1, 80):
        s += z**k / k**2.5
    return s


# tables for the inversion (n, ekin) -> (kT, z): with x = n (2 pi hbar^2/(m ekin))^{3/2}
# one has x = h(z) = g_{3/2}(z)^{5/2} / [1.5 g_{5/2}(z)]^{3/2}, monotonic in z.
_ZTAB = np.concatenate((np.linspace(1e-6, 0.9, 400, endpoint=False), 1 - np.logspace(-1, -7, 300)))
_K = np.arange(1, 400)
_G32T = np.array([np.sum(z**_K / _K**1.5) for z in _ZTAB])
_G52T = np.array([np.sum(z**_K / _K**2.5) for z in _ZTAB])
_HT = _G32T ** 2.5 / (1.5 * _G52T) ** 1.5


@njit(cache=True)
def bose_params(n, ekin):
    """(kT, z) of an ideal Bose gas with density n and mean kinetic energy ekin
    per particle, by table inversion of n lambda^3 = g_{3/2}(z),
    ekin = 1.5 kT g_{5/2}/g_{3/2}."""
    x = n * (2 * np.pi * HBAR**2 / (M * ekin)) ** 1.5
    if x <= _HT[0]:
        z = _ZTAB[0]; ratio = 1.5 * _G52T[0] / _G32T[0]
    elif x >= _HT[-1]:
        z = _ZTAB[-1]; ratio = 1.5 * _G52T[-1] / _G32T[-1]
    else:
        lo = 0; hi = _HT.shape[0] - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _HT[mid] <= x:
                lo = mid
            else:
                hi = mid
        w = (x - _HT[lo]) / (_HT[hi] - _HT[lo])
        z = _ZTAB[lo] + w * (_ZTAB[hi] - _ZTAB[lo])
        ratio = 1.5 * (_G52T[lo] + w * (_G52T[hi] - _G52T[lo])) / (_G32T[lo] + w * (_G32T[hi] - _G32T[lo]))
    kT = ekin / ratio
    return kT, z


@njit(cache=True)
def _fs_ex(vx, vy, vz, kT, z, fcap, f_ex, vcut):
    """Smooth Bose occupation plus the low-momentum excess."""
    e = 0.5 * M * (vx * vx + vy * vy + vz * vz)
    f = _fsmooth(e, kT, z, fcap)
    if vx * vx + vy * vy + vz * vz < vcut * vcut:
        f += f_ex
    return f if f < fcap else fcap


@njit(cache=True)
def _fsmooth(e, kT, z, fcap):
    x = e / kT - np.log(z)
    if x < 1e-6:
        x = 1e-6
    f = 1.0 / (np.exp(x) - 1.0)
    return f if f < fcap else fcap


@njit(cache=True)
def collision_step(r, v, alive, order, starts, ends, alpha, cell, dv, dt, nc_at, bose, fcap, Av_factor, seed,
                   new_r, new_v, max_new, do_c22, do_c12, smooth, vtol):
    """smooth=True: occupation numbers from a local Bose fit to the cell's density
    and mean kinetic energy (continuous estimate); the "in" partner is the
    particle of the cell closest to v4 within vtol.  smooth=False: 6D-cell counting."""
    """One Monte Carlo step over all spatial cells.  Particles in each cell are
    additionally sorted by velocity cell (local arrays).  Returns
    (n_c22, n_out, n_in, n_new)."""
    np.random.seed(seed)
    cellvol = cell ** 3
    fnorm = alpha * H3 / (M ** 3 * cellvol * dv ** 3)      # occupation per test particle in a 6D cell
    n22 = 0; n_out = 0; n_in = 0; n_new = 0
    for c in range(starts.shape[0]):
        a = starts[c]; b = ends[c]
        n = b - a
        if n < 1:
            continue
        # local velocity-cell keys, sorted
        ids = np.empty(n, dtype=np.int64); vk = np.empty(n, dtype=np.int64)
        cnt = 0
        for k in range(a, b):
            i = order[k]
            if alive[i]:
                ids[cnt] = i; vk[cnt] = _vkey(v[i, 0], v[i, 1], v[i, 2], dv); cnt += 1
        if cnt == 0:
            continue
        ids = ids[:cnt]; vk = vk[:cnt]
        srt = np.argsort(vk); ids = ids[srt]; vk = vk[srt]
        n = cnt
        dens = n * alpha / cellvol
        vmax2 = 0.0; vth2 = 0.0
        for k in range(n):
            i = ids[k]
            v2 = v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2
            vth2 += v2
            if v2 > vmax2:
                vmax2 = v2
        vth = np.sqrt(vth2 / (3.0 * n))
        kT = 1.0; zloc = 0.5; f_ex = 0.0; vcut = 0.5 * vth
        if bose and smooth:
            kT, zloc = bose_params(dens, 0.5 * M * vth2 / n)
            # oversaturation: the local Bose fit saturates at z = 1 and cannot
            # represent an excess population of the lowest momenta; estimate it
            # by counting the particles with |v| < vcut against the Bose expectation
            nlow = 0
            for k in range(n):
                i = ids[k]
                if v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2 < vcut * vcut:
                    nlow += 1
            # expected fraction: int_0^vcut v^2 f dv / int_0^inf v^2 f dv (20-point quadrature)
            num = 0.0; den = 0.0
            vmaxq = 8.0 * np.sqrt(kT / M)
            nq = 160
            dvq = vmaxq / nq
            for q in range(nq):
                vq = (q + 0.5) * dvq
                fq = _fsmooth(0.5 * M * vq * vq, kT, zloc, 1e9) * vq * vq
                den += fq
                if vq < vcut:
                    num += fq
            nexp = n * num / max(den, 1e-300)
            excess = nlow - nexp
            if excess > 0.0:
                f_ex = excess * alpha * H3 / (M ** 3 * cellvol * (4.0 / 3.0) * np.pi * vcut ** 3)
                if f_ex > fcap:
                    f_ex = fcap
        # ---------------- C22
        if do_c22 and n >= 2:
            Bmax = 1.0
            if bose and smooth:
                fmax = _fsmooth(0.0, kT, zloc, fcap) + f_ex
                if fmax > fcap:
                    fmax = fcap
                Bmax = (1.0 + fmax) ** 2
                if Bmax > fcap:
                    Bmax = fcap
            elif bose:
                # bound on the enhancement: largest 6D occupancy in the cell
                best = 0; run = 1
                for k in range(1, n):
                    if vk[k] == vk[k - 1]:
                        run += 1
                    else:
                        if run > best:
                            best = run
                        run = 1
                if run > best:
                    best = run
                fmax = best * fnorm
                Bmax = (1.0 + fmax) ** 2
                if Bmax > fcap:
                    Bmax = fcap
            vmax = 2.0 * np.sqrt(vmax2) + 1e-9
            ncand = 0.5 * n * (n - 1) * alpha * SIGMA * vmax * Bmax * dt / cellvol
            m = int(ncand)
            if np.random.random() < ncand - m:
                m += 1
            for _ in range(m):
                ki = np.random.randint(0, n); kj = np.random.randint(0, n)
                i = ids[ki]; j = ids[kj]
                if i == j or not alive[i] or not alive[j]:
                    continue
                dvx = v[i, 0] - v[j, 0]; dvy = v[i, 1] - v[j, 1]; dvz = v[i, 2] - v[j, 2]
                vr = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
                cth = 2.0 * np.random.random() - 1.0; sth = np.sqrt(1.0 - cth * cth); ph = 2 * np.pi * np.random.random()
                nx = sth * np.cos(ph); ny = sth * np.sin(ph); nz = cth
                cx = 0.5 * (v[i, 0] + v[j, 0]); cy = 0.5 * (v[i, 1] + v[j, 1]); cz = 0.5 * (v[i, 2] + v[j, 2])
                v3x = cx + 0.5 * vr * nx; v3y = cy + 0.5 * vr * ny; v3z = cz + 0.5 * vr * nz
                v4x = cx - 0.5 * vr * nx; v4y = cy - 0.5 * vr * ny; v4z = cz - 0.5 * vr * nz
                acc = vr / vmax
                if bose:
                    if smooth:
                        B = (1.0 + _fs_ex(v3x, v3y, v3z, kT, zloc, fcap, f_ex, vcut)) * (1.0 + _fs_ex(v4x, v4y, v4z, kT, zloc, fcap, f_ex, vcut))
                    else:
                        c3, _ = _count(vk, 0, n, _vkey(v3x, v3y, v3z, dv)); c4, _ = _count(vk, 0, n, _vkey(v4x, v4y, v4z, dv))
                        B = (1.0 + c3 * fnorm) * (1.0 + c4 * fnorm)
                    if B > fcap:
                        B = fcap
                    acc *= B / Bmax
                if np.random.random() < acc:
                    v[i, 0] = v3x; v[i, 1] = v3y; v[i, 2] = v3z
                    v[j, 0] = v4x; v[j, 1] = v4y; v[j, 2] = v4z
                    n22 += 1
        # ---------------- C12
        if do_c12:
            for k in range(n):
                i = ids[k]
                if not alive[i]:
                    continue
                nc = nc_at[i]
                if nc <= 0.0:
                    continue
                vx = v[i, 0]; vy = v[i, 1]; vz = v[i, 2]
                v2 = vx * vx + vy * vy + vz * vz; vr = np.sqrt(v2)
                # out
                p_out = 0.0
                vout2 = v2 - 4 * G_INT * nc / M
                if vout2 > 0.0:
                    vout = np.sqrt(vout2)
                    cth = 2.0 * np.random.random() - 1.0; sth = np.sqrt(1.0 - cth * cth); ph = 2 * np.pi * np.random.random()
                    nx = sth * np.cos(ph); ny = sth * np.sin(ph); nz = cth
                    o3x = 0.5 * vx + 0.5 * vout * nx; o3y = 0.5 * vy + 0.5 * vout * ny; o3z = 0.5 * vz + 0.5 * vout * nz
                    o4x = 0.5 * vx - 0.5 * vout * nx; o4y = 0.5 * vy - 0.5 * vout * ny; o4z = 0.5 * vz - 0.5 * vout * nz
                    B = 1.0
                    if bose:
                        if smooth:
                            B = 1.0 + _fs_ex(o3x, o3y, o3z, kT, zloc, fcap, f_ex, vcut) + _fs_ex(o4x, o4y, o4z, kT, zloc, fcap, f_ex, vcut)
                        else:
                            c3, _ = _count(vk, 0, n, _vkey(o3x, o3y, o3z, dv)); c4, _ = _count(vk, 0, n, _vkey(o4x, o4y, o4z, dv))
                            B = 1.0 + (c3 + c4) * fnorm
                        if B > fcap:
                            B = fcap
                    p_out = nc * SIGMA * vout * B * dt
                # in
                p_in = 0.0; c4 = 0; first4 = 0
                if vr > 1e-9:
                    ux = vx / vr; uy = vy / vr; uz = vz / vr
                    if abs(ux) < 0.9:
                        ax, ay, az = 1.0, 0.0, 0.0
                    else:
                        ax, ay, az = 0.0, 1.0, 0.0
                    e1x = ay * uz - az * uy; e1y = az * ux - ax * uz; e1z = ax * uy - ay * ux
                    nrm = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z); e1x /= nrm; e1y /= nrm; e1z /= nrm
                    e2x = uy * e1z - uz * e1y; e2y = uz * e1x - ux * e1z; e2z = ux * e1y - uy * e1x
                    Rv = Av_factor * vth; Av = np.pi * Rv * Rv
                    rad = Rv * np.sqrt(np.random.random()); ang = 2 * np.pi * np.random.random()
                    tx = rad * (np.cos(ang) * e1x + np.sin(ang) * e2x); ty = rad * (np.cos(ang) * e1y + np.sin(ang) * e2y); tz = rad * (np.cos(ang) * e1z + np.sin(ang) * e2z)
                    shift = G_INT * nc / (M * vr)
                    i4x = tx + shift * ux; i4y = ty + shift * uy; i4z = tz + shift * uz
                    i3x = vx + tx + shift * ux; i3y = vy + ty + shift * uy; i3z = vz + tz + shift * uz
                    if smooth:
                        e4 = 0.5 * M * (i4x * i4x + i4y * i4y + i4z * i4z)
                        f4 = _fs_ex(i4x, i4y, i4z, kT, zloc, fcap, f_ex, vcut) if bose else np.exp(-e4 / (M * vth * vth))
                        c4 = 1
                    else:
                        c4, first4 = _count(vk, 0, n, _vkey(i4x, i4y, i4z, dv))
                        f4 = c4 * fnorm
                    p_in = nc * SIGMA * Av * f4 * dt / (np.pi * vr)
                X = np.random.random()
                if X < p_out:
                    v[i, 0] = o3x; v[i, 1] = o3y; v[i, 2] = o3z
                    if n_new < max_new:
                        new_r[n_new, 0] = r[i, 0]; new_r[n_new, 1] = r[i, 1]; new_r[n_new, 2] = r[i, 2]
                        new_v[n_new, 0] = o4x; new_v[n_new, 1] = o4y; new_v[n_new, 2] = o4z
                        n_new += 1
                    n_out += 1
                elif X < p_out + p_in and c4 > 0:
                    # remove a partner: the particle of the cell closest to v4 within vtol
                    # (smooth) or one from the velocity cell of v4 (6D counting)
                    pick = -1
                    if smooth:
                        bd = vtol * vtol
                        for q in range(n):
                            j = ids[q]
                            if j == i or not alive[j]:
                                continue
                            d2 = (v[j, 0] - i4x) ** 2 + (v[j, 1] - i4y) ** 2 + (v[j, 2] - i4z) ** 2
                            if d2 < bd:
                                bd = d2; pick = j
                    else:
                        for q in range(c4):
                            j = ids[first4 + q]
                            if j != i and alive[j]:
                                pick = j; break
                    if pick >= 0:
                        # energy conservation with the actual partner: K3 = K2 + K_partner + g n_c,
                        # keeping the nominal direction of v3 (momentum is taken up by the condensate)
                        Kp = 0.5 * M * (v[pick, 0] ** 2 + v[pick, 1] ** 2 + v[pick, 2] ** 2)
                        K3 = 0.5 * M * v2 + Kp + G_INT * nc
                        n3 = np.sqrt(i3x * i3x + i3y * i3y + i3z * i3z)
                        if n3 > 1e-12 and K3 > 0.0:
                            sc = np.sqrt(2.0 * K3 / M) / n3
                            alive[pick] = False
                            v[i, 0] = i3x * sc; v[i, 1] = i3y * sc; v[i, 2] = i3z * sc
                            n_in += 1
    return n22, n_out, n_in, n_new
