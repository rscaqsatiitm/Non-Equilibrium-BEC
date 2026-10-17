"""Ergodic collision integrals on a uniform energy grid x_i = i*Delta
(energies in units of hbar*omega, measured from the thermal-band bottom mu_c).

State variables: n_i = rho_hat_i * g_i (atoms per unit x), i = 0..n-1.
Grid points with x_i above the truncation energy x_t are empty (g_i = 0) and
act as loss channels: an atom scattered there leaves the trap (evaporation).

C22 (BZS Eq. 26) on the uniform grid, with l = j + k - i:
  d(rho g)_i/dt = Gamma0 Delta^2 sum_{j,k} rho_hat(min(i,l,j,k))
        * [ (1+g_i)(1+g_l) g_j g_k - g_i g_l (1+g_j)(1+g_k) ]
Number and energy are conserved exactly on the grid (x_i + x_l = x_j + x_k).
The double sum is evaluated in O(n^2) by splitting on whether the minimum
energy belongs to the pair (j,k) or to the pair (i,l), with cumulative tables
over the pair sum s = j + k and the pair minimum p = min(j,k).

C12 (BZS Eqs. 53, 58-60): with B_jk = (1+g_{j+k}) g_j g_k - g_{j+k}(1+g_j)(1+g_k)
and Phi_jk the condensate kinematic factor,
  dN_c/dt            = + Gamma0 N_c Delta^2 sum_{j,k} Phi_jk B_jk
  d(rho g)_i/dt|_12  = Gamma0 N_c Delta [ sum_{j+k=i} Phi_jk B_jk - 2 sum_k Phi_ik B_ik ]
"""
import numpy as np
from numba import njit


@njit(cache=True)
def c22(g, rho, Delta, Gamma0, n_act, open_top):
    """Return d(rho g)_i/dt (array of length n) from thermal-thermal collisions.
    g: occupation numbers (g_i = 0 for i >= n_act, i.e. above the truncation
    energy; those states are still available as final states).
    """
    n = g.shape[0]
    ns = 2 * n - 1
    # pair tables over s = j+k and p = min(j,k):  T[s,p] = sum g_j g_k,
    # Tt[s,p] = sum (1+g_j)(1+g_k)   (ordered pairs, both (j,k) and (k,j))
    T = np.zeros((ns, n))
    Tt = np.zeros((ns, n))
    for p in range(n):
        gp = g[p]
        hp = 1.0 + gp
        # final-state pairs may have one partner beyond the grid (k >= n,
        # empty state, the atom leaves the trap): they enter Tt only.
        for k in range(p, ns - p):
            s = p + k
            if k == p:
                T[s, p] += gp * gp
                Tt[s, p] += hp * hp
            elif k < n:
                T[s, p] += 2.0 * gp * g[k]
                Tt[s, p] += 2.0 * hp * (1.0 + g[k])
            elif open_top:
                Tt[s, p] += 2.0 * hp
    # cumulative tables: P[s,m] = sum_{p>=m} T[s,p];  Q[s,m] = sum_{p<m} rho_p T[s,p]
    P = np.zeros((ns, n + 1))
    Pt = np.zeros((ns, n + 1))
    Q = np.zeros((ns, n + 1))
    Qt = np.zeros((ns, n + 1))
    for s in range(ns):
        acc = 0.0
        acct = 0.0
        for m in range(n - 1, -1, -1):
            acc += T[s, m]
            acct += Tt[s, m]
            P[s, m] = acc
            Pt[s, m] = acct
        acc = 0.0
        acct = 0.0
        for m in range(n):
            Q[s, m] = acc
            Qt[s, m] = acct
            acc += rho[m] * T[s, m]
            acct += rho[m] * Tt[s, m]
    out = np.zeros(n)
    for i in range(n):
        gi = g[i]
        hi = 1.0 + gi
        acc = 0.0
        for l in range(n):
            s = i + l
            m = i if i < l else l
            gl = g[l]
            gain = hi * (1.0 + gl) * (rho[m] * P[s, m] + Q[s, m])
            loss = gi * gl * (rho[m] * Pt[s, m] + Qt[s, m])
            acc += gain - loss
        out[i] = acc
    # final states l beyond the grid (l >= n) are empty: gain only, from
    # pairs with s = i + l >= i + n  -> handled by extending l; those pairs
    # need j,k <= n-1 so s <= 2n-2, l <= 2n-2-i.
    for i in range(n):
        if not open_top:
            break
        gi = g[i]
        hi = 1.0 + gi
        acc = 0.0
        for l in range(n, ns - i):
            s = i + l
            m = i          # l > i always here
            acc += hi * (rho[m] * P[s, m] + Q[s, m])
        out[i] += acc
    return out * (Gamma0 * Delta * Delta)


@njit(cache=True)
def c12(g, Phi, Delta, Gamma0, Nc, open_top):
    """Condensate-thermal collisions.  Returns (d(rho g)_i/dt, dN_c/dt)."""
    n = g.shape[0]
    out = np.zeros(n)
    dNc = 0.0
    for j in range(n):
        gj = g[j]
        for k in range(n):
            s = j + k
            if s >= n and not open_top:
                continue
            gs = g[s] if s < n else 0.0
            B = (1.0 + gs) * gj * g[k] - gs * (1.0 + gj) * (1.0 + g[k])
            w = Phi[j, k] * B
            dNc += w
            if s < n:
                out[s] += w
            out[j] -= 2.0 * w
    return out * (Gamma0 * Nc * Delta), dNc * (Gamma0 * Nc * Delta * Delta)
