"""Long-statistics detailed-balance test of the C12 kernel on a homogeneous Bose gas."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.phasespace.collisions_nb import collision_step, bose_params
rng = np.random.default_rng(0)
T = 100e-9; kT = C.kB * T; cell = 1.5e-6; V = cell**3
nc = 5e19; mu_c = C.g_int * nc
for dmu in (0.0, -0.3, -0.6, 0.0):
    Np = 300000; vth = np.sqrt(kT / C.m_Rb)
    v = rng.normal(size=(Np, 3)) * vth * 1.5
    K = 0.5 * C.m_Rb * np.sum(v * v, axis=1)
    target = 1 / np.expm1((K + mu_c - dmu * kT) / kT); prop = np.exp(-K / (kT * 2.25))
    w = target / prop; keep = rng.random(Np) < w / w.max(); v = v[keep]; n = v.shape[0]
    z = np.exp(-(mu_c - dmu * kT) / kT); lam3 = (2 * np.pi * C.hbar**2 / (C.m_Rb * kT)) ** 1.5
    g32 = sum(z**k / k**1.5 for k in range(1, 200)); nth = g32 / lam3; alpha = nth * V / n
    r = np.zeros((n, 3)); order = np.arange(n, dtype=np.int64); starts = np.array([0], dtype=np.int64); ends = np.array([n], dtype=np.int64)
    nc_at = np.full(n, nc); new_r = np.zeros((200000, 3)); new_v = np.zeros((200000, 3))
    tot_out = 0; tot_in = 0
    for it in range(600):
        vv = v.copy(); al = np.ones(n, dtype=np.bool_)
        n22, n_out, n_in, n_new = collision_step(r, vv, al, order, starts, ends, alpha, cell, 0.5 * vth, 2e-7, nc_at, True, 60.0, 3.5, it + 1000 * int(abs(dmu) * 10), new_r, new_v, 200000, False, True, True, 1.0 * vth)
        tot_out += n_out; tot_in += n_in
    print(f"dmu/kT={dmu:+.1f}: z={z:.3f}: out={tot_out} in={tot_in} out/in={tot_out/max(tot_in,1):.3f} +- {np.sqrt(1/max(tot_out,1)+1/max(tot_in,1))*tot_out/max(tot_in,1):.3f}  expected {np.exp(-dmu):.3f}", flush=True)
