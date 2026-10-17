"""Benchmarks of the kinetic model: conservation laws, detailed balance and
timing.  Run: python scripts/01_benchmarks.py [n_grid]"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.kinetic import KineticModel, Protocol, truncated_bose

n_grid = int(sys.argv[1]) if len(sys.argv) > 1 else 200
om = 2 * np.pi * 100.0
U = 1.0e-6 * C.kB
P = Protocol(omega=lambda t: om, trap_depth=lambda t: U, gamma_K=lambda t: 0.0, gamma_bg=0.0, L3=0.0)

print("=== Benchmark A: closed system (no loss channel), quenched cloud -> conservation and equilibrium")
M = KineticModel(P, n_grid=n_grid, open_top=False)
M.setup_grid(0.0, U / (C.hbar * om))
T0 = 120e-9
y0, mu = truncated_bose(M, T0, -0.5, N_target=2.0e4, Nc0=50.0)
kT = C.kB * T0 / (C.hbar * om)
y0[:M.n][M.x > 5 * kT] = 0.0
d0 = M.unpack(0, y0)
print(f"grid n={M.n} Delta={M.Delta:.3f}  N0={d0['N']:.3f}  E0/N (nK)={d0['E']/d0['N']/C.nK:.4f}")
M.rhs(0, y0)
t0 = time.time(); f = M.rhs(0, y0); print("rhs wall time (s):", time.time() - t0)
print("number balance of rhs (atoms/s):", f[:M.n].sum() * M.Delta + f[M.n])
sol = M.run(y0, 0.0, 6.0, t_eval=np.linspace(0, 6, 7), rtol=1e-7, atol=1e-3)
print("solver status", sol.status, "nfev", sol.nfev)
for k in range(len(sol.t)):
    d = M.unpack(sol.t[k], sol.y[:, k])
    print(f"t={d['t']:.2f} N={d['N']:.4f} Nc={d['Nc']:.1f} F={d['F']:.4f} E/N(nK)={d['E']/d['N']/C.nK:.4f} mc={d['mc']:.2f}")
d = M.unpack(sol.t[-1], sol.y[:, -1])
g = d["g"]; x = M.x
sel = (x > 2) & (g > 1e-4)
beta = np.polyfit(x[sel], np.log1p(1.0 / g[sel]), 1)
Tfit = C.hbar * om / (C.kB * beta[0])
print(f"final: ln(1+1/g) slope -> T = {Tfit*1e9:.2f} nK, (mu-mu_c)/hbar w = {-beta[1]/beta[0]:.4f} (0 expected)")
resid = np.log1p(1.0 / g[sel]) - np.polyval(beta, x[sel])
print("max |residual| of Bose fit:", np.abs(resid).max())

print("=== Benchmark B: open trap (evaporation), same initial state")
M2 = KineticModel(P, n_grid=n_grid, open_top=True)
M2.setup_grid(0.0, U / (C.hbar * om))
sol = M2.run(y0, 0.0, 3.0, t_eval=np.linspace(0, 3, 4), rtol=1e-6, atol=1e-2)
for k in range(len(sol.t)):
    d = M2.unpack(sol.t[k], sol.y[:, k])
    print(f"t={d['t']:.2f} N={d['N']:.2f} Nc={d['Nc']:.1f} F={d['F']:.4f} E/N(nK)={d['E']/d['N']/C.nK:.3f}")
