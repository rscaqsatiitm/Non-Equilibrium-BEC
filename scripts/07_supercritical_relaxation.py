"""Relaxation of the measured 'supercritical' state in the kinetic model.
Initial state: the red (dissipation) point of Fig. 2 at t = 2 s, N = 12570,
T = 75.7 nK, F = 0.461 at the end-of-ramp trap (f = 80 Hz with the paper's
calibrated frequency; trap depth eta k_B T with eta from the calibration), i.e.
a thermal cloud of 6780 atoms that is undersaturated (mu_th < mu_c) coexisting
with a condensate of 5790 atoms.  The model is then run with all losses off to
see how fast the condensate is depleted into the thermal cloud.
Usage: python scripts/07_supercritical_relaxation.py [eta] [n_grid] [N T_nK F f_Hz]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec import constants as C
from nebec.kinetic import KineticModel, Protocol, truncated_bose
from nebec.observables import effective_temperature, mu_thermal

eta = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 320
# default: Fig. 2 red point at t = 2 s; optional args: N T(nK) F f(Hz)
N, T, F, f_Hz = 12570.0, 75.66e-9, 0.4607, 80.0
if len(sys.argv) > 6:
    N, T, F, f_Hz = float(sys.argv[3]), float(sys.argv[4]) * 1e-9, float(sys.argv[5]), float(sys.argv[6])
om = 2 * np.pi * f_Hz
U = eta * C.kB * T
P = Protocol(omega=lambda t: om, trap_depth=lambda t: U, gamma_K=lambda t: 0.0, gamma_bg=0.0, L3=0.0)
M = KineticModel(P, n_grid=n_grid, open_top=True)
M.setup_grid(0.0, 1.15 * U / (C.hbar * om))
Nc0 = F * N
y0, mu = truncated_bose(M, T, -0.3, N_target=N - Nc0, Nc0=Nc0)
d = M.unpack(0, y0)
print(f"initial: Nth={d['Nth']:.0f} Nc={d['Nc']:.0f} T={T*1e9:.1f} nK  mu_th-mu_c = {mu*C.kB*T/C.nK:.1f} nK (fugacity ln z={mu:.3f}), mu_c={d['mu_c']/C.nK:.1f} nK, saturated N_th at this T: {d['Nth']/ (1+0):.0f}")
Nsat = 1.202 * (C.kB * T / (C.hbar * om)) ** 3
print(f"ideal-gas saturation number at T: {Nsat:.0f}  -> undersaturation {1 - d['Nth']/Nsat:.2f}")
t_eval = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
sol = M.run(y0, 0.0, t_eval[-1], t_eval=t_eval, rtol=1e-6, atol=1e-2)
F0 = None
for k in range(len(sol.t)):
    d = M.unpack(sol.t[k], sol.y[:, k])
    TE = effective_temperature(M.x, d["g"], d["rho"], d["mc"], om)
    mu_th = mu_thermal(d["Nth"], TE, om) / C.nK if np.isfinite(TE) else np.nan
    print(f"t={sol.t[k]:.3f} s: N={d['N']:.0f} Nc={d['Nc']:.0f} F={d['F']:.3f} T_E={TE*1e9:.1f} nK mu_th={mu_th:.1f} nK mu_c={d['mu_c']/C.nK:.1f} nK")
# relaxation time of F toward its final value
Fs = np.array([M.unpack(sol.t[k], sol.y[:, k])["F"] for k in range(len(sol.t))])
dF = Fs - Fs[-1]
k = np.where(np.abs(dF) < np.abs(dF[0]) / np.e)[0]
print("1/e time of F relaxation: %.3f s" % (sol.t[k[0]] if len(k) else np.nan))
