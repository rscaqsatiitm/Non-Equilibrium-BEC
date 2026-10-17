"""Experimental protocol of the PRL as time-dependent model inputs.

Time origin: t = 0 when the K MOT is switched on, 2 s before the end of the
evaporation ramp (PRL, Fig. 2 caption).  The data of Fig. 2 span t = 1 to 2 s.

Trap frequency: the supplement (Fig. S1) gives the geometric-mean frequency
during the last 2 s as f(t) = 192 - 56 t Hz (calibrated from Eq. (1)) and
f(t) = 153 - 44 t Hz (measured by parametric heating, stated to underestimate
by up to ~15% because of anharmonicity).  We use f(t) = s_omega (192 - 56 t)
with s_omega a calibration parameter in [0.8, 1.0].

Trap depth: crossed optical dipole trap, U ∝ P and omega ∝ sqrt(P), so
U(t) = U_1 (omega(t)/omega_1)^p with p = 2 for a pure dipole trap; gravity and
the finite beam waists make the effective depth fall faster, so p is also a
parameter.  U_1 is parametrised by eta_1 = U_1/(k_B T_1) with T_1 = 240 nK the
measured temperature at t = 1 s.

Dissipation: gamma_K(t) = a t for t >= 0 (Fig. 1(b): a = 0.112 Hz/s from the
digitized blue diamonds; 4x for Fig. S2(i)).  Hold phase (Fig. 4 / S2(ii)):
gamma_K switched off at t_off, trap depth and frequency frozen.
"""
import numpy as np
from . import constants as C

F1_CAL, F1_SLOPE = 192.0, 56.0            # Hz, Hz/s (supplement Fig. S1, calibrated)
A_GAMMA_K = 0.1122                        # Hz/s, slope of the digitized Fig. 1(b) diamonds
T1_MEAS = 240e-9                          # K, Fig. 2(b) at t = 1 s (digitized 239.2 nK)


def make_omega(s_omega=1.0, t_freeze=None, f0=F1_CAL, slope=F1_SLOPE):
    def omega(t):
        tt = t if t_freeze is None else min(t, t_freeze)
        return 2 * np.pi * s_omega * (f0 - slope * tt)
    return omega


def make_trap_depth(omega, eta1, p=2.0, t_ref=1.0, T_ref=T1_MEAS, t_freeze=None):
    U1 = eta1 * C.kB * T_ref
    w1 = omega(t_ref)
    def U(t):
        tt = t if t_freeze is None else min(t, t_freeze)
        return U1 * (omega(tt) / w1) ** p
    return U


def make_gamma_K(a=A_GAMMA_K, factor=1.0, t_on=0.0, t_off=np.inf, k=1.0):
    """gamma_K(t) = factor * a * (t - t_on)^k.  k = 1 with a from Fig. 1(b) is
    the measured K-induced rate; k = 2 with a = c fitted on N_red/N_blue is the
    empirical loss actually seen in Figs. 2 and S2(i) (see data/gammaK_fit.json)."""
    def gK(t):
        if t < t_on or t > t_off:
            return 0.0
        return factor * a * (t - t_on) ** k
    return gK


def integrated_gamma_K(t, a=A_GAMMA_K, factor=1.0, k=1.0):
    return factor * a * t ** (k + 1) / (k + 1)
