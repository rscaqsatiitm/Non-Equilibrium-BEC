"""Physical constants and 87Rb parameters (SI).

Provenance of the atomic parameters:
- m_Rb87: CODATA/atomic mass 86.909 u (textbook value).
- a (F=1, mF=-1): 100.4 a0 [Egorov et al., PRA 87, 053614 (2013)] is the
  modern value; the PRL supplement uses sigma = 8 pi a^2 with the standard
  value.  We use a = 100.4 a0 and keep it as a parameter.
- L3 (F=1, mF=-1 condensate): 5.8e-30 cm^6/s, Burt et al., PRL 79, 337 (1997),
  as quoted in Soeding et al., Appl. Phys. B 69, 257 (1999) [arXiv:cond-mat/9811339]
  (retrieved this session).  Thermal cloud: 3! = 6 times larger.
"""
import numpy as np

hbar = 1.054571817e-34
kB = 1.380649e-23
amu = 1.66053906660e-27
a0 = 5.29177210903e-11

m_Rb = 86.909180 * amu
a_s = 100.4 * a0
g_int = 4 * np.pi * hbar**2 * a_s / m_Rb
sigma_el = 8 * np.pi * a_s**2
L3_cond = 5.8e-30 * 1e-12        # m^6/s (condensate)
L3_thermal = 6 * L3_cond

nK = 1e-9 * kB                    # energy of 1 nK


def omega_bar(f_hz):
    """Angular geometric-mean trap frequency from a frequency in Hz."""
    return 2 * np.pi * f_hz


def aho(omega):
    return np.sqrt(hbar / (m_Rb * omega))


def mu_TF(Nc, omega):
    """Thomas-Fermi chemical potential (J) for Nc condensate atoms."""
    Nc = max(Nc, 0.0)
    return 0.5 * hbar * omega * (15.0 * Nc * a_s / aho(omega)) ** 0.4


def Gamma0(omega):
    """Collision-rate unit m^3 g^2 (hbar omega)^2 / (2 pi^3 hbar^7) [1/s]
    appearing in the ergodic collision integrals when energies are measured
    in units of hbar*omega (Bijlsma, Zaremba, Stoof, PRA 62, 063609, Eq. 26)."""
    return m_Rb**3 * g_int**2 * (hbar * omega) ** 2 / (2 * np.pi**3 * hbar**7)


def Tc_ideal(N, omega):
    """k_B T_c = 0.94 hbar omega N^{1/3} (paper's definition), in K."""
    return 0.94 * hbar * omega * N ** (1.0 / 3.0) / kB
