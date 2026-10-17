"""Simulation of the PRL protocols with the kinetic model and extraction of
the observables in the form the experiment reports them."""
import numpy as np
from scipy.special import zeta

from . import constants as C
from .kinetic import KineticModel, Protocol, truncated_bose
from .evaporation import make_omega, make_trap_depth, make_gamma_K, integrated_gamma_K, A_GAMMA_K, T1_MEAS
from .observables import gaussian_fit_temperature, effective_temperature, mu_thermal, F_eq_eq1

DEFAULT = dict(eta1=8.0, p=2.0, s_omega=1.0, gamma_bg=0.0, L3_scale=1.0, c12_scale=1.0, c22_scale=1.0,
               N1=55993.0, T1=T1_MEAS, Nc_seed=10.0, z1=1.0)


def initial_state(M, T_meas, mu_over_kT, N_th, t, Nc0, match_TE=True):
    """Truncated Bose cloud whose effective temperature (mean energy per atom,
    see observables.effective_temperature) equals T_meas.  The truncation at
    the trap depth removes energy, so the Bose temperature of the distribution
    is slightly higher than T_meas."""
    if not match_TE:
        return truncated_bose(M, T_meas, mu_over_kT, N_target=N_th, t=t, Nc0=Nc0)
    lo, hi = 0.8 * T_meas, 2.5 * T_meas
    for _ in range(40):
        T = 0.5 * (lo + hi)
        y0, mu = truncated_bose(M, T, mu_over_kT, N_target=N_th, t=t, Nc0=Nc0)
        d = M.unpack(t, y0)
        TE = effective_temperature(M.x, d["g"], d["rho"], d["mc"], d["omega"])
        if not np.isfinite(TE) or TE > T_meas:
            hi = T
        else:
            lo = T
        if hi - lo < 1e-4 * T_meas:
            break
    return y0, mu


def omega_cal(t):
    """Paper's calibrated geometric-mean frequency (rad/s) used for tau, mu."""
    return 2 * np.pi * (192.0 - 56.0 * t)


def observables(model, t, y, omega_for_analysis=None, gaussian_T=False):
    d = model.unpack(t, y)
    om_an = omega_for_analysis(t) if omega_for_analysis else d["omega"]
    TE = effective_temperature(model.x, d["g"], d["rho"], d["mc"], d["omega"])
    TG = gaussian_fit_temperature(model.x, d["g"], d["mc"], d["omega"]) if gaussian_T else np.nan
    out = dict(t=t, N=d["N"], Nc=d["Nc"], Nth=d["Nth"], F=d["F"], T=TE, T_gauss=TG,
               mu_c_nK=d["mu_c"] / C.nK, E_nK=d["E"] / C.nK, omega=d["omega"], mc=d["mc"])
    if np.isfinite(TE):
        out["mu_th_nK"] = mu_thermal(d["Nth"], TE, om_an) / C.nK
        out["tau"] = TE / C.Tc_ideal(d["N"], om_an)
        out["F_eq1"] = F_eq_eq1(TE, d["N"], om_an)
    else:
        out["mu_th_nK"] = out["tau"] = out["F_eq1"] = np.nan
    return out


def run_ramp(params=None, diss_factor=0.0, t_start=1.0, t_end=2.0, n_grid=240, dt_out=0.05,
             gaussian_T=False, rtol=1e-6, atol=1e-2, x_margin=1.15, hold_from=None, hold_until=None,
             diss_off_at=None, open_top=True):
    """Last second of the evaporation ramp (Fig. 2 / S2(i)).  The initial state at
    t_start is a truncated Bose cloud with the measured N and T of the
    no-dissipation data; with dissipation the number is reduced by
    exp(-a t_start^2/2) (uniform loss leaves the distribution shape unchanged).
    If hold_from is given, the trap is frozen from that time and the run
    continues to hold_until (Fig. 4 / S2(ii)); dissipation stops at diss_off_at."""
    P = dict(DEFAULT); P.update(params or {})
    omega = make_omega(P["s_omega"], t_freeze=hold_from)
    U = make_trap_depth(omega, P["eta1"], p=P["p"], t_freeze=hold_from)
    gk_a, gk_k = P.get("gK_a", A_GAMMA_K), P.get("gK_k", 1.0)
    gK = make_gamma_K(a=gk_a, k=gk_k, factor=diss_factor, t_off=(diss_off_at if diss_off_at is not None else np.inf)) if diss_factor > 0 else (lambda t: 0.0)
    prot = Protocol(omega=omega, trap_depth=U, gamma_K=gK, gamma_bg=P["gamma_bg"], L3=C.L3_cond * P["L3_scale"])
    M = KineticModel(prot, n_grid=n_grid, c12_scale=P["c12_scale"], c22_scale=P["c22_scale"], open_top=open_top,
                     evap_mode=P.get("evap_mode", "3d"), omega_z_ratio=P.get("omega_z_ratio", 1.0), kick_ratio=P.get("kick_ratio", 0.0))
    x_t0 = U(t_start) / (C.hbar * omega(t_start))
    M.setup_grid(t_start, (x_margin if P.get("evap_mode", "3d") == "3d" else 2.0) * x_t0)
    N1 = P["N1"] * (np.exp(-integrated_gamma_K(t_start, gk_a, diss_factor, gk_k)) if diss_factor > 0 else 1.0)
    y0, mu = initial_state(M, P["T1"], np.log(P["z1"]) if P["z1"] < 1 else 0.0, N1 - P["Nc_seed"], t_start, P["Nc_seed"], match_TE=P.get("match_TE", True))
    t_stop = hold_until if hold_until is not None else t_end
    t_eval = np.minimum(np.arange(t_start, t_stop + 1e-9, dt_out), t_stop)
    sol = M.run(y0, t_start, t_stop, t_eval=t_eval, rtol=rtol, atol=atol)
    rows = [observables(M, sol.t[k], sol.y[:, k], omega_cal, gaussian_T) for k in range(len(sol.t))]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["_model"] = M; out["_sol"] = sol; out["_params"] = P; out["_mu1"] = mu
    return out


def run_hold(N0, T0, F0, omega_hold, eta_hold, diss_factor=0.0, t_hold=4.0, params=None, n_grid=240,
             dt_out=0.1, gaussian_T=False, rtol=1e-6, atol=1e-2, t_diss_start=-2.0):
    """Hold at constant trap (Fig. 4 / S2(ii)): start from a saturated Bose cloud
    at T0 with N0 (1 - F0) thermal atoms and N0 F0 condensate atoms, trap
    frequency omega_hold (rad/s), depth eta_hold k_B T0.  With diss_factor > 0
    the dissipation gamma_K = a (t - t_diss_start) acts for t < 0 only in the
    experiment; here the state at t = 0 is given, so diss_factor is unused
    except to document the series."""
    P = dict(DEFAULT); P.update(params or {})
    omega = lambda t: omega_hold
    U = lambda t: eta_hold * C.kB * T0
    prot = Protocol(omega=omega, trap_depth=U, gamma_K=lambda t: 0.0, gamma_bg=P["gamma_bg"], L3=C.L3_cond * P["L3_scale"])
    M = KineticModel(prot, n_grid=n_grid, c12_scale=P["c12_scale"], c22_scale=P["c22_scale"],
                     evap_mode=P.get("evap_mode", "3d"), omega_z_ratio=P.get("omega_z_ratio", 1.0))
    M.setup_grid(0.0, (1.15 if P.get("evap_mode", "3d") == "3d" else 2.0) * U(0) / (C.hbar * omega_hold))
    Nc0 = N0 * F0
    y0, mu = initial_state(M, T0, 0.0, N0 - Nc0, 0.0, max(Nc0, 1.0), match_TE=P.get("match_TE", True))
    # if the saturated cloud cannot hold N0 (1 - F0) thermal atoms at T0 (the
    # measured state is oversaturated for this omega), put the excess into
    # the condensate so that the total number matches the measurement
    Nth_got = y0[:M.n].sum() * M.Delta
    if Nth_got < N0 - Nc0 - 1.0:
        Nc0 = N0 - Nth_got
        y0, mu = initial_state(M, T0, 0.0, N0 - Nc0, 0.0, Nc0, match_TE=P.get("match_TE", True))
        Nth_got = y0[:M.n].sum() * M.Delta
        y0[M.n] = N0 - Nth_got
    t_eval = np.minimum(np.arange(0.0, t_hold + 1e-9, dt_out), t_hold)
    sol = M.run(y0, 0.0, t_hold, t_eval=t_eval, rtol=rtol, atol=atol)
    rows = [observables(M, sol.t[k], sol.y[:, k], None, gaussian_T) for k in range(len(sol.t))]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["_model"] = M; out["_sol"] = sol; out["_params"] = P
    return out
