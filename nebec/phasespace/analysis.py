"""Observables of the phase-space model in the form the experiment reports:
kinetic temperatures per axis, and the temperature of a Gaussian fit to the
column density after time of flight (2D image in the plane containing the
vertical axis and one horizontal axis)."""
import numpy as np
from scipy.optimize import curve_fit
from .. import constants as C


def tof_image(cloud, cond, t_tof=20e-3, axes=(0, 2), nbins=96, extent=None, rng=None):
    """Column density of the thermal atoms after ballistic expansion (gravity
    ignored in the image frame), plus the condensate as an expanded
    Thomas-Fermi profile (mean-field driven expansion neglected: the condensate
    is added as a narrow TF peak of radius max(R_TF, v_TF t))."""
    r = cloud.r + cloud.v * t_tof
    a, b = axes
    x, y = r[:, a], r[:, b]
    if extent is None:
        s = max(np.std(x), np.std(y)) * 4
        extent = s
    H, xe, ye = np.histogram2d(x, y, bins=nbins, range=[[x.mean() - extent, x.mean() + extent], [y.mean() - extent, y.mean() + extent]])
    H = H * cloud.alpha
    return H, xe, ye


def gaussian_fit_T(cloud, t_tof=20e-3, axes=(0, 2), nbins=96, rng=None):
    """Fit a 2D Gaussian (separate widths) to the thermal column density after
    TOF; return (T_a, T_b) from sigma_i^2 = (k_B T_i/m) t_tof^2 + sigma_i,0^2,
    neglecting the in-trap size (sigma_0 ~ 5 um against 60 um)."""
    H, xe, ye = tof_image(cloud, None, t_tof, axes, nbins)
    xc = 0.5 * (xe[1:] + xe[:-1]); yc = 0.5 * (ye[1:] + ye[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    def model(xy, A, x0, y0, sx, sy, off):
        x, y = xy
        return (A * np.exp(-(x - x0) ** 2 / (2 * sx**2) - (y - y0) ** 2 / (2 * sy**2)) + off).ravel()
    p0 = [H.max(), xc[np.argmax(H.sum(1))], yc[np.argmax(H.sum(0))], np.std(cloud.r[:, axes[0]] + cloud.v[:, axes[0]] * t_tof), np.std(cloud.r[:, axes[1]] + cloud.v[:, axes[1]] * t_tof), 0.0]
    try:
        popt, _ = curve_fit(model, (X, Y), H.ravel(), p0=p0, maxfev=4000)
    except RuntimeError:
        return np.nan, np.nan
    sx, sy = abs(popt[3]), abs(popt[4])
    return C.m_Rb * sx**2 / (C.kB * t_tof**2), C.m_Rb * sy**2 / (C.kB * t_tof**2)


def momentum_temperatures(cloud):
    return C.m_Rb * np.var(cloud.v, axis=0) / C.kB


# ---------------------------------------------------------------------------
# Experiment-like bimodal analysis of a synthetic time-of-flight image
# ---------------------------------------------------------------------------
from scipy.integrate import solve_ivp


def castin_dum_scaling(omegas, t_tof):
    """Scaling factors lambda_i(t) of a Thomas-Fermi condensate released from a
    harmonic trap (Castin and Dum 1996): d^2 lambda_i/dt^2 = w_i^2/(lambda_i prod_j lambda_j)."""
    w = np.asarray(omegas, dtype=float)
    def rhs(t, y):
        lam, dlam = y[:3], y[3:]
        return np.concatenate([dlam, w**2 / (lam * np.prod(lam))])
    sol = solve_ivp(rhs, (0, t_tof), np.array([1, 1, 1, 0, 0, 0], dtype=float), rtol=1e-8, atol=1e-10)
    return sol.y[:3, -1]


def synthetic_image(cloud, cond, t_tof=20e-3, axes=(0, 2), nbins=120, half_width=None, rng=None):
    """Column density after TOF: thermal test particles (ballistic) plus the
    condensate as a Castin-Dum expanded Thomas-Fermi profile.  Returns
    (image [atoms/pixel], x centres, y centres)."""
    a, b = axes
    r = cloud.r + cloud.v * t_tof
    x, y = r[:, a], r[:, b]
    if half_width is None:
        half_width = 4.5 * max(np.std(x), np.std(y))
    x0, y0 = np.median(x), np.median(y)
    edges_x = np.linspace(x0 - half_width, x0 + half_width, nbins + 1); edges_y = np.linspace(y0 - half_width, y0 + half_width, nbins + 1)
    H, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
    H = H * cloud.alpha
    xc = 0.5 * (edges_x[1:] + edges_x[:-1]); yc = 0.5 * (edges_y[1:] + edges_y[:-1])
    if cond is not None and cond.Nc > 0:
        lam = castin_dum_scaling(cond.omegas, t_tof)
        # TF radii in the principal frame, scaled; project on the image axes
        # assuming the principal axes coincide with the lab axes (small tilt)
        R0 = np.sqrt(2 * cond.mu / (C.m_Rb * cond.omegas**2)) * lam
        Rx, Ry = R0[a], R0[b]
        X, Y = np.meshgrid(xc - cond.r0[a], yc - cond.r0[b], indexing="ij")
        prof = np.maximum(1 - X**2 / Rx**2 - Y**2 / Ry**2, 0.0) ** 1.5
        dx = xc[1] - xc[0]; dy = yc[1] - yc[0]
        prof *= cond.Nc / max(prof.sum() * dx * dy, 1e-30) * dx * dy
        H = H + prof
    return H, xc, yc


def bimodal_fit(H, xc, yc):
    """Fit Gaussian + Thomas-Fermi (column density ~ (1 - x^2/Rx^2 - y^2/Ry^2)^{3/2})
    with all pixels weighted equally.  Returns dict with N_th, N_c, F, sigma_x,
    sigma_y, Rx, Ry and the fitted temperatures."""
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    dx = xc[1] - xc[0]; dy = yc[1] - yc[0]
    def model(xy, Ag, sx, sy, Ac, Rx, Ry, x0, y0):
        x, y = xy
        g = Ag * np.exp(-(x - x0) ** 2 / (2 * sx**2) - (y - y0) ** 2 / (2 * sy**2))
        tf = Ac * np.maximum(1 - (x - x0) ** 2 / Rx**2 - (y - y0) ** 2 / Ry**2, 0.0) ** 1.5
        return (g + tf).ravel()
    i0, j0 = np.unravel_index(np.argmax(H), H.shape)
    sx0 = np.sqrt(np.sum(H * (X - xc[i0]) ** 2) / H.sum()); sy0 = np.sqrt(np.sum(H * (Y - yc[j0]) ** 2) / H.sum())
    p0 = [H.max() * 0.5, sx0, sy0, H.max() * 0.5, sx0 * 0.5, sy0 * 0.5, xc[i0], yc[j0]]
    lb = [0, dx, dy, 0, dx, dy, xc[0], yc[0]]; ub = [np.inf, 10 * sx0, 10 * sy0, np.inf, 5 * sx0, 5 * sy0, xc[-1], yc[-1]]
    try:
        popt, _ = curve_fit(model, (X, Y), H.ravel(), p0=p0, bounds=(lb, ub), maxfev=20000)
    except Exception:
        return None
    Ag, sx, sy, Ac, Rx, Ry, x0, y0 = popt
    Nth = Ag * 2 * np.pi * sx * sy / (dx * dy)
    Nc = Ac * (2 * np.pi / 5) * Rx * Ry / (dx * dy)   # integral of (1 - rho^2)^{3/2} over the ellipse: 2 pi Rx Ry / 5
    return dict(N_th=Nth, N_c=Nc, F=Nc / max(Nth + Nc, 1e-30), sigma_x=sx, sigma_y=sy, Rx=Rx, Ry=Ry)


def experiment_observables(cloud, cond, t_tof=20e-3, axes=(0, 2), nbins=120):
    """T and F as the experiment extracts them: bimodal fit to the TOF image;
    T from the Gaussian widths with sigma^2 = k_B T t^2/m (in-trap size neglected)."""
    H, xc, yc = synthetic_image(cloud, cond, t_tof, axes, nbins)
    fit = bimodal_fit(H, xc, yc)
    if fit is None:
        return dict(T_fit_x=np.nan, T_fit_y=np.nan, F_fit=np.nan, N_th_fit=np.nan, N_c_fit=np.nan)
    Tx = C.m_Rb * fit["sigma_x"] ** 2 / (C.kB * t_tof**2); Ty = C.m_Rb * fit["sigma_y"] ** 2 / (C.kB * t_tof**2)
    return dict(T_fit_x=Tx, T_fit_y=Ty, F_fit=fit["F"], N_th_fit=fit["N_th"], N_c_fit=fit["N_c"])
