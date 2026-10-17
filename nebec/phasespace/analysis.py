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
