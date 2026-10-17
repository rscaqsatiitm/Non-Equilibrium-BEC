"""Grid convergence of the ramp simulation (no dissipation, calibrated parameters)."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nebec.experiment import run_ramp
cal = json.load(open(sys.argv[1]))["best"]
P = dict(eta1=cal["eta1"], p=cal["p"], gamma_bg=cal["gamma_bg"], s_omega=cal["s_omega"], evap_mode=cal.get("evap_mode", "3d"))
out = {}
for n in (160, 240, 320, 480, 640):
    t0 = time.time()
    r = run_ramp(P, diss_factor=1.0, n_grid=n, dt_out=0.25, rtol=1e-6, atol=1e-2)
    out[n] = dict(t=r["t"].tolist(), N=r["N"].tolist(), F=r["F"].tolist(), T=(r["T"] * 1e9).tolist())
    print(f"n={n:4d} [{time.time()-t0:5.0f} s]: " + "  ".join(f"t={t:.2f}: N={N:.0f} F={F:.3f} T={T:.1f}" for t, N, F, T in zip(r["t"], r["N"], r["F"], r["T"] * 1e9)), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data", f"grid_convergence_{P['evap_mode']}.json"), "w"))
