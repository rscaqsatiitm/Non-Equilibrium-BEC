# Provenance ledger: routes 1 and 2 (anisotropy_section.tex, phasespace_section.tex)

All SIM entries: review before submission. Pointers are to log files (logs/, not
committed) and result files (data/, committed) in the repository.

| value | location | type | pointer |
|---|---|---|---|
| 35 / 45 um waists, 40 deg crossing | phasespace methods | USER/SOURCE | supplement of Mellado Munoz et al. (geometry as read this session) [CONFIRM?] |
| mu_B B'/2, B' = 10 G/cm cancels 1/3 of gravity | phasespace methods | DERIVED | mu_B B'/2 / (m g) with B' = 10 G/cm |
| 4e4 test particles, cells 1.5 um, dt 40 us, exchange every 5 steps | phasespace methods | SIM | scripts/10_phasespace_ramp.py defaults |
| C22 rate within 20% of n sigma v; 2-3 collisions per axis relaxation | phasespace methods | SIM | logs/balance_stats.log, scripts/09_phasespace_tests.py |
| detailed balance out/in = 0.97 +- 0.04 and 1.81 +- 0.08 | phasespace methods | SIM | logs/balance_stats.log, scripts/09b_balance_stats.py |
| seed 1 -> 1e3 atoms in 0.1 s at 1.6 x saturation, T = 150 nK | phasespace methods, results | SIM | logs/onset_1.6.log, scripts/09d_onset_latency.py |
| low-energy occupation 6 x Bose value after 0.3 s with exchange off | phasespace results | SIM | logs/bose_preservation.log, scripts/09c_bose_preservation.py (g(0.5) = 263 vs 40) |
| bimodal fit F_fit = 0.17 (F = 0), 0.34 (0.27), 0.56 (0.50); T_fit 20-30% low | phasespace methods | SIM | scripts/09_phasespace_tests.py (earlier session log) |
| depth ratio U(136 Hz)/U(80 Hz) > 7 for all static (g_eff, f_x) | phasespace results | SIM | scratchpad depth tables (session), reproducible with CrossedTrap.saddle_depth |
| required eta 5 (t = 1 s) and 4.5 (t = 2 s), ratio 4.1 | phasespace results | SIM | data/calibration_ramp_1d_final.json (eta_1 = 3.9, p = 3.1) and measured T |
| f_x > 0.1 g: trap cannot reach f_bar < 105 Hz | phasespace results | SIM | logs/ps_g1.0_fx0.2_K0.log, ps_g0.85_fx0.2_K0.log, ps_g1.1_fx0.2_K0.log (f_bar stalls at 103-110 Hz) |
| depths 2295 / 1071 / 183 nK at 136 / 108 / 80 Hz for g_eff = 0.7 g, f_x = 0.1 g | phasespace results | SIM | logs/ps_g0.7_fx0.1_K0.log |
| onset of condensation at t = 1.85 s (model) vs 1.15 s (data) | phasespace results | SIM / USER | logs/ps_g0.7_fx0.1_K0.log; data/digitized/fig2_nodiss.csv |
| t = 2 s: N = 2.0e4, T = 46 nK, F = 0.15 (no K); 1.35e4, F = 0.01 (K); 6.4e3, F = 0 (4K) | phasespace results | SIM | scripts/13_phasespace_compare.py on data/ps_g0.7_fx0.1_K{0,1,4}.json |
| measured t = 2 s: 2.0e4, 65 nK, 0.64; 1.26e4, 0.46; 6.0e3, 0.23 | phasespace results | USER (digitized) | data/digitized/fig2_nodiss.csv, fig2_diss.csv, figS2i_diss.csv |
| growth 1.0e4 atoms/s at T = 120 nK | phasespace results | SIM | logs/onset_1.6.log (Nc 2220 -> 4189 between 0.2 and 0.38 s) |
| T_y/T_z <= 1.2 at t = 2 s; collision rate 5-15 Hz | phasespace results | SIM | data/ps_g0.7_fx0.1_K0.json (Tmy/Tmz), c22 counters in logs |
