#!/bin/bash
# Final pipeline for the 1D-evaporation variant, run after the calibrations.
cd "$(dirname "$0")/.."
until grep -q BEST logs/calib_1d_v2.log && grep -q BEST logs/calib_hold_1d_v2.log; do sleep 15; done
cp data/calibration_ramp_s1.00_1d.json data/calibration_ramp_1d_final.json
python3 scripts/03_predict_dissipation.py data/calibration_ramp_1d_final.json 320 empirical > logs/final_predict_1d_empirical.log 2>&1 &
python3 scripts/03_predict_dissipation.py data/calibration_ramp_1d_final.json 320 fig1b > logs/final_predict_1d_fig1b.log 2>&1 &
python3 scripts/04b_hold_predict.py data/calibration_hold_1d.json 320 > logs/final_hold_1d.log 2>&1 &
python3 scripts/05_sensitivity.py data/calibration_ramp_1d_final.json 240 > logs/final_sensitivity_1d.log 2>&1 &
wait
python3 scripts/06_summary.py data/results_ramp_1d_empirical.json data/results_hold_1d.json data/sensitivity_1d.json > logs/final_summary_1d.log 2>&1
cp manuscript/ledger.md manuscript/ledger_1d.md; cp manuscript/numbers.json manuscript/numbers_1d.json
echo PIPELINE_DONE
