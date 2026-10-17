# Non-Equilibrium-BEC

Ergodic quantum-kinetic model of the dissipative distillation experiment of
Mellado Muñoz et al., Phys. Rev. Lett. 125, 020403 (2020): a trapped 87Rb Bose
gas evaporated inside a K MOT (uniform, species-selective loss).

The model drops the two approximations of the quantum kinetic theory used in
the PRL (constant temperature, infinite thermal reservoir): the full energy
distribution of the thermal cloud evolves under Bose-enhanced C22 and C12
collision integrals (Bijlsma, Zaremba, Stoof, PRA 62, 063609 (2000)) with a
Thomas-Fermi condensate and the Hartree-Fock density of states, evaporation as
a time-dependent truncation of the trap, uniform K-induced loss and three-body
loss.  Number and energy are conserved explicitly; the temperature is an output.

Layout: `nebec/` (model), `scripts/` (digitization, benchmarks, calibration,
predictions), `data/digitized/` (data recovered from the PDF figures, see
PROVENANCE.md), `figures/`, `manuscript/` (LaTeX draft and verification notes).

Install: `pip install -e .`  (numpy, scipy, matplotlib, numba, pypdf)

## Reproducing the results

```
python scripts/00_digitize.py <main.pdf> <supp.pdf>      # data/digitized
python scripts/00b_digitize_fig1b.py <fig1.png> data/digitized/fig1b_gammaK.csv
python scripts/01_benchmarks.py 200                        # conservation, equilibrium
python scripts/01b_growth_rate_benchmark.py                # vs Gardiner et al. 1998
python scripts/02_calibrate_evaporation.py 1.0 200 60 1d   # -> data/calibration_ramp_s1.00_1d.json
python scripts/04_hold_calibration.py 200 60 1d            # -> data/calibration_hold_1d.json
python scripts/03_predict_dissipation.py data/calibration_ramp_1d_final.json 320 empirical
python scripts/04b_hold_predict.py data/calibration_hold_1d.json 320
python scripts/05_sensitivity.py data/calibration_ramp_1d_final.json 240
python scripts/07_supercritical_relaxation.py 6.0 320
python scripts/06_summary.py data/results_ramp_1d_empirical.json data/results_hold_1d.json data/sensitivity_1d.json
```
The manuscript draft is `manuscript/theory_section.tex` (inputs: intro_section,
model_section, results_sections, figures_section, appendix_uniform_loss,
numbers.tex); `manuscript/verification_notes.md` and `manuscript/ledger_1d.md`
hold the confidence labels and the provenance ledger.
