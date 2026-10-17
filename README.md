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
