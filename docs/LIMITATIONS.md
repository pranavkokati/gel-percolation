# Limitations

1. The modulus exponent and the rigidity threshold are not separately determined. Fits give `p_r` = 0.394 to 0.459 and `f` = 1.28 to 2.57 depending on the fitting window (`results/thermal/pr_independent_estimates.csv`). The joint-collapse exponents in `collapse_joint_fit.csv` are effective values.
2. `eps` is a phenomenological transverse stiffness. It is not mapped to temperature or to a measured material, and it does not reproduce the temperature dependence of finite-temperature scaling theory.
3. The decision comparison covers one 4 x 4 rate grid at N = 513, 12 seeds per cell. It has not been repeated at larger size.
4. The Q-target of 0.2 of the initial modulus is a design choice. Sensitivity to it is in `results/sensitivity/qtarget_sensitivity.csv`; at `eps = 0` and `eps = 0.003` every disagreement, for every target tested, is the mean-field law calling a network-unsafe run safe.
5. Time calibration relies on one PEG-peptide gel study with a different chemistry. It supports an order-of-magnitude statement only (model 5.9 to 47 min against a measured gel time of 111 min).
6. The FCC validation reproduces the rigidity threshold to 2.4 % and the effective-medium exponent to 4 %, on a single lattice size (L = 8, 6 seeds).
7. No experimental data are included. The model's predictions have not been tested against a measured degrading gel.
8. Bond-removal histories are random (force-independent) in all production runs.
