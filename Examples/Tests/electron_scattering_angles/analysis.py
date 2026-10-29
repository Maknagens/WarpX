#!/usr/bin/env python3
"""
Check the energy-dependent electron scattering angle models for elastic and excitation
collisions, with `elastic_scattering_angle_model = okhrimovskyy` and
`excitation1_scattering_angle_model = vahedisurendra`.

A mono-energetic electron beam (along +z) crosses a gas of argon atoms at rest
during a single time step, so that each electron undergoes at most one collision.
After the step, the electrons that did not collide still have their initial momentum,
the electrons that collided elastically have nearly their initial energy, and the
electrons that excited an argon atom have lost the excitation energy. We check that:
- the elastically scattered electrons keep their energy, up to the recoil of the atom
- the electrons that excited an atom lost the excitation energy
- the polar angles of the electrons (with respect to the incident direction)
  follow the Okhrimovskyy (elastic) and Vahedi-Surendra (excitation) distributions,
  at the energy of the electron after the collision
The distributions are checked with Kolmogorov-Smirnov tests.
"""

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e, m_e
from scipy.stats import kstest

# Parameters of the input files
E_inc_eV = 100.0  # energy of the incident electrons
E_exc_eV = 11.5  # excitation energy of argon
hartree_eV = 27.211386245988
mc2_eV = m_e * c**2 / e

ts = OpenPMDTimeSeries("./diags/diag1")
iteration = ts.iterations[-1]
ux, uy, uz = ts.get_particle(
    ["ux", "uy", "uz"], species="electrons", iteration=iteration
)

# The electrons that did not collide still move exactly along z
collided = (ux != 0.0) | (uy != 0.0)
ux, uy, uz = ux[collided], uy[collided], uz[collided]
u2 = ux**2 + uy**2 + uz**2
energy_eV = (np.sqrt(1.0 + u2) - 1.0) * mc2_eV
cos_chi = uz / np.sqrt(u2)

elastic = energy_eV > E_inc_eV - 0.5 * E_exc_eV
excitation = ~elastic
n_elastic = np.count_nonzero(elastic)
n_excitation = np.count_nonzero(excitation)
print(f"Collisions: {n_elastic} elastic, {n_excitation} excitation")
assert n_elastic > 5000
assert n_excitation > 1000

# Energy conservation, up to the small recoil energy of the argon atom
mean_loss_elastic = E_inc_eV - np.mean(energy_eV[elastic])
mean_loss_excitation = E_inc_eV - np.mean(energy_eV[excitation])
print(f"Mean energy loss: {mean_loss_elastic} eV (elastic), {mean_loss_excitation} eV (excitation)")
assert 0.0 <= mean_loss_elastic < 0.05
assert np.isclose(mean_loss_excitation, E_exc_eV, atol=0.05)

one_minus_cos = 1.0 - cos_chi

# Elastic, Okhrimovskyy: invert cos(chi) = 1 - 2 R / (1 + a (1 - R)), with a = 8 * energy / hartree,
# to recover the random number R, which should be uniformly distributed on [0, 1]
a = 8.0 * energy_eV[elastic] / hartree_eV
R_elastic = one_minus_cos[elastic] * (1.0 + a) / (2.0 + one_minus_cos[elastic] * a)
ks_elastic = kstest(R_elastic, "uniform")
print(f"Okhrimovskyy angular distribution (elastic): KS p-value = {ks_elastic.pvalue}")
assert ks_elastic.pvalue > 1.0e-3

# Excitation, Vahedi-Surendra: invert cos(chi) = (2 + E - 2 (1 + E)^R) / E, with E in eV
E = energy_eV[excitation]
R_excitation = np.log(1.0 + 0.5 * E * one_minus_cos[excitation]) / np.log(1.0 + E)
ks_excitation = kstest(R_excitation, "uniform")
print(
    f"Vahedi-Surendra angular distribution (excitation): KS p-value = {ks_excitation.pvalue}"
)
assert ks_excitation.pvalue > 1.0e-3

# At 100 eV, both models give mostly forward scattering
assert np.mean(cos_chi[elastic]) > 0.5
assert np.mean(cos_chi[excitation]) > 0.5
