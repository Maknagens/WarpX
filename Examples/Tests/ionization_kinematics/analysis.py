#!/usr/bin/env python3
"""
Check the kinematics of electron-impact ionization, with
`ionization_energy_sharing = opb` and `ionization_scattering_angle_model = okhrimovskyy`.

A mono-energetic electron beam (along +z) crosses a gas of argon atoms at rest
during a single time step, so that each electron undergoes at most one ionization event.
After the step, the electrons that did not collide still have their initial momentum,
while each ionization event produced two electrons (the scattered incident electron and
the ejected electron) that share the energy E_out = E_inc - E_ion. We check that:
- there are as many scattered electrons, ejected electrons and ions
- the two electrons of each event share the energy E_out (on average)
- the energies of the ejected electrons follow the Opal-Peterson-Beaty distribution
- the polar angles of the electrons (with respect to the incident direction)
  follow the Okhrimovskyy distribution, at the energy of each electron
The distributions are checked with Kolmogorov-Smirnov tests.
"""

import argparse

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, e, m_e
from scipy.stats import kstest

parser = argparse.ArgumentParser()
parser.add_argument(
    "--relativistic",
    action="store_true",
    help="compute the electron energies relativistically (as in background MCC)"
    " instead of non-relativistically (as in DSMC)",
)
args = parser.parse_args()

# Parameters of the input files
E_inc_lab_eV = 100.0  # energy of the incident electrons
E_ion_eV = 15.7596112  # ionization energy of argon
B_eV = 10.0  # parameter of the Opal-Peterson-Beaty model
hartree_eV = 27.211386245988
mc2_eV = m_e * c**2 / e


def kinetic_energy_eV(u2):
    """Kinetic energy of electrons with normalized momentum squared u2 = (gamma*beta)^2,
    computed the same way as in the WarpX kernel of the tested collision type"""
    if args.relativistic:
        return (np.sqrt(1.0 + u2) - 1.0) * mc2_eV
    return 0.5 * u2 * mc2_eV


ts = OpenPMDTimeSeries("./diags/diag1")
iteration = ts.iterations[-1]
ux, uy, uz = ts.get_particle(
    ["ux", "uy", "uz"], species="electrons", iteration=iteration
)
(w_ions,) = ts.get_particle(["w"], species="ar_ions", iteration=iteration)

# The electrons that did not collide still move exactly along z;
# the electrons produced by ionization events are deflected
u_beam = np.sqrt((1.0 + E_inc_lab_eV / mc2_eV) ** 2 - 1.0)
E_inc_eV = kinetic_energy_eV(u_beam**2)
E_out_eV = E_inc_eV - E_ion_eV
collided = (ux != 0.0) | (uy != 0.0)
ux, uy, uz = ux[collided], uy[collided], uz[collided]
u2 = ux**2 + uy**2 + uz**2
energy_eV = kinetic_energy_eV(u2)
cos_chi = uz / np.sqrt(u2)

# By construction, the ejected electron is the one with the lower energy
ejected = energy_eV < 0.5 * E_out_eV
n_ejected = np.count_nonzero(ejected)
n_scattered = np.count_nonzero(~ejected)
print(
    f"Ionization events: {w_ions.size} ions, {n_scattered} scattered and {n_ejected} ejected electrons"
)
assert n_ejected > 1000
assert n_ejected == n_scattered
assert n_ejected == w_ions.size

# Energy conservation: E_sc + E_ej = E_out for each event
# (up to the small recoil of the ion and the center-of-mass motion in DSMC)
mean_energy_per_event = energy_eV.sum() / n_ejected
print(
    f"Mean electron energy per event: {mean_energy_per_event} eV (expected {E_out_eV} eV)"
)
assert np.isclose(mean_energy_per_event, E_out_eV, rtol=1.0e-3)

# Energy of the ejected electrons: the Opal-Peterson-Beaty cumulative distribution
# function on [0, E_out/2] maps them to a uniform distribution on [0, 1]
cdf_opb = np.arctan(energy_eV[ejected] / B_eV) / np.arctan(0.5 * E_out_eV / B_eV)
ks_energy = kstest(cdf_opb, "uniform")
print(f"OPB energy distribution of ejected electrons: KS p-value = {ks_energy.pvalue}")
assert ks_energy.pvalue > 1.0e-3
# With B = 10 eV and E_out = 84 eV, the ejected electrons get much less than half the energy
assert np.mean(energy_eV[ejected]) < 0.25 * E_out_eV

# Scattering angles: invert cos(chi) = 1 - 2 R / (1 + a (1 - R)), with a = 8 * energy / hartree,
# to recover the random number R, which should be uniformly distributed on [0, 1]
a = 8.0 * energy_eV / hartree_eV
one_minus_cos = 1.0 - cos_chi
R = one_minus_cos * (1.0 + a) / (2.0 + one_minus_cos * a)
for name, selection in [("scattered", ~ejected), ("ejected", ejected)]:
    ks_angle = kstest(R[selection], "uniform")
    print(
        f"Okhrimovskyy angular distribution of {name} electrons: KS p-value = {ks_angle.pvalue}"
    )
    assert ks_angle.pvalue > 1.0e-3
# The scattered electrons, which keep most of the energy, are mostly forward-scattered
assert np.mean(cos_chi[~ejected]) > 0.5
