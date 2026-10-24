/* Copyright 2021-2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Modern Electron, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
#define WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Algorithm.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>

enum class ScatteringProcessType {
    INVALID,
    ELASTIC,
    TWO_PRODUCT_REACTION,
    CHARGE_EXCHANGE,
    EXCITATION,
    IONIZATION,
};

class ScatteringProcess
{
public:
    ScatteringProcess (
                const std::string& scattering_process,
                const std::string& cross_section_file,
                amrex::ParticleReal energy,
                ScatteringAngleModel scattering_angle_model
                );

    template <typename InputVector>
    ScatteringProcess (
                const std::string& scattering_process,
                const InputVector&& energies,
                const InputVector&& sigmas,
                amrex::ParticleReal energy,
                ScatteringAngleModel scattering_angle_model
                );

    ~ScatteringProcess() = default;

    ScatteringProcess (ScatteringProcess const&)            = delete;
    ScatteringProcess& operator= (ScatteringProcess const&) = delete;
    ScatteringProcess (ScatteringProcess &&)                = default;
    ScatteringProcess& operator= (ScatteringProcess &&)     = default;

    /** Read the given cross-section data file to memory.
     *
     * @param cross_section_file the path to the file containing the cross-
     *        section data
     * @param energies vector storing energy values in eV
     * @param sigmas vector storing cross-section values
     *
     */
    static
    void readCrossSectionFile (
                               const std::string& cross_section_file,
                               amrex::Vector<amrex::ParticleReal>& energies,
                               amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas
                               );

    static
    void sanityCheckEnergyGrid (
                                const amrex::Vector<amrex::ParticleReal>& energies
                                );

    struct Executor {
        /** Get the collision cross-section using a simple linear interpolator. If
         * the energy value is lower (higher) than the given energy range the
         * first (last) cross-section value is used.
         *
         * @param E_coll collision energy in eV
         *
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::ParticleReal getCrossSection (amrex::ParticleReal E_coll) const
        {
            if (E_coll < m_energy_lo) {
                return m_sigma_lo;
            } else if (E_coll > m_energy_hi) {
                return m_sigma_hi;
            } else {
                // Locate the pair of energy grid points bracketing E_coll using a
                // bisection search. This does not assume an evenly spaced energy grid.
                // `amrex::bisect` returns idx_1 such that
                // m_energies_data[idx_1] <= E_coll < m_energies_data[idx_1 + 1].
                const int idx_1 = amrex::bisect(m_energies_data, 0, m_grid_size - 1, E_coll);
                const int idx_2 = idx_1 + 1;

                // linearly interpolate to the given energy value
                const amrex::ParticleReal temp =
                    (E_coll - m_energies_data[idx_1])
                    / (m_energies_data[idx_2] - m_energies_data[idx_1]);
                return m_sigmas_data[idx_1] + (m_sigmas_data[idx_2] - m_sigmas_data[idx_1]) * temp;
            }
        }

        /** Sample the kinetic energy of the electron ejected by an electron-impact
         * ionization event, according to the energy sharing model of this process.
         * The energy of the scattered (incident) electron is then E_out_eV minus
         * the returned value.
         *
         * @param E_out_eV energy left after ionization (incident electron energy minus the
         *        ionization energy) in eV, to be divided between the two electrons
         * @param engine the random engine
         * @return the kinetic energy of the ejected electron in eV, in [0, E_out_eV/2]
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        double getEjectedElectronEnergy (double E_out_eV, amrex::RandomEngine const& engine) const
        {
            if (m_energy_sharing == IonizationEnergySharing::OPB) {
                // Invert the cumulative distribution of the Opal-Peterson-Beaty differential
                // cross section, sigma(E_ej) ~ 1 / (1 + (E_ej/B)^2), on [0, E_out_eV/2];
                // see Vahedi & Surendra, Comput. Phys. Commun. 87, 179 (1995), Eq. (20)
                const auto B = static_cast<double>(m_opb_B);
                const double R = amrex::Random(engine);
                const double E_ej_eV = B * std::tan(R * std::atan(0.5 * E_out_eV / B));
                return amrex::max(0.0, amrex::min(E_ej_eV, 0.5 * E_out_eV));
            }
            return 0.5 * E_out_eV;
        }

        /** True if electron-impact ionization uses the original kinematics model of WarpX:
         * equal energy sharing and isotropic scattering of both electrons. **/
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        bool hasEqualSharingIsotropicIonization () const
        {
            return (m_energy_sharing == IonizationEnergySharing::Equal) &&
                   (m_scattering_angle_model == ScatteringAngleModel::Isotropic);
        }

        amrex::ParticleReal* m_energies_data = nullptr;
        amrex::ParticleReal* m_sigmas_data = nullptr;
        int m_grid_size = 0;
        amrex::ParticleReal m_energy_lo, m_energy_hi, m_sigma_lo, m_sigma_hi, m_dE;
        amrex::ParticleReal m_energy_penalty;
        ScatteringProcessType m_type;
        ScatteringAngleModel m_scattering_angle_model;
        /** True if this process creates new product particles (ionization, charge exchange,
         *  two-product reaction); false for purely scattering processes (elastic, excitation). **/
        bool m_produces_products = false;
        /** For ionization: how the remaining energy is divided between the two electrons **/
        IonizationEnergySharing m_energy_sharing = IonizationEnergySharing::Equal;
        /** For ionization with Opal-Peterson-Beaty energy sharing: the B parameter in eV **/
        amrex::ParticleReal m_opb_B = 0;
    };

    [[nodiscard]]
    Executor const& executor () const {
#ifdef AMREX_USE_GPU
        return m_exe_d;
#else
        return m_exe_h;
#endif
    }

    [[nodiscard]] amrex::ParticleReal getCrossSection (amrex::ParticleReal E_coll) const
    {
        return m_exe_h.getCrossSection(E_coll);
    }

    [[nodiscard]] amrex::ParticleReal getEnergyPenalty () const { return m_exe_h.m_energy_penalty; }
    [[nodiscard]] amrex::ParticleReal getMinEnergyInput () const { return m_exe_h.m_energy_lo; }
    [[nodiscard]] amrex::ParticleReal getMaxEnergyInput () const { return m_exe_h.m_energy_hi; }
    [[nodiscard]] amrex::ParticleReal getEnergyInputStep () const { return m_exe_h.m_dE; }

    [[nodiscard]] ScatteringProcessType type () const { return m_exe_h.m_type; }
    [[nodiscard]] ScatteringAngleModel getScatteringAngleModel () const { return m_exe_h.m_scattering_angle_model; }

    /** Set how the energy left after an electron-impact ionization event is divided
     * between the scattered and ejected electrons.
     *
     * @param energy_sharing the energy sharing model
     * @param opb_B the B parameter (in eV) of the Opal-Peterson-Beaty model; only used
     *        if energy_sharing is IonizationEnergySharing::OPB
     */
    void setIonizationEnergySharing (IonizationEnergySharing energy_sharing,
                                     amrex::ParticleReal opb_B);

    static
    ScatteringProcessType parseProcessType(const std::string& process);

private:

    void init (const std::string& scattering_process, amrex::ParticleReal energy,
               ScatteringAngleModel scattering_angle_model);

    amrex::Vector<amrex::ParticleReal> m_energies;

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_energies_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_sigmas_d;
    Executor m_exe_d;
#endif
    amrex::Gpu::HostVector<amrex::ParticleReal> m_sigmas_h;
    Executor m_exe_h;
};

#endif // WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
