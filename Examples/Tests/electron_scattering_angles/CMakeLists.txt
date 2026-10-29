# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_electron_scattering_angles_dsmc  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_electron_scattering_angles_dsmc  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1 --skip-fields"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_electron_scattering_angles_mcc  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_electron_scattering_angles_mcc  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1 --skip-fields"  # checksum
    OFF  # dependency
)
