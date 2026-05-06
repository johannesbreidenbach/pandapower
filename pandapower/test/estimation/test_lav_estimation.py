# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


# import os
# from copy import deepcopy, copy

import numpy as np
import pandas as pd
# import pytest

# from pandapower import pp_dir
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line_from_parameters, \
    create_measurement, create_load, create_transformer, create_line, create_sgen, create_transformer3w, create_switch
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate
from pandapower.auxiliary import pandapowerNet
# from pandapower.file_io import from_json
# from pandapower.networks.cigre_networks import create_cigre_network_mv
# from pandapower.networks.power_system_test_cases import case9
# from pandapower.run import runpp
# from pandapower.std_types import create_std_type


def _create_2bus_test_net() -> pandapowerNet:
    """
    Create a simple 2-bus test network with measurements for state estimation.

    The network consists of:
        - Two buses connected by a single line
        - One slack (external grid) at bus 0
        - One line between bus 0 and bus 1

    Measurements included:
        - Active power flow (P) on the line (from side)
        - Reactive power flow (Q) on the line (from side)
        - Voltage magnitude at both buses

    The measurement standard deviations are chosen such that:
        - Voltage measurements are more precise (smaller std_dev)
        - Power measurements are less precise

    This setup is suitable for testing:
        - Weighted Least Squares (WLS)
        - Least Absolute Value (LAV)
        - Weighted LAV (WLAV)

    Returns:
        pandapowerNet: Configured 2-bus network with measurements.
    """

    net = create_empty_network()
    create_bus(net, name='bus1', vn_kv=1.)
    create_bus(net, name='bus2', vn_kv=1.)
    create_ext_grid(net, 0)
    create_line_from_parameters(
        net, 0, 1, 1, r_ohm_per_km=1, x_ohm_per_km=0.5, c_nf_per_km=0, max_i_ka=1
    )

    # create_measurement(net, 'p', 'line', 0.0111, 0.05, 0, 'from')
    # create_measurement(net, 'q', 'line', 0.06, 0.05, 0, 'from')
    # create_measurement(net, 'v', 'bus', 1.019, 0.01, 0)
    # create_measurement(net, 'v', 'bus', 1.04, 0.01, 1)

    create_measurement(net, 'p', 'line', 0.0111, 0.005, 0, 'from')
    create_measurement(net, 'q', 'line', 0.06, 0.5, 0, 'from')
    create_measurement(net, 'v', 'bus', 1.019, 0.01, 0)
    create_measurement(net, 'v', 'bus', 1.04, 0.1, 1)

    return net


def test_2bus_lav_wlav() -> None:
    """
    Test LAV and WLAV state estimation on a simple 2-bus system.

    This test creates a small 2-bus network with active/reactive power
    and voltage measurements. It runs both unweighted LAV and weighted
    LAV (WLAV) state estimation and compares the results.

    The test verifies:
        - Both estimations converge successfully
        - Results are finite
        - LAV produces expected reference results
        - WLAV produces a potentially different solution

    Note:
        LAV and WLAV generally produce different solutions because:
            - LAV minimizes sum of absolute residuals
            - WLAV applies measurement weights (inverse std deviation)

    Raises:
        AssertionError: If estimation fails or results are invalid.
    """

    net_lav = _create_2bus_test_net()
    net_wlav = _create_2bus_test_net()
    net_wls = _create_2bus_test_net()

    # Run estimations
    if not estimate(net_lav, algorithm='lav', init='flat', wlav=False):
        raise AssertionError('LAV estimation failed!')

    if not estimate(net_wlav, algorithm='lav', init='flat', wlav=True):
        raise AssertionError('WLAV estimation failed!')

    if not estimate(net_wls, init='flat'):
        raise AssertionError('Estimation failed!')

    # Extract results
    v_lav = net_lav.res_bus_est.vm_pu.values
    delta_lav = net_lav.res_bus_est.va_degree.values

    v_wlav = net_wlav.res_bus_est.vm_pu.values
    delta_wlav = net_wlav.res_bus_est.va_degree.values

    v_wls = net_wls.res_bus_est.vm_pu.values
    delta_wls = net_wls.res_bus_est.va_degree.values

    res_df = pd.DataFrame({
        'V WLS': v_wls,
        'angle Wls': delta_wls,
        'V LAV': v_lav,
        'angle LAV': delta_lav,
        'V WLAV': v_wlav,
        'angle WLAV': delta_wlav
    })

    # Basic sanity checks
    assert np.all(np.isfinite(v_lav)), 'LAV voltage contains invalid values'
    assert np.all(np.isfinite(delta_lav)), 'LAV angle contains invalid values'
    assert np.all(np.isfinite(v_wlav)), 'WLAV voltage contains invalid values'
    assert np.all(np.isfinite(delta_wlav)), 'WLAV angle contains invalid values'

    # Known LAV reference (empirically determined)
    target_v_lav = np.array([1.019, 1.04])
    target_delta_lav = np.array([0.0, 4.5479048])
