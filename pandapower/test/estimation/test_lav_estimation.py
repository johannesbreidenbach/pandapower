# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


# import os
# from copy import deepcopy, copy

import numpy as np
import pandas as pd
import copy
# import pytest

# from pandapower import pp_dir
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line_from_parameters, \
    create_measurement  # , create_load, create_transformer, create_line, create_sgen, create_transformer3w, create_switch
from pandapower.estimation import estimate  #, chi2_analysis, remove_bad_data
from pandapower.auxiliary import pandapowerNet
# from pandapower.file_io import from_json
from pandapower.networks.cigre_networks import create_cigre_network_mv
# from pandapower.networks.power_system_test_cases import case9
from pandapower.run import runpp
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


def _r(v: float = 0.03) -> float:
    """
        Return a random scaling factor from a normal distribution.

        The value is drawn from a normal distribution with mean ``1.0`` and standard deviation ``v``. This can be used
        to apply a multiplicative perturbation to quantities such as powers or loads.

        Parameters:
            v: Standard deviation of the normal distribution
                :math:`\\mathcal{N}(1.0, v)`. Defaults to ``0.03``.

        Returns:
            Random scaling factor sampled from
            :math:`\\mathcal{N}(1.0, v)`.
        """
    return float(np.random.normal(1.0, v))


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


def test_cigre_network(init='flat'):
    """
       Run power flow and three different state estimations (WLS, LAV, WLAV) on a CIGRE MV network and return bus
       voltages, angles, and deviations.

       The network is solved once with a power flow, measurements are created, and then the network is copied for each
       estimation algorithm. Results from power flow and estimation, as well as their differences, are collected in a
       pandas DataFrame.

       Parameters:
           init: Initialization method for the state estimation (e.g. ``"flat"``).

       Returns:
           DataFrame with columns:
           ``[
               'V PF', 'angle PF',
               'V WLS', 'angle WLS', 'dV WLS', 'dAngle WLS',
               'V LAV', 'angle LAV', 'dV LAV', 'dAngle LAV',
               'V WLAV', 'angle WLAV', 'dV WLAV', 'dAngle WLAV'
           ]``.
       """
    # 1. create network
    # test the mv ring network with all available voltage measurements and bus powers
    # test if switches and transformer will work correctly with the state estimation
    np.random.seed(123456)
    net_base = create_cigre_network_mv(with_der=False)
    runpp(net_base)

    # 2. create measurements ONCE on the base network
    for bus, row in net_base.res_bus.iterrows():
        create_measurement(
            net_base,
            meas_type="v",
            element_type="bus",
            value=row.vm_pu * _r(.0),  # .01
            std_dev=0.01,
            element=bus
        )
        create_measurement(
            net=net_base,
            meas_type="p",
            element_type="bus",
            value=row.p_mw * _r(.0),  # default
            std_dev=max(0.001, abs(0.03 * row.p_mw)),
            element=bus
        )
        create_measurement(
            net=net_base,
            meas_type="q",
            element_type="bus",
            value=row.q_mvar * _r(.0),  # default
            std_dev=max(0.001, abs(0.03 * row.q_mvar)),
            element=bus
        )

    # 3. create copies for each estimation algorithm
    net_wls = copy.deepcopy(net_base)
    net_lav = copy.deepcopy(net_base)
    net_wlav = copy.deepcopy(net_base)


    # 4. run estimations with soft-fail (collect errors, do not raise immediately)
    failures = []
    # WLS
    if not estimate(net_wls, init="flat", wlav=False):
        failures.append("WLS estimation failed")
        v_wls = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_wls = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_wls = net_wls.res_bus_est.vm_pu.values
        delta_wls = net_wls.res_bus_est.va_degree.values

    # LAV
    if not estimate(net_lav, algorithm='lav', init="flat", wlav=False, debug_mode=True):
        failures.append("LAV estimation failed")
        v_lav = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_lav = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_lav = net_lav.res_bus_est.vm_pu.values
        delta_lav = net_lav.res_bus_est.va_degree.values

    # WLAV
    if not estimate(net_wlav, algorithm='lav', init="flat", wlav=True):
        failures.append("WLAV estimation failed")
        v_wlav = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_wlav = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_wlav = net_wlav.res_bus_est.vm_pu.values
        delta_wlav = net_wlav.res_bus_est.va_degree.values


    # 5. power flow results (runpp) aus net_base
    v_pf = net_base.res_bus.vm_pu.values
    delta_pf = net_base.res_bus.va_degree.values


    # 6. Differences Estimation - PowerFlow
    dV_wls = v_wls - v_pf
    dAngle_wls = delta_wls - delta_pf

    dV_lav = v_lav - v_pf
    dAngle_lav = delta_lav - delta_pf

    dV_wlav = v_wlav - v_pf
    dAngle_wlav = delta_wlav - delta_pf


    # 7. pack results into DataFrame
    res_total_df = pd.DataFrame(
        {
            "V PF": v_pf,
            "angle PF": delta_pf,
            "V WLS": v_wls,
            "angle WLS": delta_wls,
            "dV WLS": dV_wls,
            "dAngle WLS": dAngle_wls,
            "V LAV": v_lav,
            "angle LAV": delta_lav,
            "dV LAV": dV_lav,
            "dAngle LAV": dAngle_lav,
            "V WLAV": v_wlav,
            "angle WLAV": delta_wlav,
            "dV WLAV": dV_wlav,
            "dAngle WLAV": dAngle_wlav,
        },
        index=net_base.res_bus.index,  # Bus-Index als Index
    )
    res_diff_df = pd.DataFrame(
        {
            "dV WLS": dV_wls,
            "dAngle WLS": dAngle_wls,
            "dV LAV": dV_lav,
            "dAngle LAV": dAngle_lav,
            "dV WLAV": dV_wlav,
            "dAngle WLAV": dAngle_wlav,
        },
        index=net_base.res_bus.index,  # Bus-Index als Index
    )

    # 8. Checks: apply only to successful estimates
    if "WLS estimation failed" not in failures:
        assert np.nanmax(abs(dV_wls)) < 0.0043
        assert np.nanmax(abs(dAngle_wls)) < 0.2

    if "LAV estimation failed" not in failures:
        assert np.nanmax(abs(dV_lav)) < 0.0043
        assert np.nanmax(abs(dAngle_lav)) < 0.2

    if "WLAV estimation failed" not in failures:
        assert np.nanmax(abs(dV_wlav)) < 0.0043
        assert np.nanmax(abs(dAngle_wlav)) < 0.2

    # 9. Evaluate the overall result at the end
    if failures:
        # In this case, the test is only terminated at the end, once all networks have been calculated
        raise AssertionError(" | ".join(failures))
