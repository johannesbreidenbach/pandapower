# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import numpy as np
import pandas as pd

# imports from pandapower
import pandapower.networks as pn
from pandapower.run import runpp
from pandapower.estimation import estimate
from pandapower.create import create_measurement
from pandapower.auxiliary import pandapowerNet

from pandapower.test.estimation.test_lav_estimation import _r


# begin functions
def _create_network_with_measurements_af() -> pandapowerNet:
    # 1. load mv oberrhein
    np.random.seed(123456)

    net_base = pn.mv_oberrhein()
    runpp(net_base)

    # 2. create measurements ONCE on the base network
    for i, (bus, row) in enumerate(net_base.res_bus.iterrows()):
        if i % 10 != 0:
            continue

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
    return net_base


def test_general_function(init="flat"):
    """
       Run power flow and three different state estimations (WLS, LAV, WLAV) on different networks and return bus
       voltages, angles, and deviations. The power grid is unobservable.

       The network is solved once with a power flow, measurements are created, and then the network is copied for each
       estimation algorithm. Results from power flow and estimation, as well as their differences, are collected in a
       pandas DataFrame.

       Parameters:
           init: Initialization method for the state estimation (e.g. ``"flat"``).

       Returns:
           DataFrame with columns:
           ``[
               "V PF", "angle PF",
               "V WLS", "angle WLS", "dV WLS", "dAngle WLS",
               "V LAV", "angle LAV", "dV LAV", "dAngle LAV",
               "V WLAV", "angle WLAV", "dV WLAV", "dAngle WLAV"
           ]``.
       """

    net_base = _create_network_with_measurements_af()

    # 3. create copies for each estimation algorithm
    net_afwls = copy.deepcopy(net_base)
    net_aflav = copy.deepcopy(net_base)
    net_afwlav = copy.deepcopy(net_base)
    # net_wlav = copy.deepcopy(net_base)


    # 4. run estimations with soft-fail (collect errors, do not raise immediately)
    failures = []

    # AF-WLS
    AFWLS = estimate(net_afwls, algorithm="af-wls", init="flat", wlav=False)
    if not AFWLS["success"]:
        failures.append("AF-WLS estimation failed")
        v_afwls = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_afwls = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_afwls = net_afwls.res_bus_est.vm_pu.values
        delta_afwls = net_afwls.res_bus_est.va_degree.values

    # LAV
    AFLAV = estimate(net_aflav, algorithm="af-lp", wlav=False, with_ortools=False, init="flat", debug_mode=False)
    if not AFLAV["success"]:
        failures.append("LAV estimation failed")
        v_aflav = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_aflav = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_aflav = net_aflav.res_bus_est.vm_pu.values
        delta_aflav = net_aflav.res_bus_est.va_degree.values

    # AF-WLAV
    AFWLAV = estimate(net_afwlav, algorithm="af-lp", wlav=True, with_ortools=False, init="flat", debug_mode=False)
    if not AFWLAV["success"]:
        failures.append("AF-WLAV estimation failed")
        v_afwlav = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_afwlav = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_afwlav = net_afwlav.res_bus_est.vm_pu.values
        delta_afwlav = net_afwlav.res_bus_est.va_degree.values

#    # WLAV
#    WLAV = estimate(net_wlav, algorithm="lp", wlav=True, with_ortools=False, init="flat", debug_mode=False)
#    if not WLAV["success"]:
#        failures.append("AF-WLAV estimation failed")
#        v_wlav = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
#        delta_wlav = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
#    else:
#        v_wlav = net_wlav.res_bus_est.vm_pu.values
#        delta_wlav = net_wlav.res_bus_est.va_degree.values

    # 5. power flow results (runpp) aus net_base
    v_pf = net_base.res_bus.vm_pu.values
    delta_pf = net_base.res_bus.va_degree.values

    # 6. Differences Estimation - PowerFlow
    dV_afwls = v_afwls - v_pf
    dAngle_afwls = delta_afwls - delta_pf

    dV_aflav = v_aflav - v_pf
    dAngle_aflav = delta_aflav - delta_pf

    dV_afwlav = v_afwlav - v_pf
    dAngle_afwlav = delta_afwlav - delta_pf

    # 7. pack results into DataFrame
    res_total_df = pd.DataFrame(
        {
            "V PF": v_pf,
            "angle PF": delta_pf,
            "V AF-WLS": v_afwls,
            "angle AF-WLS": delta_afwls,
            "V AF-LAV": v_aflav,
            "angle AF-LAV": delta_aflav,
            "V AF-WLAV": v_afwlav,
            "angle AF-WLAV": delta_afwlav,
        },
        index=net_base.res_bus.index,  # Bus-Index als Index
    )
    res_diff_df = pd.DataFrame(
        {
            "dV AF-WLS": dV_afwls,
            "dAngle AF-WLS": dAngle_afwls,
            "dV AF-LAV": dV_aflav,
            "dAngle AF-LAV": dAngle_aflav,
            "dV AF-WLAV": dV_afwlav,
            "dAngle AF-WLAV": dAngle_afwlav,
        },
        index=net_base.res_bus.index,  # Bus-Index als Index
    )

    res_max_diff_df = pd.DataFrame(
        data={
            "AF WLS V": [np.max(np.abs(dV_afwls))],
            "AF WLS A": [np.max(np.abs(dAngle_afwls))],
            "AF VLA V": [np.max(np.abs(dV_aflav))],
            "AF VLA A": [np.max(np.abs(dAngle_aflav))],
            "AF WLAV V": [np.max(np.abs(dV_afwlav))],
            "AF WLAV A": [np.max(np.abs(dAngle_afwlav))]
        }
    )

    # 8. Checks: apply only to successful estimates
    if "AF-WLS estimation failed" not in failures:
        assert np.nanmax(abs(dV_afwls)) < 0.0043
        assert np.nanmax(abs(dAngle_afwls)) < 0.2

    if "AF-LAV estimation failed" not in failures:
        assert np.nanmax(abs(dV_aflav)) < 0.0043
        assert np.nanmax(abs(dAngle_aflav)) < 0.2

    if "AF-WLAV estimation failed" not in failures:
        assert np.nanmax(abs(dV_afwlav)) < 0.0043
        assert np.nanmax(abs(dAngle_afwlav)) < 0.2

    # 9. Evaluate the overall result at the end
    if failures:
        # In this case, the test is only terminated at the end, once all networks have been calculated
        raise AssertionError(" | ".join(failures))


if __name__ == '__main__':

    test_general_function()

    net14 = pn.case14()
    runpp(net14)

    net30 = pn.case30()
    runpp(net30)

    print(f"whats up")