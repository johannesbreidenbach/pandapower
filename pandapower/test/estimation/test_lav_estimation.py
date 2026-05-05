# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
from copy import deepcopy

import numpy as np
# import pytest

# from pandapower import pp_dir
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line_from_parameters, \
    create_measurement, create_load, create_transformer, create_line, create_sgen, create_transformer3w, create_switch
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate
# from pandapower.file_io import from_json
# from pandapower.networks.cigre_networks import create_cigre_network_mv
# from pandapower.networks.power_system_test_cases import case9
# from pandapower.run import runpp
# from pandapower.std_types import create_std_type


def test_2bus():
    # 1. Create network
    net = create_empty_network()
    create_bus(net, name="bus1", vn_kv=1.)
    create_bus(net, name="bus2", vn_kv=1.)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=1, x_ohm_per_km=0.5,
                                c_nf_per_km=0, max_i_ka=1)

    create_measurement(net, "p", "line", 0.0111, 0.05, 0, "from")  # p12
    create_measurement(net, "q", "line", 0.06, 0.05, 0, "from")  # q12

    create_measurement(net, "v", "bus", 1.019, 0.01, 0)  # u1
    create_measurement(net, "v", "bus", 1.04, 0.01, 1)  # u2

    # 2. Do state estimation
    if not estimate(net, algorithm='lav', init='flat'):
        raise AssertionError("Estimation failed!")

    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([[1.02083378, 1.03812899]])
    diff_v = target_v - v_result
    target_delta = np.array([[0.0, 3.11356604]])
    diff_delta = target_delta - delta_result

    if np.nanmax(abs(diff_v)) >= 1e-6 or np.nanmax(abs(diff_delta)) >= 1e-6:
        raise AssertionError("Estimation failed!")