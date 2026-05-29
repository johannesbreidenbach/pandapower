# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import numpy as np
import pandas as pd
import os

# imports from pandapower
import pandapower.networks as pn
from pandapower.run import runpp
from pandapower.estimation import estimate
from pandapower.create import (create_measurement, create_empty_network, create_bus, create_ext_grid,
                               create_line_from_parameters, create_load, create_sgen)
from pandapower.auxiliary import pandapowerNet

from pandapower.test.estimation.test_lav_estimation import _r


# begin functions
def _create_measurement_18_bus_grid(net: pandapowerNet, rv: float = .0, rp: float = .0, rq: float = .0) -> None:

    # =========================================================================
    # Bus 1: v, p, q (index: 0)
    # =========================================================================
    create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=net.res_bus.vm_pu[0] * _r(rv),
        std_dev=0.01,
        element=0
    )
    create_measurement(
        net,
        meas_type="p",
        element_type="bus",
        value=net.res_bus.p_mw[0] * _r(rp),
        std_dev=max(0.001, abs(0.03 * net.res_bus.p_mw[0])),
        element=0)
    create_measurement(
        net,
        meas_type="q",
        element_type="bus",
        value=net.res_bus.q_mvar[0] * _r(rq),
        std_dev=max(0.001, abs(0.03 * net.res_bus.q_mvar[0])),
        element=0)
    # =========================================================================
    # Bus 4: voltage measurement (index: 3)
    # =========================================================================
    create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=net.res_bus.vm_pu[3] * _r(rv),
        std_dev=0.01,
        element=3
    )
    # =========================================================================
    # Line 4 -> 5 (Bus): active/reactive power flow measurement
    # Line index = 3 ( 3 -> 4 Busindex)
    # =========================================================================
    create_measurement(
        net,
        meas_type="p",
        element_type="line",
        value=net.res_line.p_from_mw[3] * _r(rp),
        std_dev=max(0.001, abs(0.03 * net.res_line.p_from_mw[3])),
        element=3,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[3] * _r(rq),
        std_dev=max(0.001, abs(0.03 * net.res_line.q_from_mvar[3])),
        element=3,
        side="from"
    )
    # =========================================================================
    # Line 4 -> 15 (Bus): active/reactive power flow measurement
    # Line index = 13 (3 -> 14 Busindex)
    # =========================================================================
    create_measurement(
        net,
        meas_type="p",
        element_type="line",
        value=net.res_line.p_from_mw[13] * _r(rp),
        std_dev=max(0.001, abs(0.03 * net.res_line.p_from_mw[13])),
        element=13,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[13] * _r(rq),
        std_dev=max(0.001, abs(0.03 * net.res_line.q_from_mvar[13])),
        element=13,
        side="from"
    )
    # =========================================================================
    # Bus 10: voltage measurement (index: 9)
    # =========================================================================
    create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=net.res_bus.vm_pu[9] * _r(rv),
        std_dev=0.01,
        element=9
    )
    # =========================================================================
    # Line 10 -> 11 (Bus): active/reactive power flow measurement
    # Line index = 9 (9 -> 10 Busindex)
    # =========================================================================
    create_measurement(
        net,
        meas_type="p",
        element_type="line",
        value=net.res_line.p_from_mw[9] * _r(rp),
        std_dev=max(0.001, abs(0.03 * net.res_line.p_from_mw[9])),
        element=9,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[9] * _r(rq),
        std_dev=max(0.001, abs(0.03 * net.res_line.q_from_mvar[9])),
        element=9,
        side="from"
    )


def _create_18_bus_grid_with_measurement_af(
        base_mva: float = 10.0,
        slack_v: float = 1.0,
        slack_va_degree: float = 0.0,
        seed: int | None = None,
        rv: float = .01,
        rp: float = .03,
        rq: float = .03
) -> tuple[pandapowerNet, dict[str, np.ndarray]]:
    """
    Create the 18-bus radial distribution network from the original MATLAB implementation and run a power flow
    calculation using pandapower.

    The network is modeled as an 11 kV radial distribution grid with residential loads, commercial loads, photovoltaic
    (PV) generation, and wind generation connected to different buses.

    The original MATLAB implementation uses per-unit (p.u.) values. Since pandapower expects physical units, all line
    impedances and power values are converted to engineering units before creating the network elements.

    A random operating point is generated for each simulation by scaling residential loads, commercial loads,
    PV generation and wind generation with uniformly distributed random factors.

    Parameters:
        base_mva: Base apparent power of the system in MVA. Corresponds to ``Sb`` in the MATLAB implementation.

        slack_v: Voltage magnitude of the slack bus in per-unit.

        slack_va_degree: Voltage angle of the slack bus in degrees.

        seed:
            Optional random seed for reproducible simulations. If ``None``, random values are generated for every call.

        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage

        rp: standard deviation to apply a multiplicative perturbation to quantities for active power

        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power

    Returns:
        Return a pandapower net and a dict with the random scaling factors:

            - **net**: The pandapower network including buses, lines, loads, generators, and power flow results.
            - **K**: Dictionary containing the random scaling coefficients:

                - ``KL_res``: residential load scaling
                - ``KL_com``: commercial load scaling
                - ``KG_pv``: PV generation scaling
                - ``KG_wind``: wind generation scaling
    """

    if seed is not None:
        np.random.seed(seed)  # Set deterministic random numbers for reproducible simulations

    # =========================================================================
    # Base values
    # =========================================================================
    # The original MATLAB implementation uses per-unit (p.u.) values.
    #
    # pandapower expects physical units:
    #   - voltage in kV
    #   - power in MW / MVAr
    #   - impedance in Ohm
    #
    # Therefore, the per-unit values must be converted.
    #
    # Base voltage of the distribution grid
    v_b = 11.0  # kV
    # Base impedance:
    #
    #     Z_base = V_base² / S_base
    #
    # With:
    #     V_base in kV
    #     S_base in MVA
    #
    # This gives:
    #     Z_base = 11² / 10 = 12.1 Ohm
    #
    # Used to convert line impedances:
    #
    #     Z_ohm = Z_pu * Z_base
    z_base = (v_b ** 2) / base_mva  # Ohm

    # =========================================================================
    # Create empty power network for power flow calculation
    # =========================================================================
    net_pf: pandapowerNet = create_empty_network(sn_mva=base_mva, f_hz=50)

    # =========================================================================
    # 1) Buses
    # =========================================================================
    # Create 18 buses corresponding to the original MATLAB network.
    #
    # Bus 1:
    #     Slack / reference bus
    #
    # Bus 2-18:
    #     PQ buses with loads and distributed generation
    #
    buses = []
    for i in range(1, 19):
        bus = create_bus(net_pf, vn_kv=v_b, name=f"Bus {i}")
        buses.append(bus)
    # Create slack bus / external grid connection
    # vm_pu: voltage magnitude in per-unit
    # degree: voltage angle in degrees
    #
    create_ext_grid(net_pf, bus=buses[0], vm_pu=slack_v, degree=slack_va_degree, name="Slack")

    # =========================================================================
    # 2) Lines
    # =========================================================================
    start = np.array([1, 2, 3, 4, 5, 6, 6, 8, 9, 10, 11, 11, 13, 4, 15, 16, 16])
    end   = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    # Per-unit line resistances
    r_pu = np.array([0.00001, 0.0174, 0.0001, 0.0052, 0.0003, 0.0010, 0.0017, 0.0022, 0.0001, 0.0016, 0.0007, 0.0299,
                     0.0010, 0.0025, 0.0041, 0.0034, 0.0013])  # change zeros to small value for pf-calc
    # Per-unit line reactances
    x_pu = np.array([0.1000, 0.0085, 0.0001, 0.0028, 0.0002, 0.0010, 0.0008, 0.0011, 0.00002, 0.0008, 0.0003, 0.0081,
                     0.0010, 0.0007, 0.0013, 0.0009, 0.0004])  # change zeros to small value for pf-calc
    # Create lines
    #
    # The MATLAB data provides total line impedances in per-unit.
    #
    # Pandapower requires:
    #     r_ohm_per_km
    #     x_ohm_per_km
    #
    # Since no physical line lengths are available,
    # each line is modeled with:
    #
    #     length_km = 1.0
    #
    # Therefore:
    #
    #     impedance_per_km == total_impedance
    #
    for idx, (s, e, r, x) in enumerate(zip(start, end, r_pu, x_pu)):
        create_line_from_parameters(
            net_pf,
            from_bus=buses[s - 1],
            to_bus=buses[e - 1],
            length_km=1.0,
            r_ohm_per_km=r * z_base,
            x_ohm_per_km=x * z_base,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
            name=f"Line {idx + 1}"
        )
    net_se = copy.deepcopy(net_pf)  # create second net for state estimation
    # =========================================================================
    # 3) Loads and distributed generation
    # =========================================================================
    # The original MATLAB model contains:
    #
    #   - residential loads
    #   - commercial loads
    #   - photovoltaic generation
    #   - wind generation
    #
    # All values are defined in per-unit on the system base power.
    #
    # Positive values:
    #     generation
    #
    # Negative values in MATLAB:
    #     loads
    #
    # In pandapower:
    #     loads are modeled as positive consumption
    #
    p_l_res = 2 * np.array([0.05, 0.08, 0, 0.05, 0, 0.06, 0, 0.02, 0.04, 0, 0.09, 0, 0.08, 0, 0, 0.05, 0.07])
    q_l_res = 2 * np.array([0.01, 0.02, 0, 0.01, 0, 0.01, 0, 0.01, 0.01, 0, 0.01, 0, 0.01, 0, 0, 0.01, 0.01])

    p_l_com = np.array([0.03, 0.08, 0, 0.05, 0, 0.05, 0, 0.07, 0.03, 0, 0.01, 0, 0.03, 0, 0, 0.01, 0.02])
    q_l_com = np.array([0.02, 0.02, 0, 0.01, 0, 0.01, 0, 0.01, 0.01, 0, 0.01, 0, 0.01, 0, 0, 0.01, 0.01])

    p_g_pv = np.array([0.04, 0.05, 0, 0.02, 0, 0.08, 0, 0.05, 0.03, 0, 0.04, 0, 0.05, 0, 0, 0.07, 0.02])
    q_g_pv = -np.array([0.00, 0.00, 0, 0.00, 0, 0.01, 0, 0.00, 0.00, 0, 0.00, 0, 0.01, 0, 0, 0.00, 0.00])

    p_g_wind = np.array([0.00, 0.00, 0, 0.07, 0, 0.00, 0, 0.00, 0.03, 0, 0.00, 0, 0.08, 0, 0, 0.04, 0.04])
    q_g_wind = -np.array([0.00, 0.00, 0, 0.00, 0, 0.00, 0, 0.00, 0.01, 0, 0.00, 0, 0.01, 0, 0, 0.00, 0.00])

    # Dictionary storing the random scaling factors
    k_dc = {
        "KL_res": np.zeros(17),
        "KL_com": np.zeros(17),
        "KG_pv": np.zeros(17),
        "KG_wind": np.zeros(17)
    }
    # =========================================================================
    # Create loads and generators at each bus
    # =========================================================================
    for idx in range(17):
        bus = buses[idx + 1] # busses 2-18
        # Random operating-point scaling factors
        #
        # Residential load:
        #     50% - 80%
        #
        # Commercial load:
        #     30% - 60%
        #
        # PV:
        #     30% - 40%
        #
        # Wind:
        #     20% - 40%
        #
        var_l_res = 0.5 + 0.3 * np.random.rand()
        var_l_com = 0.3 + 0.3 * np.random.rand()
        var_g_pv = 0.3 + 0.1 * np.random.rand()
        var_g_wind = 0.2 + 0.2 * np.random.rand()

        k_dc["KL_res"][idx] = var_l_res
        k_dc["KL_com"][idx] = var_l_com
        k_dc["KG_pv"][idx] = var_g_pv
        k_dc["KG_wind"][idx] = var_g_wind

        # =====================================================================
        # Apply scaling factors to nominal per-unit values
        # =====================================================================
        p_res_pu = float(p_l_res[idx] * var_l_res)
        q_res_pu = float(q_l_res[idx] * var_l_res)

        p_com_pu = float(p_l_com[idx] * var_l_com)
        q_com_pu = float(q_l_com[idx] * var_l_com)

        p_pv_pu = float(p_g_pv[idx] * var_g_pv)
        q_pv_pu = float(q_g_pv[idx] * var_g_pv)

        p_wind_pu = float(p_g_wind[idx] * var_g_wind)
        q_wind_pu = float(q_g_wind[idx] * var_g_wind)

        # =====================================================================
        # Convert per-unit values to MW / MVAr
        # =====================================================================
        #
        # Conversion:
        #
        #     P_MW = P_pu * S_base
        #
        #     Q_MVAr = Q_pu * S_base
        #
        # Loads are created as separate elements:
        #     - residential
        #     - commercial
        #
        # Generators are created as:
        #     - pv
        #     - wind
        # For state estimation the nominal values are used

        # Residential load
        if p_res_pu != 0 or q_res_pu != 0:
            create_load(
                net_pf,
                bus=bus,
                p_mw=p_res_pu * base_mva,
                q_mvar=q_res_pu * base_mva,
                name=f"Residential Load Bus {idx + 2}",
                type="residential"
            )
            create_load(
                net_se,
                bus=bus,
                p_mw=p_l_res[idx] * base_mva,
                q_mvar=q_l_res[idx] * base_mva,
                name=f"Residential Load Bus {idx + 2}",
                type="residential"
            )
        # Commercial load
        if p_com_pu != 0 or q_com_pu != 0:
            create_load(
                net_pf,
                bus=bus,
                p_mw=p_com_pu * base_mva,
                q_mvar=q_com_pu * base_mva,
                name=f"Commercial Load Bus {idx + 2}",
                type="commercial"
            )
            create_load(
                net_se,
                bus=bus,
                p_mw=p_l_com[idx] * base_mva,
                q_mvar=q_l_com[idx] * base_mva,
                name=f"Commercial Load Bus {idx + 2}",
                type="commercial"
            )
        # Photovoltaic generation
        if p_pv_pu != 0 or q_pv_pu != 0:
            create_sgen(
                net_pf,
                bus=bus,
                p_mw=p_pv_pu * base_mva,
                q_mvar=q_pv_pu * base_mva,
                name=f"PV Bus {idx + 2}",
                type="pv"
            )
            create_sgen(
                net_se,
                bus=bus,
                p_mw=p_g_pv[idx] * base_mva,
                q_mvar=q_g_pv[idx] * base_mva,
                name=f"PV Bus {idx + 2}",
                type="pv"
            )
        # Wind generation
        if p_wind_pu != 0 or q_wind_pu != 0:
            create_sgen(
                net_pf,
                bus=bus,
                p_mw=p_wind_pu * base_mva,
                q_mvar=q_wind_pu * base_mva,
                name=f"Wind Bus {idx + 2}",
                type="wind"
            )
            create_sgen(
                net_se,
                bus=bus,
                p_mw=p_g_wind[idx] * base_mva,
                q_mvar=q_g_wind[idx] * base_mva,
                name=f"Wind Bus {idx + 2}",
                type="wind"
            )

    # =========================================================================
    # Run power flow calculation
    # =========================================================================
    #
    # The BFSW (Backward-Forward Sweep) algorithm is used because:
    #
    #   - the network is radial
    #   - the network has high R/X ratios
    #   - some lines have very small reactance
    #
    # BFSW is numerically more robust for distribution grids than Newton-Raphson in this case.
    #
    runpp(net_pf)
    for key in net_pf.keys():
        if key.startswith("res_"):
            net_se[key] = net_pf[key].copy(deep=True)
    _create_measurement_18_bus_grid(net=net_se, rv=rv, rp=rp, rq=rq)  # rv=.01, rp=.03, rq=.03
    return net_se, k_dc


def _add_measurements_af(
        net_base: pandapowerNet,
        measurement_interval: int = 1,
        seed: int = 112,
        rv: float = .01,
        rp: float = .03,
        rq: float = .03
) -> pandapowerNet:
    # 1. load mv oberrhein
    np.random.seed(seed)

    runpp(net_base)

    # 2. create measurements ONCE on the base network
    for i, (bus, row) in enumerate(net_base.res_bus.iterrows()):
        if i % measurement_interval != 0:
            continue

        create_measurement(
            net_base,
            meas_type="v",
            element_type="bus",
            value=row.vm_pu * _r(rv),
            std_dev=0.01,
            element=bus
        )
        create_measurement(
            net=net_base,
            meas_type="p",
            element_type="bus",
            value=row.p_mw * _r(rp),
            std_dev=max(0.001, abs(0.03 * row.p_mw)),
            element=bus
        )
        create_measurement(
            net=net_base,
            meas_type="q",
            element_type="bus",
            value=row.q_mvar * _r(rq),
            std_dev=max(0.001, abs(0.03 * row.q_mvar)),
            element=bus
        )
    return net_base


def test_general_function(net_base: pandapowerNet, init="flat" , save_path: str = "."):
    """
       Run power flow and three different state estimations (WLS, LAV, WLAV) on different networks and return bus
       voltages, angles, and deviations. The power grid is unobservable.

       The network is solved once with a power flow, measurements are created, and then the network is copied for each
       estimation algorithm. Results from power flow and estimation, as well as their differences, are collected in a
       pandas DataFrame.

       Parameters:
           net_base: A pandapower grid, where the different state estimation solver will be used.
           init: Initialization method for the state estimation (e.g. ``"flat"``).
           save_path: path where the resolution dataframes will save.

       Returns:
           DataFrame with columns:
           ``[
               "V PF", "angle PF",
               "V WLS", "angle WLS", "dV WLS", "dAngle WLS",
               "V LAV", "angle LAV", "dV LAV", "dAngle LAV",
               "V WLAV", "angle WLAV", "dV WLAV", "dAngle WLAV"
           ]``.
       """


    # 3. create copies for each estimation algorithm
    net_afwls = copy.deepcopy(net_base)
    net_aflav = copy.deepcopy(net_base)
    net_afwlav = copy.deepcopy(net_base)


    # 4. run estimations with soft-fail (collect errors, do not raise immediately)
    failures = []

    # AF-WLS
    afwls = estimate(net_afwls, algorithm="af-wls", init="flat", wlav=False, af_init_value=.5)
    if not afwls["success"]:
        failures.append("AF-WLS estimation failed")
        v_afwls = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_afwls = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_afwls = net_afwls.res_bus_est.vm_pu.values
        delta_afwls = net_afwls.res_bus_est.va_degree.values

    # LAV
    aflav = estimate(net_aflav, algorithm="af-lp", wlav=False, with_ortools=False, init="flat", debug_mode=False)
    if not aflav["success"]:
        failures.append("LAV estimation failed")
        v_aflav = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_aflav = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_aflav = net_aflav.res_bus_est.vm_pu.values
        delta_aflav = net_aflav.res_bus_est.va_degree.values

    # AF-WLAV
    afwlav = estimate(net_afwlav, algorithm="af-lp", wlav=True, with_ortools=False, init="flat", debug_mode=False)
    if not afwlav["success"]:
        failures.append("AF-WLAV estimation failed")
        v_afwlav = np.full_like(net_base.res_bus.vm_pu.values, np.nan, dtype=float)
        delta_afwlav = np.full_like(net_base.res_bus.va_degree.values, np.nan, dtype=float)
    else:
        v_afwlav = net_afwlav.res_bus_est.vm_pu.values
        delta_afwlav = net_afwlav.res_bus_est.va_degree.values

    # 5. power flow results (runpp) aus net_base
    v_pf = net_base.res_bus.vm_pu.values
    delta_pf = net_base.res_bus.va_degree.values

    # 6. Differences Estimation - PowerFlow
    d_v_afwls = v_afwls - v_pf
    d_a_afwls = delta_afwls - delta_pf

    d_v_aflav = v_aflav - v_pf
    d_a_aflav = delta_aflav - delta_pf

    d_v_afwlav = v_afwlav - v_pf
    d_a_afwlav = delta_afwlav - delta_pf

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
    res_total_df.to_excel(os.path.join(save_path, "total_df.xlsx"), index=False)

    res_diff_df = pd.DataFrame(
        {
            "dV AF-WLS": d_v_afwls,
            "dA AF-WLS": d_a_afwls,
            "dV AF-LAV": d_v_aflav,
            "dA AF-LAV": d_a_aflav,
            "dV AF-WLAV": d_v_afwlav,
            "dA AF-WLAV": d_a_afwlav,
        },
        index=net_base.res_bus.index,  # Bus-Index als Index
    )
    res_diff_df.to_excel(os.path.join(save_path, "diff_df.xlsx"), index=False)

    res_diff_max_df = pd.DataFrame(
        data={
            "AF WLS V": [np.max(np.abs(d_v_afwls))],
            "AF WLS A": [np.max(np.abs(d_a_afwls))],
            "AF VLA V": [np.max(np.abs(d_v_aflav))],
            "AF VLA A": [np.max(np.abs(d_a_aflav))],
            "AF WLAV V": [np.max(np.abs(d_v_afwlav))],
            "AF WLAV A": [np.max(np.abs(d_a_afwlav))]
        }
    )
    res_diff_max_df.to_excel(os.path.join(save_path, "diff_max_df.xlsx"), index=False)

    # set index for resolution dataframe (allocation facotors)
    afwls["allocation_factors"].index = ["AF-WLS"]
    afwlav["allocation_factors"].index = ["AF-WLAV"]
    aflav["allocation_factors"].index = ["AF-LAV"]

    res_af_df = pd.concat([afwls["allocation_factors"], afwlav["allocation_factors"],aflav["allocation_factors"]])
    res_af_df.to_excel(os.path.join(save_path, "af_df.xlsx"), index=True)

    # 8. Checks: apply only to successful estimates
    if "AF-WLS estimation failed" not in failures:
        assert np.nanmax(abs(d_v_afwls)) < 0.0043
        assert np.nanmax(abs(d_a_afwls)) < 0.2

    if "AF-LAV estimation failed" not in failures:
        assert np.nanmax(abs(d_v_aflav)) < 0.0043
        assert np.nanmax(abs(d_a_aflav)) < 0.2

    if "AF-WLAV estimation failed" not in failures:
        assert np.nanmax(abs(d_v_afwlav)) < 0.0043
        assert np.nanmax(abs(d_a_afwlav)) < 0.2

    # 9. Evaluate the overall result at the end
    if failures:
        # In this case, the test is only terminated at the end, once all networks have been calculated
        raise AssertionError(" | ".join(failures))


if __name__ == '__main__':

    s_path = r"D:\forschungspunkte\state_estimation\data\18_bus"

    mv_b: bool = False
    ieee14_b: bool = False
    ieee30_b: bool = False
    bus18: bool = True

    if mv_b:
        # rv=.01, rp=.03, rq=.03
        net_mv = _add_measurements_af(pn.mv_oberrhein(),15, 112, .0, .0, .0)
        test_general_function(net_mv, save_path=s_path)

    if ieee14_b:
        net14 = _add_measurements_af(pn.case14(), 2, 112, .0, .0, .0)
        test_general_function(net14, save_path=s_path)

    if ieee30_b:
        net30 = _add_measurements_af(pn.case30(), 5, 112, .0, .0, .0)
        test_general_function(net30, save_path=s_path)

    if bus18:
        net18, k = _create_18_bus_grid_with_measurement_af(seed=112, rv=.0, rp=.0, rq=.0) # rv=.01, rp=.03, rq=.03
        test_general_function(net18, save_path=s_path)

    print(f"whats up")
