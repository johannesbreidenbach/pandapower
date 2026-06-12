# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import numpy as np
import pandas as pd
import os
import simbench as sb
import time

from datetime import timedelta
from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dotenv import load_dotenv

# imports from pandapower
import pandapower.networks as pn
from pandapower import to_pickle, from_pickle
from pandapower.run import runpp
from pandapower.estimation import estimate
from pandapower.create import (create_measurement, create_empty_network, create_bus, create_ext_grid,
                               create_line_from_parameters, create_load, create_sgen)
from pandapower.auxiliary import pandapowerNet

from pandapower.test.estimation.test_lav_estimation import _r


# begin functions
def _add_measurements_af(
        net_base: pandapowerNet,
        seed_m: int | None = None,
        measurement_interval: int = 1,
        rv: float = .01,
        rp: float = .03,
        rq: float = .03
) -> None:
    """
    Add measurements to test gird for state estimation with allocation factors.

    With the parameter ``measurement_interval`` various properties can be set. Fully observable (1),
    non-observable (>1). The measurement uncertainty can be set with ``rv, rp, rq`` for voltage, active and reactive
    power. To the net will add measurements which comes from powerflow calculation results with statistic uncertainties.

    Arguments:
        net_base: net with or without load flow calculations can be imported.
        seed_m: Attention if None, no seed will be used different to normal use case.
        measurement_interval:
            Integer greater than or equal to 1. Measurements are added to every `measurement_interval`-th bus (according
            to the bus order in `net_base.res_bus`). Higher values result in fewer measurements. The allocation is
            systematic and not random.
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power

    Returns:
        None
    """
    if seed_m is not None:
        np.random.seed(seed_m)

    if not net_base.converged:
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
            std_dev=max(.001, abs(rv * row.vm_pu)),
            element=int(bus)
        )
        create_measurement(
            net=net_base,
            meas_type="p",
            element_type="bus",
            value=row.p_mw * _r(rp),
            std_dev=max(.001, abs(rp * row.p_mw)),
            element=int(bus)
        )
        create_measurement(
            net=net_base,
            meas_type="q",
            element_type="bus",
            value=row.q_mvar * _r(rq),
            std_dev=max(.001, abs(rq * row.q_mvar)),
            element=int(bus)
        )


def _fill_measurement_values_from_powerflow(
    net: pandapowerNet,
    seed_m: int | None = None,
    rv: float = 0.01,
    ri: float = 0.01,
    rp: float = 0.03,
    rq: float = 0.03
) -> None:
    """
    Fill existing pandapower measurements with values from power flow results.

    Existing entries in net.measurement are not recreated. Only `value` and `std_dev` are updated. The measurement
    values are taken from the corresponding result tables and multiplied by `_r(...)` to add measurement uncertainty.

    Parameters:
        net: net with or without load flow calculations can be imported.
        seed_m: Attention if None, no seed will be used different to normal use case
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        ri: standard deviation to apply a multiplicative perturbation to quantities for current
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power

    Returns: None
    """

    if seed_m is not None:
        np.random.seed(seed_m)

    if not net.converged or net.res_bus.empty:
        runpp(net)

    for idx, meas in net.measurement.iterrows():
        meas_type = meas["measurement_type"]
        element_type = meas["element_type"]
        element = int(meas["element"])
        side = meas.get("side", None)

        value: float | None = None
        std_dev: float | None = None

        # --- bus measurements ---
        if element_type == "bus":
            if meas_type == "v":
                base_value = float(net.res_bus.at[element, "vm_pu"])
                value = base_value * _r(rv)
                std_dev = max(0.001, abs(rv * base_value))

            elif meas_type == "p":
                base_value = float(net.res_bus.at[element, "p_mw"])
                value = base_value * _r(rp)
                std_dev = max(0.001, abs(rp * base_value))

            elif meas_type == "q":
                base_value = float(net.res_bus.at[element, "q_mvar"])
                value = base_value * _r(rq)
                std_dev = max(0.001, abs(rq * base_value))

        # --- line measurements ---
        elif element_type == "line":
            if side == "from":
                prefix = "from"
            elif side == "to":
                prefix = "to"
            else:
                raise ValueError(f"Invalid or missing side for line measurement at index {idx}: {side}")

            if meas_type == "p":
                base_value = float(net.res_line.at[element, f"p_{prefix}_mw"])
                value = base_value * _r(rp)
                std_dev = max(0.001, abs(rp * base_value))

            elif meas_type == "q":
                base_value = float(net.res_line.at[element, f"q_{prefix}_mvar"])
                value = base_value * _r(rq)
                std_dev = max(0.001, abs(rq * base_value))

            elif meas_type == "i":
                base_value = float(net.res_line.at[element, f"i_{prefix}_ka"])
                value = base_value * _r(ri)
                std_dev = max(0.001, abs(ri * base_value))

        # --- transformer measurements ---
        elif element_type == "trafo":
            if side not in ["hv", "lv"]:
                raise ValueError(f"Invalid or missing side for trafo measurement at index {idx}: {side}")

            if meas_type == "p":
                base_value = float(net.res_trafo.at[element, f"p_{side}_mw"])
                value = base_value * _r(rp)
                std_dev = max(0.001, abs(rp * base_value))

            elif meas_type == "q":
                base_value = float(net.res_trafo.at[element, f"q_{side}_mvar"])
                value = base_value * _r(rq)
                std_dev = max(0.001, abs(rq * base_value))

            elif meas_type == "i":
                base_value = float(net.res_trafo.at[element, f"i_{side}_ka"])
                value = base_value * _r(ri)
                std_dev = max(0.001, abs(ri * base_value))

        if value is not None:
            net.measurement.at[idx, "value"] = value
            net.measurement.at[idx, "std_dev"] = std_dev


def _create_simbench_mc_case(
    net,
    seed_pf: int | None = None,
    load_range: tuple[float, float] = (.5, .8),
    sgen_range: tuple[float, float] = (.3, .5),
) -> dict[str, dict]:
    """
    Generate a random operating point for a SimBench network and perform a power flow calculation.

    A copy of the input network is created and the active and reactive powers of all loads and static generators are
    scaled by uniformly distributed random factors. The resulting network represents the "true" operating state for the
    current Monte Carlo iteration.

    After the power flow calculation, the result tables (``res_*``) of the perturbed network are copied back to the
    original network. Consequently, the original network retains its nominal load and generation values while containing
    the power flow results of the randomly perturbed operating point. This enables state estimation with nominal values
    and measurements derived from varying operating conditions.

    Parameters:
        net: SimBench network containing the nominal load and generation values.
        seed_pf: Attention if None, no seed will be used different to normal use case.
        load_range: Lower and upper bounds of the uniformly distributed scaling factors applied to loads.
        sgen_range: Lower and upper bounds of the uniformly distributed scaling factors applied to sgens.

    Returns:
        Scaling parameters for loads and sgens.
    """

    # ToDo: Check what the scaling factor in load, sgen, gen does
    if seed_pf is not None:
        np.random.seed(seed_pf)

    net_pf = copy.deepcopy(net)

    k = {
        "load": {},
        "sgen": {},
    }

    # scale loads in net_pf
    for idx in net_pf.load.index:
        factor = np.random.uniform(*load_range)
        k["load"][idx] = factor

        net_pf.load.at[idx, "p_mw"] *= factor
        net_pf.load.at[idx, "q_mvar"] *= factor

    # sclae sgen in net_pf
    for idx in net_pf.sgen.index:
        factor = np.random.uniform(*sgen_range)
        k["sgen"][idx] = factor

        net_pf.sgen.at[idx, "p_mw"] *= factor
        net_pf.sgen.at[idx, "q_mvar"] *= factor

    # run powerflow with scaled values
    runpp(net_pf)

    # copy results from powerflow to net for state estimation
    for key in net_pf.keys():
        if key.startswith("res_"):
            net[key] = net_pf[key].copy(deep=True)
    return k


def _create_measurement_18_bus_grid(
        net: pandapowerNet,
        rv: float = .01,
        rp: float = .03,
        rq: float = .03
) -> None:
    """
    Add measurements to the 18 bus test gird for state estimation.

    This function adds a predefined set of measurement points at fixed locations. The measurement values are derived
    from the power flow results of the grid and are perturbed with Gaussian noise, whose magnitude is controlled by the
    given standard deviations.

    Parameters:
        net: power grid
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power
    """
    # if seed is not None:
    #    np.random.seed(seed)  # Set deterministic random numbers for reproducible simulations
    # =========================================================================
    # Bus 1: v, p, q (index: 0)
    # =========================================================================
    create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=net.res_bus.vm_pu[0] * _r(rv),
        std_dev=max(.001, abs(rv * net.res_bus.vm_pu[0])),
        element=0
    )
    create_measurement(
        net,
        meas_type="p",
        element_type="bus",
        value=net.res_bus.p_mw[0] * _r(rp),
        std_dev=max(.001, abs(rp * net.res_bus.p_mw[0])),
        element=0)
    create_measurement(
        net,
        meas_type="q",
        element_type="bus",
        value=net.res_bus.q_mvar[0] * _r(rq),
        std_dev=max(.001, abs(rq * net.res_bus.q_mvar[0])),
        element=0
    )
    # =========================================================================
    # Bus 4: voltage measurement (index: 3)
    # =========================================================================
    create_measurement(
        net,
        meas_type="v",
        element_type="bus",
        value=net.res_bus.vm_pu[3] * _r(rv),
        std_dev=max(.001, abs(rv * net.res_bus.vm_pu[3])),
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
        std_dev=max(.001, abs(rp * net.res_line.p_from_mw[3])),
        element=3,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[3] * _r(rq),
        std_dev=max(.001, abs(rq * net.res_line.q_from_mvar[3])),
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
        std_dev=max(.001, abs(rp * net.res_line.p_from_mw[13])),
        element=13,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[13] * _r(rq),
        std_dev=max(.001, abs(rq * net.res_line.q_from_mvar[13])),
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
        std_dev=max(.001, abs(rv * net.res_bus.vm_pu[9])),
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
        std_dev=max(.001, abs(rp * net.res_line.p_from_mw[9])),
        element=9,
        side="from"
    )
    create_measurement(
        net,
        meas_type="q",
        element_type="line",
        value=net.res_line.q_from_mvar[9] * _r(rq),
        std_dev=max(.001, abs(rq * net.res_line.q_from_mvar[9])),
        element=9,
        side="from"
    )


def _create_18_bus_grid(
        base_mva: float = 10.0,
        slack_v: float = 1.0,
        slack_va_degree: float = 0.0,
        load_range: tuple[float, float] = (.5, .8),
        com_range: tuple[float, float] = (.3, .6),
        pv_range: tuple[float, float] = (.3, .4),
        wind_range: tuple[float, float] = (.2, .4)
) -> tuple[pandapowerNet, dict[str, np.ndarray]]:
    """
    Create the 18-bus radial distribution network from the original MATLAB implementation and run a power flow
    calculation using pandapower.

    The network is modeled as an 11 kV radial distribution grid with residential loads, commercial loads, photovoltaic
    (PV) generation, and wind generation connected to different buses.

    The original MATLAB implementation uses per-unit (p.u.) values. Since pandapower expects physical units, all line
    impedance and power values are converted to engineering units before creating the network elements.

    A random operating point is generated for each simulation by scaling residential loads, commercial loads,
    PV generation and wind generation with uniformly distributed random factors.

    For the **power flow calculation**, these randomly scaled (statistically perturbed) values are used.
    For the **state estimation network**, the corresponding **nominal** (unscaled) load and generation values are
    stored, while the power flow results (voltages, line flows, etc.) from the randomly scaled case ar copied into the
    result tables of the state estimation network.

    Parameters:
        base_mva: Base apparent power of the system in MVA. Corresponds to ``Sb`` in the MATLAB implementation.
        slack_v: Voltage magnitude of the slack bus in per-unit.
        slack_va_degree: Voltage angle of the slack bus in degrees.
        load_range: Range of scaling factor (uniformly distributed) for load (residential load)
        com_range: Range of scaling factor (uniformly distributed) for load (commercial load)
        pv_range: Range of scaling factor (uniformly distributed) for sgne (PV generation)
        wind_range: Range of scaling factor (uniformly distributed) for wind (wind generation)

    Returns:
        Return a pandapower net with the results from the powerflow calculation and the nominal loads and powers for
        state estimation. The dict includes the random scaling factors:

            - **net**: The pandapower network including buses, lines, loads, generators, and power flow results.
            - **K**: Dictionary containing the random scaling coefficients:

                - ``KL_res``: residential load scaling
                - ``KL_com``: commercial load scaling
                - ``KG_pv``: PV generation scaling
                - ``KG_wind``: wind generation scaling
    """

    # if seed is not None:
    #     np.random.seed(seed)  # Set deterministic random numbers for reproducible simulations

    # =========================================================================
    # Base values
    # =========================================================================
    # The original MATLAB implementation uses per-unit (p.u.) values.
    #
    # pandapower expects physical units:
    #   - voltage in kV
    #   - power in MW / MVAR
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
    # Used to convert line impedance:
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
    # The MATLAB data provides total line impedance in per-unit.
    #
    # pandapower requires:
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

    # Nominal values these values are used for state estimation and adapted for powerflow calculation
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
        bus = buses[idx + 1] # buses 2-18
        # Random operating-point scaling factors
        #
        # Residential load:
        #     50% - 80%
        # Commercial load:
        #     30% - 60%
        # PV:
        #     30% - 40%
        # Wind:
        #     20% - 40%
        #
        var_l_res = np.random.uniform(*load_range)  # ToDo: create other scaling values to get better behavior
        var_l_com = np.random.uniform(*com_range)
        var_g_pv = np.random.uniform(*pv_range)
        var_g_wind = np.random.uniform(*wind_range)

        k_dc["KL_res"][idx] = var_l_res
        k_dc["KL_com"][idx] = var_l_com
        k_dc["KG_pv"][idx] = var_g_pv
        k_dc["KG_wind"][idx] = var_g_wind

        # =====================================================================
        # Apply scaling factors to nominal per-unit values for powerflow
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
        # Convert per-unit values to MW / MVAR
        # =====================================================================
        #
        # Conversion:
        #
        #     P_MW = P_pu * S_base
        #
        #     Q_MVAR = Q_pu * S_base
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
                p_mw=p_l_res[idx] * base_mva,  # nominal value
                q_mvar=q_l_res[idx] * base_mva,  # nominal value
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
                p_mw=p_l_com[idx] * base_mva,  # nominal value
                q_mvar=q_l_com[idx] * base_mva,  # nominal value
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
                p_mw=p_g_pv[idx] * base_mva,  # nominal value
                q_mvar=q_g_pv[idx] * base_mva,  # nominal value
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
                p_mw=p_g_wind[idx] * base_mva,  # nominal value
                q_mvar=q_g_wind[idx] * base_mva,  # nominal value
                name=f"Wind Bus {idx + 2}",
                type="wind"
            )

    # =========================================================================
    # Run power flow calculation
    # =========================================================================
    # after run powerflow the results will save in the net tables for state estimation
    # ToDo: for future general use net_pf and net_se should separately return
    runpp(net_pf)
    for key in net_pf.keys():
        if key.startswith("res_"):
            net_se[key] = net_pf[key].copy(deep=True)
    return net_se, k_dc


def _calc_different_se(net_base: pandapowerNet, failures: list, num_it: str, save_path: str = ".") -> None:
    """
    un three different allocation-factor-based state estimation methods (AF-WLS, AF-WLAV, AF-LAV) for a specific power
    grid and return bus voltages, angles, deviations and the allocation factors. The power grid is unobservable.

    For each estimator:
        1. A deep copy of ``net_base`` is created.
        2. State estimation is run with the specified algorithm.
        3. The resulting pandapower network, including state estimation results, is saved as a pickle file.
        4. The estimated allocation factors are collected, and a method-specific index (AF-WLS, AF-LAV, AF-WLAV) is
           assigned.
        5. If the estimator does not converge (``success`` is False), a failure message is appended to ``failures``.

    After all three estimations have been performed, the allocation factors of all methods are concatenated into a
    single DataFrame and saved as a CSV file.

    Parameters:
        net_base: Base pandapower grid on which all three state estimation methods are applied.
        failures: List that is extended by a text entry for each estimation that fails to converge.
        num_it: Identifier for this run (e.g. iteration counter) used in all output filenames.
        save_path: Directory where the PICKLE and CSV result files are stored.

    Returns:
        None
    """

    # run AF-WLS estimation
    af_wls_file = os.path.join(save_path, f"af_wls_{num_it}.p")
    if os.path.exists(af_wls_file):
        print(f"file {af_wls_file} already exists")
    else:
        net_af_wls = copy.deepcopy(net_base)
        af_wls = estimate(net_af_wls, algorithm="af-wls", wlav=False)  # , af_target_value=.4, af_std_value=.15
        af_wls["allocation_factors"].index = ["AF-WLS"]  # set index for saving data
        if not af_wls["success"]:
            failures.append(f"AF-WLS, {num_it}, failed")  # add failures information to a list
        to_pickle(net_af_wls, af_wls_file)  # save grid to pickle

    # run LAV estimation
    af_lav_file = os.path.join(save_path, f"af_lav_{num_it}.p")
    if os.path.exists(af_lav_file):
        print(f"file {af_lav_file} already exists")
    else:
        net_af_lav = copy.deepcopy(net_base)
        af_lav = estimate(net_af_lav, algorithm="af-lp", wlav=False, with_ortools=False)
        af_lav["allocation_factors"].index = ["AF-LAV"]  # set index for saving data
        if not af_lav["success"]:
            failures.append(f"AF-LAV, {num_it}, failed")
        to_pickle(net_af_lav, af_lav_file)

    # run AF-WLAV estimation
    af_wlav_file = os.path.join(save_path, f"af_wlav_{num_it}.p")
    if os.path.exists(af_wlav_file):
        print(f"file {af_wlav_file} already exists")
    else:
        net_af_wlav = copy.deepcopy(net_base)
        af_wlav = estimate(net_af_wlav, algorithm="af-lp", wlav=True, with_ortools=False)
        af_wlav["allocation_factors"].index = ["AF-WLAV"]  # set index for saving data
        if not af_wlav["success"]:
            failures.append(f"AF-WLAV, {num_it}, failed")
        to_pickle(net_af_wlav, af_wlav_file)

    # save allocation factors from all estimation solvers in one csv file
    res_af_file = os.path.join(save_path, f"af_df_{num_it}.csv")
    if os.path.exists(res_af_file):
        print(f"file {res_af_file} already exists")
    else:
        res_af_df = pd.concat(
            [af_wls["allocation_factors"], af_wlav["allocation_factors"], af_lav["allocation_factors"]]
        )
        res_af_df.to_csv(res_af_file, index=True)


def create_random_18_bus_grid_random_estimation(
        path: str = ".",
        itr: int = 1000,
        seed: int = 112,
        rv: float = .01,
        rp: float = .03,
        rq: float = .03,
        load_range: tuple[float, float] = (.5, .8),
        com_range: tuple[float, float] = (.3, .6),
        pv_range: tuple[float, float] = (.3, .4),
        wind_range: tuple[float, float] = (.2, .4)
) -> None:
    """
    Every iteration applies random perturbations to the load and generation values used in the power flow calculation
    and, in addition, to the measurement values used for state estimation.

    Parameters:
        path: Place where grid and allocation factors will be saved.
        itr: Number of iterations.
        seed:
            Optional random seed for reproducible simulations. If ``None``, random values are generated for every call.
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power
        load_range:
            Range of scaling factor (uniformly distributed) for load (residential load) :func:`_create_18_bus_grid`
        com_range:
            Range of scaling factor (uniformly distributed) for load (commercial load) :func:`_create_18_bus_grid`
        pv_range:
            Range of scaling factor (uniformly distributed) for sgne (PV generation) :func:`_create_18_bus_grid`
        wind_range:
            Range of scaling factor (uniformly distributed) for wind (wind generation) :func:`_create_18_bus_grid`
    Returns: None
    """

    np.random.seed(seed)
    failures: list = []  # The list contains information about the final status of the state estimation
    for i in tqdm(range(itr)):
        name_str = f"{i:03d}"  # number 1 -> 001, 56 -> 056 etc.
        net18, k = _create_18_bus_grid(
            load_range=load_range, com_range=com_range, pv_range=pv_range, wind_range=wind_range
        )
        _create_measurement_18_bus_grid(net=net18, rv=rv, rp=rp, rq=rq)  #
        _calc_different_se(net18, failures, name_str, path)

    # if failures is not empty the list will save as txt file.
    if not failures:
        print("List is empty, very good")
    else:
        print(f"List is not empty: {failures}")
        with open(os.path.join(path, "failures.txt"), "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(failure + "\n")


def create_random_estimations_simbench(
        net: pandapowerNet,
        path: str = ".",
        itr: int = 1000,
        seed: int = 112,
        seed_pf: int | None = None,
        seed_m: int | None = None,
        rv: float = .01,
        ri: float = .01,
        rp: float = .03,
        rq: float = .03
) -> None:
    """
    Every iteration applies random perturbations to the load and generation values used in the power flow calculation
    and, in addition, to the measurement values used for state estimation.

    Parameters:
        net: Power grid with powerflow calculation results and without measurements.
        path: Place where grid and allocation factors will be saved.
        itr: Number of iterations.
        seed:
            Optional random seed for reproducible simulations. If ``None``, random values are generated for every call.
        seed_pf:
            Parameter for :func:`_create_simbench_mc_case`. Attention if None, no seed will be used different to normal use
            case.
        seed_m:
            Parameter for :func:`_add_measurements_af`. Attention if None, no seed will be used different to normal use
            case.
        rv: standard deviation to apply a multiplicative perturbation to quantities for voltage
        ri: standard deviation to apply a multiplicative perturbation to quantities for current
        rp: standard deviation to apply a multiplicative perturbation to quantities for active power
        rq: standard deviation to apply a multiplicative perturbation to quantities for reactive power
    Returns: None
    """

    np.random.seed(seed)
    failures: list = []  # The list contains information about the final status of the state estimation
    for i in range(itr):
        name_str = f"{i:03d}"  # number 1 -> 001, 56 -> 056 etc.
        k = _create_simbench_mc_case(net, seed_pf)
        _fill_measurement_values_from_powerflow(net, seed_m, rv, ri, rp, rq)
        _calc_different_se(net, failures, name_str, path)

    # if failures is not empty the list will save as txt file.
    if not failures:
        print("List is empty, very good")
    else:
        print(f"List is not empty: {failures}")
        with open(os.path.join(path, "failures.txt"), "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(failure + "\n")


def load_failures(path: str = ".") -> set[tuple[str, int]]:
    """
    Load previously recorded failed solver runs from a text file and return them as a set of ``(solver, iteration)``
    tuples.

    The function reads the file ``failures.txt`` located in ``path``. Each row is expected to contain a solver name,
    an iteration number, and a status field. The status column is ignored when constructing the return value.

    Parameters:
        path: Directory containing the ``failures.txt`` file. Defaults to the current working directory (``"."``).

    Returns:
        Set of unique ``(solver, iteration)`` pairs representing failed solver runs.
    """
    failures = pd.read_csv(
        os.path.join(path, "failures.txt"),
        header=None,
        names=["solver", "iteration", "status"],
        skipinitialspace=True,
        dtype={"solver": str, "iteration": str, "status": str}
    )
    failures["iteration"] = failures["iteration"].astype(int)
    failure_set = set(zip(failures["solver"], failures["iteration"]))
    return failure_set


def evaluation_af(path: str = ".") -> None:
    """
    Evaluate and visualize the distribution of allocation factors across all simulation runs.

    This function aggregates allocation factor results from multiple CSV files (``af_df_*.csv``), excludes failed state
    estimation runs listed in ``failures.txt``, and creates an HTML file containing boxplots and histograms for each
    allocation factor and solver. Creates the file ``allocation_factor_plots.html`` in ``path_eval`` (subdirectory which
    will created if it does not exist), containing for each solver and allocation factor:
        * a boxplot of the allocation factor over all valid iterations, and
        * a histogram of the corresponding value distribution.

    Parameters:
        path:
            Directory where the result files are stored. Defaults to the current directory (" . "). Expected files:
                * ``failures.txt``: text file with three columns (solver, iteration, status), used to identify and skip
                    failed runs.
                * ``af_df_*.csv``: CSV files containing allocation factor results for each iteration. Each file must
                    have solvers as index (e.g. ``AF-WLS``, ``AF-LAV``, ``AF-WLAV``) and allocation factors as columns.

    Raises:
        FileNotFoundError: If no CSV files matching ``af_df_*.csv`` are found in the given directory.
    """
    path_eval = os.path.join(path, "evaluation")
    os.makedirs(path_eval, exist_ok=True)
    html_file = os.path.join(path_eval, "allocation_factor_plots.html")  # check if the file exists
    if os.path.exists(html_file):
        print(f"html file already exists: {html_file}")
        return
    # -------------------------------------------------------------------------
    # Read in failures
    # -------------------------------------------------------------------------
    failure_set = load_failures(path)
    # -------------------------------------------------------------------------
    # Read in CSV file with allocation factors
    # -------------------------------------------------------------------------
    csv_files = sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.startswith("af_df_") and f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError("No af_df_*.csv files found")
    # load general data about solver and allocation factors
    solver_ls = pd.read_csv(csv_files[0], index_col=0).index.tolist()
    af_ls = pd.read_csv(csv_files[0], index_col=0).columns.tolist()
    # -------------------------------------------------------------------------
    # collect data
    # -------------------------------------------------------------------------
    af_total_dc = {solver: pd.DataFrame(columns=af_ls) for solver in solver_ls}
    for i in tqdm(range(len(csv_files))):
        if not os.path.exists(csv_files[i]):
            print(f"Missing: {csv_files[i]}")
            continue
        for solver in solver_ls:
            if (solver, i) in failure_set:  # only data are added, where the state estimation runs successfully
                continue
            df = pd.read_csv(csv_files[i], index_col=0)
            af_total_dc[solver].loc[i, af_ls] = df.loc[solver]
    # -------------------------------------------------------------------------
    # create plot
    # -------------------------------------------------------------------------
    rows_box = len(solver_ls)
    rows_hist = len(solver_ls)
    total_rows = rows_box + rows_hist

    subplot_titles = []
    for solver in solver_ls:
        for af in af_ls:
            subplot_titles.append(f"Boxplot<br>{solver}<br>{af}")

    for solver in solver_ls:
        for af in af_ls:
            subplot_titles.append(f"Histogram<br>{solver}<br>{af}")

    fig = make_subplots(rows=total_rows, cols=len(af_ls), subplot_titles=subplot_titles, vertical_spacing=0.05)
    # -------------------------------------------------------------------------
    # Boxplots
    # -------------------------------------------------------------------------
    for row, solver in enumerate(solver_ls, start=1):  # plotly starts with 1
        df_solver = af_total_dc[solver]
        for col, af in enumerate(af_ls, start=1):
            fig.add_trace(
                go.Box(
                    y=df_solver[af].dropna(),
                    name=f"{solver}-{af}",
                    boxmean=True,
                    showlegend=False
                ),
                row=row,
                col=col
            )

    # -------------------------------------------------------------------------
    # Histogram in same HTML file like boxplot
    # -------------------------------------------------------------------------
    for row_offset, solver in enumerate(solver_ls, start=1):
        df_solver = af_total_dc[solver]
        row = rows_box + row_offset
        for col, af in enumerate(af_ls, start=1):
            fig.add_trace(
                go.Histogram(
                    x=df_solver[af].dropna(),
                    nbinsx=30,  # number of bars -> value range
                    name=f"{solver}-{af}",
                    showlegend=False
                ),
                row=row,
                col=col
            )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    fig.update_layout(
        title="Allocation Factors - Boxplots and Histograms",
        height=2500,
        width=1600,
        margin=dict(t=200)
    )
    fig.write_html(html_file)
    print(f"saved html to: {html_file}")


def build_eval_dataframe(result_dict, variable="vm_pu") -> pd.DataFrame:
    """
    Construct an evaluation DataFrame from simulation result tables.

    The function extracts the specified variable from each result DataFrame in ``result_dict`` and combines the values
    into a single DataFrame. Each row corresponds to one simulation iteration, while each column corresponds to a
    network element (e.g. bus or line).

    Parameters:
        result_dict: Dictionary mapping iteration numbers to pandas DataFrames containing simulation results.
        variable: Name of the result column to extract from each DataFrame. Defaults to ``"vm_pu"``.

    Returns:
        DataFrame containing the selected variable for all iterations. Rows represent iterations and columns represent
        network elements. The index is sorted in ascending order.
    """
    return pd.DataFrame({iteration: df[variable].astype(float) for iteration, df in result_dict.items()}).T.sort_index()


def evaluation_vp(path: str = ".") -> None:
    """
    Evaluate voltage magnitude and branch active power estimation results for all state estimation methods and generate
    interactive HTML visualizations.

    This function loads the result networks of the three allocation-factor-based state estimation methods (AF-WLS,
    AF-WLAV, AF-LAV), excludes failed runs listed in ``failures.txt``, and compares estimated values against the true
    network results.

    For each estimator:
        1. Bus voltage magnitudes (``vm_pu``) and branch active powers (``p_from_mw``) are collected from all successful
            simulation runs.
        2. Mean values of the true and estimated quantities are calculated for each bus and branch.
        3. The root-mean-square error (RMSE) is determined and converted into an expanded uncertainty using a coverage
            factor of ``k = 3``.
        4. Interactive Plotly figures are created showing:
               * estimated and true voltage magnitudes,
               * estimated and true branch active powers,
               * expanded uncertainty of the voltage magnitude estimates.
        5. The figures are saved as separate HTML files.

    The following output files are created and saved to subdirectory estimation:
        * ``voltage_magnitude.html``: Mean estimated and true bus voltage magnitudes with uncertainty bands.
        * ``line_power.html``: Mean estimated and true branch active powers with uncertainty bands.
        * ``voltage_uncertainty.html``: Expanded uncertainty of the voltage magnitude estimates.

    Parameters:
        path:
            Directory containing the result files and where the generated HTML reports will be written. Defaults to the
            current directory ("."). Expected files:

                * ``failures.txt``: Text file with three columns (solver, iteration, status), used to identify and skip
                    failed state estimation runs.
                * ``af_wls_*.p``: Result networks generated with the AF-WLS estimator.
                * ``af_wlav_*.p``: Result networks generated with the AF-WLAV estimator.
                * ``af_lav_*.p``: Result networks generated with the AF-LAV estimator.

    Raises:
        FileNotFoundError: If required PICKLE result files cannot be found.
    """
    path_eval = os.path.join(path, "evaluation")
    os.makedirs(path_eval, exist_ok=True)
    html_vm_file = os.path.join(path_eval, "voltage_magnitude.html")
    html_lp_file = os.path.join(path_eval, "line_power.html")
    html_ve_file = os.path.join(path_eval, "voltage_uncertainty.html")
    if os.path.exists(html_vm_file) and os.path.exists(html_lp_file) and os.path.exists(html_ve_file):
        print(f"html files already exists: {html_vm_file, html_lp_file, html_ve_file}")
        return
    # -------------------------------------------------------------------------
    # Read in failures
    # -------------------------------------------------------------------------
    failure_set = load_failures(path)
    # -------------------------------------------------------------------------
    # Read in bus and line data from pickle
    # -------------------------------------------------------------------------
    solver_ls = ["AF-WLS", "AF-WLAV", "AF-LAV"]

    wls_ls = sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.startswith("af_wls_") and f.endswith(".p")
    )
    wlav_ls = sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.startswith("af_wlav_") and f.endswith(".p")
    )
    lav_ls = sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.startswith("af_lav_") and f.endswith(".p")
    )

    pkl_files_ls = [wls_ls, wlav_ls, lav_ls]

    res_bus_dc = {solver: {} for solver in solver_ls}
    res_bus_est_dc = {solver: {} for solver in solver_ls}

    res_line_dc = {solver: {} for solver in solver_ls}
    res_line_est_dc = {solver: {} for solver in solver_ls}

    for i in tqdm(range(len(wls_ls))):
        for j in range(len(solver_ls)):
            if (solver_ls[j], i) in failure_set:
                print(f"skip failure: solver={solver_ls[j]}, iteration={i}")
                continue
            net_ij = from_pickle(pkl_files_ls[j][i])

            if not hasattr(net_ij, "res_bus"):
                print(f"missing res_bus: solver={solver_ls[j]}, iteration={i}")
                continue

            if not hasattr(net_ij, "res_bus_est"):
                print(f"missing res_bus_est: solver={solver_ls[j]}, iteration={i}")
                continue

            res_bus_dc[solver_ls[j]][i] = net_ij.res_bus.copy()
            res_bus_est_dc[solver_ls[j]][i] = net_ij.res_bus_est.copy()

            res_line_dc[solver_ls[j]][i] = net_ij.res_line.copy()
            res_line_est_dc[solver_ls[j]][i] = net_ij.res_line_est.copy()
        print(f"{i} finished")

    # -------------------------------------------------------------------------
    # Plot Figures separately
    # -------------------------------------------------------------------------
    fig_vm = go.Figure()
    fig_lp = go.Figure()
    fig_ve = go.Figure()

    k = 3.0  # expanded uncertainty factor

    for solver in solver_ls:
        # ---------------------------------------------------------------------
        # Voltage magnitude
        # ---------------------------------------------------------------------
        # DataFrame for bus
        vm_true = build_eval_dataframe(res_bus_dc[solver], "vm_pu").astype(float)
        vm_est = build_eval_dataframe(res_bus_est_dc[solver], "vm_pu").astype(float)

        # calc mean for every bus separate
        vm_true_mean = vm_true.mean(axis=0)
        vm_est_mean = vm_est.mean(axis=0)

        # clac RMSE pro bus
        vm_error = vm_est - vm_true
        vm_rmse = np.sqrt((vm_error ** 2).mean(axis=0))
        # clac expanded uncertainty
        vm_u = k * vm_rmse

        x_bus = np.arange(len(vm_est_mean))

        # plot uncertainty band
        fig_vm.add_trace(
            go.Scatter(
                x=np.concatenate([x_bus, x_bus[::-1]]),
                y=np.concatenate([
                    (vm_est_mean + vm_u).to_numpy(),
                    (vm_est_mean - vm_u).to_numpy()[::-1]
                ]),
                fill="toself",
                line=dict(width=0),
                opacity=0.2,
                name=f"{solver} uncertainty"
            )
        )
        # plot mean value from estimated voltage magnitude for every bus
        fig_vm.add_trace(
            go.Scatter(
                x=x_bus,
                y=vm_est_mean.to_numpy(),
                mode="lines",
                name=f"{solver} estimated V"
            )
        )
        # plot mean value from calculated voltage magnitude (powerflow) for every bus
        fig_vm.add_trace(
            go.Scatter(
                x=x_bus,
                y=vm_true_mean.to_numpy(),
                mode="lines",
                line=dict(dash="dash"),
                name=f"{solver} true V"
            )
        )

        # ---------------------------------------------------------------------
        # Branch active power same like voltage magnitude above
        # ---------------------------------------------------------------------
        p_true = build_eval_dataframe(res_line_dc[solver], "p_from_mw").astype(float)
        p_est = build_eval_dataframe(res_line_est_dc[solver], "p_from_mw").astype(float)

        p_true_mean = p_true.mean(axis=0)
        p_est_mean = p_est.mean(axis=0)

        p_error = p_est - p_true
        p_rmse = np.sqrt((p_error ** 2).mean(axis=0))
        p_u = k * p_rmse

        x_line = np.arange(len(p_est_mean))

        fig_lp.add_trace(
            go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([
                    (p_est_mean + p_u).to_numpy(),
                    (p_est_mean - p_u).to_numpy()[::-1]
                ]),
                fill="toself",
                line=dict(width=0),
                opacity=0.2,
                name=f"{solver} uncertainty"
            )
        )
        fig_lp.add_trace(
            go.Scatter(
                x=x_line,
                y=p_est_mean.to_numpy(),
                mode="lines",
                name=f"{solver} estimated P"
            )
        )
        fig_lp.add_trace(
            go.Scatter(
                x=x_line,
                y=p_true_mean.to_numpy(),
                mode="lines",
                line=dict(dash="dash"),
                name=f"{solver} true P"
            )
        )

        # ---------------------------------------------------------------------
        # Voltage uncertainty
        # ---------------------------------------------------------------------
        fig_ve.add_trace(
            go.Scatter(
                x=x_bus,
                y=(100 * vm_u).to_numpy(),
                mode="lines",
                name=f"{solver} V uncertainty [%]"
            )
        )

    fig_vm.update_layout(
        title="Voltage Magnitude Estimation",
        xaxis_title="Bus",
        yaxis_title="Voltage magnitude [p.u.]",
        height=700,
        width=1400
    )

    fig_lp.update_layout(
        title="Branch Active Power Estimation",
        xaxis_title="Branch",
        yaxis_title="Active power [MW]",
        height=700,
        width=1400
    )

    fig_ve.update_layout(
        title="Voltage Magnitude Expanded Uncertainty",
        xaxis_title="Bus",
        yaxis_title="Expanded uncertainty [%]",
        height=700,
        width=1400
    )

    fig_vm.write_html(html_vm_file)
    fig_lp.write_html(html_lp_file)
    fig_ve.write_html(html_ve_file)

    print(f"saved html to: {html_vm_file}")
    print(f"saved html to: {html_lp_file}")
    print(f"saved html to: {html_ve_file}")


if __name__ == "__main__":
    time_start = time.perf_counter()
    load_dotenv()

    mv_b: bool = False
    ieee14_b: bool = False
    ieee30_b: bool = False
    bus18_b: bool = True
    simbench_b: bool = False

    if mv_b:
        # rv=.01, rp=.03, rq=.03
        net_mv = pn.mv_oberrhein()
        runpp(net_mv)
        _add_measurements_af(net_mv, 112,15, .0, .0, .0)

    if ieee14_b:
        net14 = pn.case14()
        runpp(net14)
        _add_measurements_af(net14, 112, 2, .0, .0, .0)

    if ieee30_b:
        net30 = pn.case30()
        runpp(net30)
        _add_measurements_af(net30, 112, 5, .0, .0, .0)

    if bus18_b:
        s_path = os.path.join(str(os.getenv("PATH_18BUS")), "008")
        os.makedirs(s_path, exist_ok=True)
        create_random_18_bus_grid_random_estimation(
            s_path,
            1000,
            112,
            .01,
            .01,
            .01,
            (.5,.8),
            (.3,.6),
            (.1, .9),
            (.1,.9)
        )
        evaluation_af(s_path)
        evaluation_vp(s_path)

    if simbench_b:
        sb_path = str(os.getenv("PATH_SB"))
        sb_grid_ls = ["1-MV-semiurb--0-sw", "1-MV-urban--0-sw", "1-MV-comm--0-sw"]
        for sb_grid in sb_grid_ls:
            s_path = os.path.join(sb_path, sb_grid)
            os.makedirs(s_path, exist_ok=True)

            net_sb = sb.get_simbench_net(sb_grid)
            net_sb.load["type"] = net_sb.load["type"].fillna("residential")
            # print(
            #     f"Grid: {sb_grid}\n"
            #     f"Allocation Factors Load: {net_sb.load["type"].unique()}\n"
            #     f"Allocation Factors Generator: {net_sb.gen["type"].unique()}\n"
            #     f"Allocation Factors Static Generator{net_sb.sgen["type"].unique()}\n"
            #     f"Number of buses: {len(net_sb.bus)}\n"
            # )
            create_random_estimations_simbench(
                net_sb, s_path, 1000, 112, None, None, .01, .01, .01, .01
            )
            evaluation_af(s_path)
            evaluation_vp(s_path)
            # net_sb.measurement.drop(net_sb.measurement.index, inplace=True)
            print(f"finished: {sb_grid}")
    runtime = time.perf_counter() - time_start
    print(f"calculated in: {timedelta(seconds=runtime)}")
    print(f"you shall not pass")
