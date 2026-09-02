import simbench as sb
import numpy as np

from pandapower.estimation import estimate
from pandapower.test.estimation.af_wlav import (
    create_random_estimations_simbench,
    _create_simbench_mc_case,
    _fill_measurement_values_from_powerflow
)

if __name__ == "__main__":

    sb_grid_name = "1-MV-rural--0-sw"  # "1-MV-rural--0-sw" "1-MV-urban--0-sw" "1-MV-comm--0-sw"

    net_sb = sb.get_simbench_net(sb_grid_name)

    p_loads = net_sb.load["p_mw"].abs()
    if sb_grid_name == "1-MV-rural--0-sw":
        net_sb.load["type"] = np.select(  # "1-MV-rural--0-sw"  -> wls algorithm does not work so good
            [
                p_loads <= 0.3,
                p_loads > 0.3
            ],
            [
                "residential",
                "commercial"
            ],
            default="unknown"
        )
    if sb_grid_name == "1-MV-urban--0-sw":
        net_sb.load["type"] = np.select(  # "1-MV-urban--0-sw"
            [
                p_loads <= 0.35,
                p_loads > 0.35
            ],
            [
                "residential",
                "commercial"
            ],
            default="unknown"
        )
    if sb_grid_name == "1-MV-comm--0-sw":
        net_sb.load["type"] = np.select(  # "1-MV-comm--0-sw"
            [
                p_loads <= 0.70,
                p_loads > 0.70
            ],
            [
                "residential",
                "commercial"
            ],
            default="unknown"
        )

    np.random.seed(112)
    # k = _create_simbench_mc_case(net_sb, None)  # create different scaling factors for load and sgen
    _fill_measurement_values_from_powerflow(net_sb, None, .01, .01, .01, .01)

    res_wlav = estimate(
        net_sb,
        algorithm="af-lp",
        wlav=True,
        with_ortools=False,
        with_af_constraints=False,
        linprog_method="highs-ipm",
        maximum_iterations=100
    )
    res_wls = estimate(net_sb, algorithm="af-wls", maximum_iterations=200)
    res_lav = af_lav = estimate(
        net_sb,
        algorithm="af-lp",
        wlav=False,
        with_ortools=False,
        with_af_constraints=True,
        linprog_method="highs-ipm",
        maximum_iterations=100
    )

    print(f"end")
