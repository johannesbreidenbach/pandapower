# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from scipy.optimize import minimize

from pandapower.estimation.algorithm.base import BaseAlgorithm
from pandapower.estimation.algorithm.estimator import BaseEstimatorOpt, get_estimator
from pandapower.estimation.ppc_conversion import ExtendedPPCI

from typing import Literal

DEFAULT_OPT_METHOD = "Newton-CG"

ESTIMATOR_OPT_TYPES = Literal["wls", "lav", "qc", "ql"]

# DEFAULT_OPT_METHOD = "TNC"
# DEFAULT_OPT_METHOD = "SLSQP"
# DEFAULT_OPT_METHOD = 'L-BFGS-B'


class OptAlgorithm(BaseAlgorithm):
    def estimate(
            self,
            eppci: ExtendedPPCI,
            estimator: ESTIMATOR_OPT_TYPES = "wls",
            verbose: bool = True,
            **kwargs
    ) -> ExtendedPPCI:
        """
        Perform power system state estimation via a general nonlinear optimization approach.

        This method formulates the state estimation problem directly in terms of the nonlinear measurement model
        :math:`h(E)` and solves it with :func:`scipy.optimize.minimize`. The objective function (e.g. WLS cost, robust
        cost, or a nonlinear LAV-type cost) and its Jacobian are provided by the selected estimator.

        In contrast to the LP-based (W)LAV algorithms, which solve a *linear* programming problem on a linearized
        measurement model in each iteration, this optimization-based approach treats the full problem as a (typically)
        nonlinear optimization task in the state variables :math:`E`.

        Parameters:
            eppci: Extended power system case input (ExtendedPPCI) containing the current state vector, network model
                and measurement data.
            estimator: Estimator type used to build the optimization-based cost function and its Jacobian
                (e.g. ``"wls"``). The concrete estimator class is obtained via :func:`get_estimator` with
                :class:`BaseEstimatorOpt` as base.
            verbose:  If ``True``, pass ``disp=True`` to :func:`scipy.optimize.minimize``so that optimization progress
                messages are printed. If ``False``, the optimizer runs silently (``disp=False``).
            **kwargs:
                Additional keyword arguments forwarded to the estimator returned by :func:`get_estimator`. May also
                include:

                     - ``opt_method`` (str): Optimization method name passed to :func:`scipy.optimize.minimize` (e.g.
                       ``"trust-constr"``, ``"L-BFGS-B"``). If omitted, ``DEFAULT_OPT_METHOD`` is used.

        Raises:
            Exception: If the optimization fails (``res.success`` is ``False``), an exception is raised with the message
            ``"Optimization failed! State Estimation not successful!"``.

        Returns:
            ExtendedPPCI:
                The updated ExtendedPPCI container with the estimated state vector  ``E`` if the optimization terminates
                successfully.
        """
        opt_method = DEFAULT_OPT_METHOD if "opt_method" not in kwargs else kwargs["opt_method"]

        # matrix calculation object
        estm = get_estimator(BaseEstimatorOpt, estimator)(eppci, **kwargs)

        jac = estm.create_cost_jacobian
        res = minimize(  # type: ignore[call-overload]
            estm.cost_function,
            x0=eppci.E,
            method=opt_method, jac=jac, tol=self.tolerance,
            options={"disp": verbose}
        )

        self.successful = res.success
        if self.successful:
            E = res.x
            eppci.update_E(E)
            return eppci
        else:
            raise Exception("Optimization failed! State Estimation not successful!")
