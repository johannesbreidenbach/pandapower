# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from numpy.typing import NDArray

from typing import Literal, Union

from scipy.sparse import issparse, csr_matrix, vstack, hstack, eye, spmatrix
from scipy.optimize import linprog

from ortools.linear_solver import pywraplp

from pandapower.estimation.ppc_conversion import ExtendedPPCI
from pandapower.estimation.algorithm.base import BaseAlgorithm
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra

import logging
std_logger = logging.getLogger(__name__)
std_logger.setLevel(logging.DEBUG)


LinprogMethod = Literal["highs", "highs-ds", "highs-ipm"]
_ALLOWED_LINPROG_METHODS = {"highs", "highs-ds", "highs-ipm"}
ArrayLike = Union[NDArray[np.float64], spmatrix]

class LPAlgorithm(BaseAlgorithm):
    def __init__(self, tolerance: float, maximum_iterations: int, logger: logging.Logger = std_logger) -> None:
        """
        The algorithm solves a (weighted) 'Least Absolute Value (LAV)' optimization problem to estimate the system
        state vector from possibly bad or noisy measurements.

        Parameters:
            tolerance:
                Convergence threshold for the state update ‖ΔE‖_∞. The iterative process stops once the maximum
                absolute update is below this value.
            maximum_iterations:
                Maximum number of iterations allowed before the algorithm is considered not converged.
            logger:
                Logger instance used for diagnostic and error messages.
        """
        # Initialize base algorithm
        super(LPAlgorithm, self).__init__(tolerance, maximum_iterations, logger)

        # store results for diagnostics
        self.r = None  # residual z-h(x)
        self.H = None  # Jacobian matrix
        self.hx = None  # calculated measurements h(x)
        self.obj_func = None  # objective function J(x)

    def estimate(
            self,
            eppci: ExtendedPPCI,
            debug_mode=False,
            linprog_method: LinprogMethod = 'highs',
            wlav: bool = False,
            with_ortools: bool = True,
            **kwargs
    ) -> ExtendedPPCI | bool:
        """
        Perform power system state estimation using the (W)LAV formulation.

        The method solves an iterative linear programming problem based on the linearized measurement model
            r = z - h(x),  r_new ≈ r - H ΔE,

        where the objective is to minimize the (weighted) 1-norm of the residuals
            min Σ_i w_i |r_i|.

        In each iteration, a linear program is solved via ``scipy.optimize.linprog`` to obtain the state update ΔE.
        The state vector ``E`` inside ``eppci`` is updated in-place.

        Parameters:
            eppci:
                Central data container (ExtendedPPCI) containing the network model, measurements, current state vector
                ``E``, measurement vector ``z``, and (optionally) covariance information ``r_cov`` for WLAV.
            debug_mode:
                If ``True``, additional diagnostic information is logged, including the current state update norm and
                the current LAV objective value.
            linprog_method:
                Method name passed to ``scipy.optimize.linprog`` (e.g. ``"highs"``).
            wlav:
                If ``True``, perform weighted LAV, where the weights are computed as ``1 / sigma`` with
                ``sigma = max(r_cov, 1e-5)``. If ``False``, all measurements are weighted equally.
            with_ortools:
                If ``True``, use the OR-Tools solver (https://github.com/google/or-tools)

        Keyword Arguments:
            **kwargs:
                Currently unused. Present for API compatibility and possible future extensions.

        Returns:
            ExtendedPPCI | bool:
                The updated data container with the estimated state variables if the optimization is successful.
                Additionally, on success the following attributes are populated for diagnostics:

                * ``self.r``: final residual vector ``z - h(E)``
                * ``self.H``: final Jacobian matrix at the estimated state
                * ``self.hx``: final calculated measurements ``h(E)``
                * ``self.obj_func``: final LAV objective value
                * ``self.iterations``: number of iterations performed

                Returns ``False`` if the optimization fails or an exception occurs.
        """
        # initialize eppci and check the observability
        self.initialize(eppci)

        # matrix calculation object for the state estimation parameter
        sem = BaseAlgebra(eppci)
        current_error, cur_it = 100., 0
        E = eppci.E

        while current_error > self.tolerance and cur_it < self.max_iterations:
            try:
                # residual r=z-h(x)
                r = LPAlgorithm._to_dense_auto(sem.create_rx(E))
                # create Jacobian matrix convert to csr -> zeros not save -> better for lager grids -> less RAM
                H_raw = sem.create_hx_jacobian(E)  # create jacobian matrix from data set
                H = H_raw.tocsr() if issparse(H_raw) else csr_matrix(H_raw)  #
                # m number of measurements -> z element R^{m}, n number of state variable
                m, n = H.shape

                if wlav:
                    sigma = np.maximum(LPAlgorithm._to_dense_auto(eppci.r_cov), 1e-5)
                    weights = 1.0 / sigma
                else:
                    weights = np.ones(m)

                if with_ortools:
                    d_E, obj_value = self._solve_lav_ortools(H, r, weights)
                else:
                    d_E, obj_value = self._solve_lav_scipy(H, r, weights, linprog_method)

                current_error = float(np.max(np.abs(d_E)))

                # Optional step limiting:
                # This factor was derived from the 50Hz project, where a flat start in large-scale grids caused
                # excessively large state updates (dE). Limiting the step to 0.35 helps prevent those large jumps during
                # the first iterations.
                if current_error > 0.35:  # 50Hz project heuristic to limit excessive state updates
                    d_E = d_E * 0.35 / current_error

                # Update state vector
                E += d_E
                eppci.update_E(E)

                if debug_mode:
                    self.obj_func = obj_value
                    self.logger.debug(f'Current delta_x: {current_error:.7f}')
                    self.logger.debug(f'Current LAV objective value: {obj_value:.7f}')

                cur_it += 1

            except Exception as err:
                self.logger.error(f'A problem appeared while running LAV estimation: {err}')
                return False

        self.check_result(current_error, cur_it)
        self.iterations = cur_it

        if debug_mode:
            print(f'number of required iterations: {cur_it} of {self.max_iterations}')
            print(f'current_error = {current_error}, threshold = {self.tolerance}')

        if self.successful:
            self.r = np.asarray(sem.create_rx(E)).reshape(-1)
            self.H = np.asarray(sem.create_hx_jacobian(E))
            self.hx = sem.create_hx(eppci.E)

        return eppci

    @staticmethod
    def _to_dense_auto(x: ArrayLike) -> np.ndarray:
        """
        Convert input to a dense NumPy array and automatically adjust its shape.

        Sparse matrices are converted using ``toarray()``. Other array-like inputs
        are converted using ``numpy.asarray``. Vector-like arrays with shape ``(n,)``,
        ``(n, 1)``, or ``(1, n)`` are returned as one-dimensional arrays. Matrix-like
        arrays are returned unchanged.

        Parameters:
            x: Input data, such as a NumPy array, list, or SciPy sparse matrix.

        Returns:
            Dense NumPy array. Vector-like inputs are flattened to shape ``(n,)``.
        """
        arr = x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
        arr = np.asarray(arr, dtype=np.float64)

        if arr.ndim == 0:
            return arr.reshape(1)

        if arr.ndim == 1:
            return arr

        if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            return arr.reshape(-1)

        return arr

    @staticmethod
    def _solve_lav_scipy(
            H: csr_matrix,
            r: NDArray[np.float64],
            weights: NDArray[np.float64],
            linprog_method: LinprogMethod
    ) -> tuple[NDArray[np.float64], float]:
        """
        Solve the LAV/WLAV linear programming problem using ``scipy.optimize.linprog``.

        The optimization problem is formulated as: min Σ_i w_i * u_i
        subject to: -u <= r - H dE <= u
        where:
            * ``dE`` are the state updates
            * ``u`` are auxiliary nonnegative variables representing the absolute residuals
            * ``r`` is the current residual vector
            * ``H`` is the Jacobian matrix

        The LP variable vector is defined as: y = [dE_1, ..., dE_n, u_1, ..., u_m]
        The inequality constraints are rewritten into standard LP form:
            H dE - u <= r
           -H dE - u <= -r

        Sparse matrices are used throughout the formulation to improve performance and memory efficiency for large-scale
        power systems.

        Parameters:
            H: Sparse Jacobian matrix with shape ``(m, n)``.
            r: Residual vector ``z - h(x)`` with shape ``(m,)``.
            weights: Weight vector for WLAV. For standard LAV, this is typically ``np.ones(m)``.
            linprog_method: Method passed to ``scipy.optimize.linprog`` (e.g. ``"highs"``).

        Returns:
            tuple[NDArray[np.float64], float]:
                Tuple containing:
                * ``d_E``:
                  State update vector with shape ``(n,)``.
                * ``objective_value``:
                  Final value of the LP objective function.

        Raises:
            np.linalg.LinAlgError:
                If the LP optimization fails or no feasible solution is found.
        """
        # m number of measurements -> z element R^{m}, n number of state variable
        m, n = H.shape

        # We solve (Abur - Power system state estimation: theory and implementation, 2004):
        # min sum(abs(r_i)) -> abs non-linear -> -u_i <= r <= u_i and r_i = z_i - h_i(x)
        # r_i -> risidual of iteration i (differenc between measurement values and model function)
        # h_i(x) is nonlinear, we like to get the best state x with an iterative linear approach
        # r_i^{new} = z_i - h_i(x + dx) = z_i - (h_i(x) + H dx) = r_i - H_i dx
        # -> -u_i <= r_i - H_i dx <= u_i  # Attention pandapower E = x

        # min sum(u_i)
        # subject to:
        # -u_i <= r_i - H_i * dE <= u_i

        # linprog (scipy) -> -u_i <= r_i - H_i * dE <= u_i -> 2 constraints among themselves
        # Variable vector:
        # y = [dE_1, ..., dE_n, u_1, ..., u_m]
        # In the unweighted case, c_i(dE_i) = 0 and c_i(E) = 1

        c = np.r_[np.zeros(n), weights]

        # Constraint:
        # r - H dE <= u
        # -H dE - u <= -r
        I = eye(m, format="csr")  # Sparse identity instead of dense np.eye(m)
        A1 = hstack([-H, -I], format="csr")
        b1 = -r

        # Constraint:
        # -(r - H dE) <= u
        # H dE - u <= r
        A2 = hstack([H, -I], format="csr")
        b2 = r

        # ub stands for upper bound -> A_ub * y <= b_ub
        A_ub = vstack([A1, A2], format="csr")
        b_ub = np.r_[b1, b2]

        # u_i >= 0, dE free
        bounds = [(None, None)] * n + [(0, None)] * m

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method=linprog_method
        )

        if not result.success:
            raise np.linalg.LinAlgError(result.message)
        if result.x is None:
            raise np.linalg.LinAlgError("SciPy LAV optimization returned no solution vector.")
        if result.fun is None:
            raise np.linalg.LinAlgError("SciPy LAV optimization returned no objective value.")

        # Extract state update
        d_E: NDArray[np.float64] = np.asarray(result.x[:n], dtype=np.float64)
        objective_value: float = float(result.fun)

        return d_E, objective_value

    @staticmethod
    def _solve_lav_ortools(
            H: csr_matrix,
            r: NDArray[np.float64],
            weights: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        """
        Solve the LAV/WLAV linear programming problem using OR-Tools SCIP.

        The optimization problem is formulated as: min Σ_i w_i * u_i
        subject to: -u <= r - H dE <= u
        where:
            * ``dE`` are the state updates
            * ``u`` are auxiliary nonnegative variables representing the absolute residuals
            * ``r`` is the current residual vector
            * ``H`` is the sparse Jacobian matrix
        The LP variable vector is: y = [dE_1, ..., dE_n, u_1, ..., u_m]
        The inequality constraints are implemented row-wise using sparse CSR iteration:
            H_i dE - u_i <= r_i
           -H_i dE - u_i <= -r_i
        Only nonzero Jacobian entries are processed, which significantly improves performance for large-scale sparse
        systems.

        Parameters:
            H: Sparse Jacobian matrix in CSR format with shape ``(m, n)``.
            r: Residual vector ``z - h(x)`` with shape ``(m,)``.
            weights: Weight vector for WLAV. For standard LAV, this is typically ``np.ones(m)``.

        Returns:
            tuple[NDArray[np.float64], float]:
                Tuple containing:
                * ``d_E``:
                  State update vector with shape ``(n,)``.
                * ``objective_value``:
                  Final value of the LP objective function.

        Raises:
            np.linalg.LinAlgError: If the solver is unavailable or if no feasible solution is found.
        """
        # m number of measurements -> z element R^{m}, n number of state variable
        m, n = H.shape

        # Ignore very small coefficients to improve numerical stability
        error_margin = 1e-10

        # Create SCIP solver
        solver = pywraplp.Solver.CreateSolver("SCIP")

        if solver is None:
            raise np.linalg.LinAlgError("OR-Tools SCIP solver is not available.")

        infinity = solver.infinity()

        # State update variables dE:
        # free variables (-inf <= dE <= inf)
        dE = [
            solver.NumVar(-infinity, infinity, f"dE_{j}")
            for j in range(n)
        ]

        # Auxiliary variables u:
        # u_i >= 0
        u = [
            solver.NumVar(0.0, infinity, f"u_{i}")
            for i in range(m)
        ]

        # Add inequality constraints row-wise using sparse CSR access
        for i in range(m):
            # CSR row access
            start, end = H.indptr[i], H.indptr[i + 1]

            cols = H.indices[start:end]
            vals = H.data[start:end]

            # Build sparse linear expression:
            # Σ_j H_ij * dE_j
            h_expr = solver.Sum(
                vals[k] * dE[cols[k]]
                for k in range(len(vals))
                if abs(vals[k]) > error_margin
            )

            # Constraint:
            # H_i dE - u_i <= r_i
            solver.Add(h_expr - u[i] <= float(r[i]))

            # Constraint:
            # -H_i dE - u_i <= -r_i
            solver.Add(-h_expr - u[i] <= float(-r[i]))

        # Objective function:
        # min Σ_i weights_i * u_i
        objective = solver.Sum(
            float(weights[i]) * u[i]
            for i in range(m)
        )

        solver.Minimize(objective)

        # Solve LP
        status = solver.Solve()

        if status not in (
                pywraplp.Solver.OPTIMAL,
                pywraplp.Solver.FEASIBLE
        ):
            raise np.linalg.LinAlgError(
                "OR-Tools LAV optimization failed."
            )

        # Extract state update vector
        d_E: NDArray[np.float64] = np.asarray([var.solution_value() for var in dE], dtype=np.float64)
        objective_value = solver.Objective().Value()
        return d_E, float(objective_value)