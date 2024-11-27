# Copyright (c) 2024-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sympy as sp
import numpy as np
from math import comb
from typing import Optional
from SumOfSquares import SOSProblem, poly_variable
from src.utils import timeout


@timeout(300)
def SOS_checker(poly_sp, output_matrix: bool = False, solver: Optional[str] = None):
    """
    Takes a polynomial sympy expression and checks whether it is a sum of squares
    INPUTS:
    - polynome in sympy
    - output_matrix: bool. If True returns a semidefinite matrix representing the Sum of Squares
    If False only returns
    OUPUTS:
    - Result of the optimization (True, False or None)
    - Optional: Output matrix
    """
    poly_sp = sp.simplify(poly_sp)
    prob = SOSProblem()
    vars = list(poly_sp.free_symbols)
    try:
        c1 = prob.add_sos_constraint(poly_sp, vars)
        if solver is None:
            solver = "cvxopt"
        prob.solve(solver=solver)
        if prob.status in ["optimal", "feasible"]:
            if output_matrix:
                Qval = prob.get_variable("_Q1")
                return (True, Qval)
            return True
        elif prob.status == "infeasible":
            return False
        else:
            return None
    except Exception as e:
        print("infeasibility in SOS checker", e.args[0])
        if e.args[0] == 3 or (isinstance(e.args[0], str) and e.args[0].startswith("Polynomial degree must be even!")):
            return False
        else:
            print("optimisation error", e.args)
            return None


@timeout(2400)
def findlyap(
    system_sp,
    degree,
    options=None,
    rounded: int = -1,
    solver: Optional[str] = None,
    proper: Optional[bool] = False,
    env_rng=None,
    max_int: Optional[int] = None,
):
    """
    Find a Lyapunov function for a system
    system_sp: system in sympy
    degree: maximal degree of the Lyapunov function
    """
    assert rounded == -1 or isinstance(rounded, int)

    if options is None:
        options = {"solver": "cvxopt"}  # Default solver
    vars = []
    for i in range(len(system_sp)):
        vars.append(sp.S(f"x{i}"))
    if degree < 2:
        raise ValueError("Degree must be at least 2")
    if degree % 2 != 0:
        raise ValueError("Degree must be even")
    if len(vars) != len(system_sp):
        raise ValueError("Length of vars and f must be the same")

    # Create an SOS problem
    prob = SOSProblem()

    # Normalize the system
    sys_sp = global_scaling_normalization(system_sp)

    # Declare a new polynomial variable for the Lyapunov function
    V_list = []
    for i in range(2, degree + 1):
        # For each degree below the maximal degree of the Lyapunov function, add all monomials
        V_list.append(poly_variable(f"V{i}", vars, i, hom=True))

    # Create the Lyapunov function with symbolic coefficients to be found
    V = sum(V_list)

    # If want a proper Lyapunov function (and not a barrier Lyapunov function)
    if proper:
        assert env_rng is not None
        assert max_int is not None
        if rounded > 1:
            epsilon = 10 ** (-rounded)
        else:
            epsilon = 1
        degs = [2 * env_rng.randint(1, 1 + np.floor(degree / 2)) for _ in range(len(vars))]
        coeffs = [env_rng.randint(1, max_int + 1) for _ in range(len(vars))]
        V0 = sum([sp.S(epsilon * coeffs[i] * vars[i] ** degs[i]) for i in range(len(degs))])
    else:
        V0 = sp.S(0)
    l1 = V - V0

    # Semi-positive definiteness condition of V
    c0 = prob.add_sos_constraint(l1, vars)

    # Semi-positive definiteness condition on derivative
    dV_dt = sum([-sp.diff(V, var) * f_i for var, f_i in zip(vars, sys_sp)])
    try:
        c1 = prob.add_sos_constraint(dV_dt, vars)
    except AssertionError as e:
        if e.args[0].startswith("Polynomial degree must be even"):
            raise RuntimeError("Unfeasible degree") from e
        else:
            raise AssertionError(e.args[0]) from e

    # Solve the SOS problem
    if solver is None:
        solver = "cvxopt"
    try:
        prob.solve(solver=solver)
    except Exception as e:
        return None
    if prob.status not in ["optimal", "feasible"]:
        print("No Lyapunov function has been found for this system.")
        return None
    else:
        V = sp.S(V)
        coeffs = []
        n_vars = len(vars)

        # Retrieve the coefficients
        numbers_monomial = []
        for i in range(2, degree + 1):
            this_number_monomial = comb(n_vars + i - 1, i)
            numbers_monomial.append(this_number_monomial)
            for j in range(this_number_monomial):
                coeffs.append(prob.get_valued_variable(f"V{i}_{j}"))
        assert len(coeffs) == sum(numbers_monomial)  # Safeguard
        # Prepare the scaling
        coeff_max = np.max(coeffs)
        coeff_norm = coeffs.copy() / coeff_max

        # Substitute in the Lyapunov function
        coeff_number = 0
        for i, this_number_monomial in enumerate(numbers_monomial):
            for j in range(this_number_monomial):
                this_coeff = coeff_norm[coeff_number]
                if rounded > 0:
                    new_coeff_norm = round(this_coeff, rounded)
                else:
                    new_coeff_norm = this_coeff
                V = V.subs(f"V{i+2}_{j}", new_coeff_norm)
                coeff_number += 1
        assert coeff_number == len(coeff_norm)

        dV_dt = sum([sp.diff(V, var) * f_i for var, f_i in zip(vars, sys_sp)])
        if rounded > 0:
            c1 = SOS_checker(V)
            if not c1:
                raise RuntimeError("Rounding does not work for this function at c1, False")
            elif c1 is None:
                raise RuntimeError("Rounding might not work for this function at c1, None")
            c2 = SOS_checker(-dV_dt)
            if not c2:
                raise RuntimeError("Rounding does not work for this function at c2, False")
            if c2 is None:
                raise RuntimeError("Rounding might not work for this function at c2, None")
        return V, -dV_dt


def global_scaling_normalization(system_sp):
    """Normalize the system using a global scaling factor"""
    max_coeff = max(sp.Abs(sp.LC(eq)) for eq in system_sp if eq != 0)
    if max_coeff == 0:
        return system_sp.copy()
    return [eq / max_coeff for eq in system_sp]
