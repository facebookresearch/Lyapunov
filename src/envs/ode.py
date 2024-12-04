# Copyright (c) 2024-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import io
import sys
from typing import List, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sympy as sp
import scipy.optimize as opt

from sympy.core.cache import clear_cache

from ..utils import bool_flag
from ..SOS_utils import SOS_checker, findlyap
from ..utils import timeout, MyTimeoutError, DomainError


CLEAR_SYMPY_CACHE_FREQ = 10000

SPECIAL_WORDS = ["<s>", "</s>", "<pad>", "(", ")"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]

EPSILON = 1e-14

logger = getLogger()


def ln1(arg):
    """
    The function x -> ln(1+x).
    """
    return sp.ln(1 + arg)


def sqrt1(arg):
    """
    The function x -> sqrt(1+x)
    """
    return sp.sqrt(1 + arg)


def cos1(arg):
    """
    The function x -> 1+cos(x)
    """
    return 1 + sp.cos(arg)


def sin1(arg):
    """
    The function x -> 1+sin(x)
    """
    return 1 + sp.sin(arg)


class UnknownSymPyOperator(Exception):
    pass


def has_inf_nan(*args):
    """
    Detect whether some SymPy expressions contain a NaN / Infinity symbol.
    """
    for f in args:
        if f.has(sp.nan) or f.has(sp.oo) or f.has(-sp.oo) or f.has(sp.zoo):
            return True
    return False


def last_index(x, bal):
    try:
        p1 = x[::-1].index(bal)
        return len(x) - 1 - p1
    except:
        return len(x)


def simplify(f, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0

    @timeout(seconds)
    def _simplify(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                logger.warning(f"Detected Dummy symbol when simplifying {f} to {f2}")
                return f
            else:
                return f2
        except MyTimeoutError:
            return f
        except Exception as e:
            logger.warning(f"{type(e).__name__} exception when simplifying {f}")
            return f

    return _simplify(f)


def expr_to_fun(x, f, point):
    """
    Transforms a sympy expression into a callable function that returns a float for optimization.
    To be deprecated in the future using sympy lambdify that is much faster
    """
    for i in range(len(point)):
        v = "x" + str(i)
        f = f.subs(v, x[i])
    f = f.evalf()
    f = min(f, 1e20)
    f = max(f, -1e20)
    f = float(f)
    return f


def test_V_positive(V, point, domain: Optional[List["Node"]] = None, debug=False):
    """
    Take an object, a sympy expression, and a point and test the positivity of V.
    To be deprecated in the future using sympy lambdify that is much faster
    """
    n_vars = len(point)
    # Compute the gradient for the minimization
    x = sp.symbols(f"x0:{n_vars}")
    grad_V = [sp.diff(V, x[i]) for i in range(n_vars)]
    grad_V_num = [sp.lambdify(x, grad_V[i], "numpy") for i in range(n_vars)]

    def grad_V_fun(x):
        return [grad_V_num[i](*x) for i in range(n_vars)]

    if domain is None or len(domain) == 0:
        ### Function is defined on the whole \mathbb{R}^{n}
        if debug:
            print("no cons_sympy")
            print("function to minimize", sp.simplify(sp.expand(V)))
        bounds = []
        for _ in range(len(point)):
            bounds.append((-10, 10))
        y = opt.shgo(
            expr_to_fun,
            bounds,
            args=(V, point),
            sampling_method="simplicial",
            minimizer_kwargs={"options": {"maxiter": 3000, "disp": False}, "jac": grad_V_fun},
        )
        if debug:
            print(y)
        if y.fun < -5e19:
            return 0
        if not y.success:
            return -1
        c = 1 if y.fun > -EPSILON else 0
    else:
        ### Function is defined on a domain only
        cons_sympy = []
        cons_sympy_neq = []
        constraint_tab = []
        constraint_tab_neq = []
        for el_dom in domain:
            if el_dom.value == "!=":
                assert el_dom.children[1].eq(Node(0)), el_dom
                new_cons = f"1/({el_dom.children[0].infix()})"
                cons_sympy_neq.append(sp.S(new_cons))
                constraint_tab_neq.append(lambda x: expr_to_fun(x, cons_sympy_neq[-1], point))
            else:
                assert el_dom.children[0].eq(Node(0)), el_dom
                new_cons = el_dom.children[1].infix()
                cons_sympy.append(sp.S(new_cons))
                constraint_tab.append(lambda x: expr_to_fun(x, cons_sympy[-1], point))
        constraints = [opt.NonlinearConstraint(ci, 0, np.inf) for ci in constraint_tab]
        constraints.extend([opt.NonlinearConstraint(ci, -1e20, 1e20) for ci in constraint_tab_neq])
        if debug:
            print("cons_sympy", cons_sympy)
            print("cons_sympy_neq", cons_sympy_neq)
            print("Here", V)
        bounds = []
        for _ in range(len(point)):
            bounds.append((-10, 10))
        y = opt.shgo(
            expr_to_fun,
            bounds,
            args=(V, point),
            minimizer_kwargs={"constraints": constraints, "options": {"maxiter": 3000, "disp": False}, "jac": grad_V_fun},
        )

        if debug:
            print(y)
        if not y.success or y.fun < -5e19:
            return -1
        c = 1 if y.fun > -EPSILON else 0
    return c


class TreeParser:
    """
    Class for parsing mathematical expressions (that are represented as a tree by default)
    """

    def __init__(self, adds, muls, funcs, variables, int_base, mulcode="*"):
        self.adds = adds
        self.muls = muls
        self.funcs = funcs
        self.variables = variables
        self.int_base = int_base
        self.mulcode = mulcode
        self.symbols = ["(", ")"] + self.adds + self.muls + self.funcs + self.variables

    def next_token(self, s):
        i = 0
        while i < len(s) and s[i] in [" ", "\t"]:
            i += 1
        if i == len(s):
            return "EOS", i
        for symb in self.symbols:
            if s[i:].startswith(symb):
                return symb, i + len(symb)
        if s[i].isdigit():
            val = int(s[i])
            i += 1
            while i < len(s) and s[i].isdigit():
                val = val * self.int_base + int(s[i])
                i += 1
            return val, i
        return "wut", i + 1

    def read_expr(self, s, node):
        e, i = self.next_token(s)
        if e == "(":
            i2 = self.read_expr(s[i:], node)
            f, i3 = self.next_token(s[i + i2 :])
            if f == ")":
                return i + i2 + i3
            else:
                raise ValueError(f"no closing parenthesis in {s}", s)
        else:
            son1 = Node(None)
            pos = self.read_term(s, son1)
            e, i = self.next_token(s[pos:])
            if e in self.adds:
                son2 = Node(None)
                node.value = e
                node.push_child(son1)
                pos2 = self.read_expr(s[pos + i :], son2)
                node.push_child(son2)
                return pos + i + pos2
            else:
                node.value = son1.value
                node.children = son1.children
                return pos

    def read_term(self, s, node):
        son1 = Node(None)
        pos = self.read_neg(s, son1)
        e, i = self.next_token(s[pos:])
        if e in self.muls:
            son2 = Node(None)
            node.value = e
            node.push_child(son1)
            pos2 = self.read_term(s[pos + i :], son2)
            node.push_child(son2)
            return pos + i + pos2
        else:
            node.value = son1.value
            node.children = son1.children
            return pos

    def read_neg(self, s, node):
        e, i = self.next_token(s)
        cnt = 0
        pos = 0
        while e == "-":
            cnt += 1
            pos += i
            e, i = self.next_token(s[pos:])
        cnt = cnt % 2
        if cnt == 1:
            if isinstance(e, int):
                i = self.read_factor(s[pos:], node)
                node.value = -node.value
                return pos + i
            else:
                node.value = self.mulcode
                node.push_child(Node(-1))
                ch = Node(None)
                node.push_child(ch)
                i = node.read_factor(s[pos:], ch)
                return pos + i
        else:
            i = self.read_factor(s[pos:], node)
            return pos + i

    def read_factor(self, s, node):
        e, i = self.next_token(s)
        if e in self.funcs:
            node.value = e
            ch = Node(None)
            node.push_child(ch)
            i2 = self.read_expr(s[i:], ch)
            return i + i2
        elif e == "(":
            i2 = self.read_expr(s, node)
            return i2
        elif e in self.variables or isinstance(e, int):
            node.value = e
            return i
        else:
            raise ValueError(f"Incorrect leaf at {s}", s)

    def S(self, s):
        tree = Node(None)
        self.read_expr(s, tree)
        return tree


class Node:
    """
    Class representing the mathematical expression as Trees
    """
    def __init__(self, value, children=None):
        self.value = value
        self.children: List["Node"] = children if children else []
        self._domain: Optional[List["Node"]] = None

    def push_child(self, child):
        self.children.append(child)

    def prefix(self):
        """
        Enumerate tree in prefix expression
        """
        s = str(self.value)
        for c in self.children:
            s += ", " + c.prefix()
        return s

    # export to latex qtree format: prefix with \Tree, use package qtree
    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self) -> "str":
        """
        Convert tree in readable mathematical expression
        """
        assert isinstance(self, Node), self
        nb_children = len(self.children)
        if nb_children <= 1:
            s = str(self.value)
            if isinstance(self.value, int) and self.value < 0:
                s = "(" + s + ")"
            elif nb_children == 1:
                s += "(" + self.children[0].infix() + ")"
            return s
        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def __len__(self):
        """
        Length of the mathematical expression
        """
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        """
        Alias of the infix (readable mathematical expression)
        """
        return self.infix()

    def __add__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("+", [self, node])

    def __sub__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("-", [self, node])

    def __radd__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("+", [node, self])

    def __mul__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("*", [self, node])

    def __rmul__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("*", [node, self])

    def __ne__(self, node):
        return Node("!=", [self, autocast(node)])

    def __le__(self, node):
        return Node("<=", [self, autocast(node)])

    def __lt__(self, node):
        return Node("<", [self, autocast(node)])

    def __ge__(self, node):
        return Node("<=", [autocast(node), self])

    def __gt__(self, node):
        return Node("<", [autocast(node), self])

    def __pow__(self, node) -> "Node":
        return Node("^", [self, autocast(node)])

    def ln(self) -> "Node":
        """
        Log function
        """
        return Node("ln", [self])

    def exp(self) -> "Node":
        """
        Exponential function
        """
        return Node("exp", [self])

    def eq(self, node) -> bool:
        """
        Check if two trees are equal
        """
        return self.prefix() == node.prefix()

    def clone(self) -> "Node":
        """
        Create a copy of the Node
        """
        return Node(self.value, [c.clone() for c in self.children])

    def replace(self, x, y):
        """
        copy of the tree where we replaced the source tree by a target tree.
        """
        if self.eq(x):
            return y
        return Node(self.value, [c.replace(x, y) for c in self.children])

    def replace_ops(self, ops1: str, ops_lst: List[str], except_exp: bool = False) -> "Node":
        """
        replace a specific operator in a tree. The number of children should be compatible.
        INPUTS:
        - ops1: operator to be replaced
        - ops_lst: operators to replace by, in order.
        - except_exp: if True and ops1 is unary the function has no effect if its children is exponential.
        OUTPUT:
        - New tree where the operator has been replaced
        """
        if self.value == ops1:
            if except_exp and len(self.children) == 1 and self.children[0].value == "exp":
                return Node(self.value, [c.clone() for c in self.children])
            current_children = [c.clone() for c in self.children]
            for op in ops_lst:
                new_node = Node(op, current_children)
                current_children = [new_node]
            return new_node
        return Node(self.value, [c.replace_ops(ops1, ops_lst) for c in self.children])

    def remove_ops(self, ops: str, parent_node: Optional["Node"] = None, node_index: Optional[int] = 0) -> "Node":
        """
        Remove a unary operator from a tree. e.g. exp(abs(5))->exp(5) for ops=abs.
        """
        if self.value == ops:
            assert len(self.children) == 1
            if parent_node == None:
                return self.children[0].clone()
            new_children = [child.clone() for child in parent_node.children]
            new_children[node_index] = self.children[0].clone()
            return Node(parent_node.value, new_children)
        return Node(self.value, [c.remove_ops(ops, self, i) for i, c in enumerate(self.children)])

    def _find_domain(self, refresh=False):
        self._domain: List["Node"] = []
        for c in self.children:
            if refresh or c.domain() is None:
                c._find_domain(refresh)
            self._domain.extend(c.domain())
        if self.value in {"acos", "asin"}:
            self._domain.append(self.children[0] + Node(1) >= 0)
            self._domain.append(Node(1) - self.children[0] >= 0)
        if self.value == "tan":
            self._domain.append(self.children[0] + Node("/", [Node("pi"), Node(2)]) > 0)
            self._domain.append(Node("/", [Node("pi"), Node(2)]) - self.children[0] > 0)
        if self.value == "sqrt":
            self._domain.append(self.children[0] >= 0)
        elif self.value == "ln":
            self._domain.append(self.children[0] > 0)
        elif self.value == "div":
            self._domain.append(self.children[1] != 0)
        elif self.value == "^":
            assert len(self.children) == 2
            c0, c1 = self.children
            c2 = c1.value
            try:
                c2_f = float(c2)
                c_is_int = int(c2_f) - c2_f == 0
                if c2_f < 0 and c_is_int:
                    self._domain.append(c0 != 0)
                elif c2_f < 0:
                    self._domain.append(c0 > 0)
            except ValueError as e:
                if e.args[0].startswith("could not convert string to float:"):
                    self._domain.append(c0 > 0)
                else:
                    raise DomainError(f"Domain not implemented for exponent {c2}, current node is {self}")

    def domain(self, refresh=False) -> List["Node"]:
        """
        Returns a set of constraints that have to be satisfied for the mathematical expression (represented by the Node) to make sense.
        e.g. ln(\sqrt(x0)) will return both x0>=0 and \sqrt(x0) > 0
        """
        if (self._domain is None) or refresh:
            # To avoid recomputing every time if needed several time
            self._find_domain()
        return self._domain


def autocast(x):
    if isinstance(x, Node):
        return x
    elif type(x) is int:
        return Node(x)
    else:
        raise RuntimeError(f"Unexpected type: {x}")


class ODEEnvironment(object):
    """
    Main environment class for the mathematical problem studied
    """

    TRAINING_TASKS = {"ode_lyapunov"}

    def __init__(self, params):
        """
        Definition of the parameters for the generation.
        See the parser's generation below in this file for more detail on each parameter.
        """
        self.max_degree = params.max_degree
        self.min_degree = params.min_degree
        assert self.min_degree >= 2
        assert self.max_degree >= self.min_degree

        self.max_ops = 200

        self.int_base = params.int_base
        self.max_int = params.max_int
        self.positive = params.positive
        self.nonnull = params.nonnull

        self.custom_unary_probs = params.custom_unary_probs
        self.prob_trigs = params.prob_trigs
        self.prob_arc_trigs = params.prob_arc_trigs
        self.prob_logs = params.prob_logs
        self.prob_others = 1.0 - self.prob_trigs - self.prob_arc_trigs - self.prob_logs
        assert self.prob_others >= 0.0

        self.prob_int = params.prob_int
        self.precision = params.precision

        self.max_len = params.max_len
        self.max_output_len = params.max_output_len
        self.eval_value = params.eval_value
        self.skip_zero_gradient = params.skip_zero_gradient
        self.prob_positive = params.prob_positive
        self.lyap_max_degree = params.lyap_max_degree
        self.lyap_n_terms = params.lyap_n_terms
        self.lyap_polynomial_V = params.lyap_polynomial_V
        self.lyap_polynomial_H = params.lyap_polynomial_H
        self.lyap_pure_polynomial = params.lyap_pure_polynomial
        self.lyap_cross_term = params.lyap_cross_term
        self.lyap_max_nb_cross_term = params.lyap_max_nb_cross_term
        self.lyap_nb_ops_proper = params.lyap_nb_ops_proper
        self.lyap_nb_ops_lyap = params.lyap_nb_ops_lyap
        self.lyap_generate_gradient_flow = params.lyap_generate_gradient_flow
        self.lyap_gen_weight = params.lyap_gen_weight
        self.lyap_max_order_pure_poly = params.lyap_max_order_pure_poly
        self.lyap_max_n_term_fwd = params.lyap_max_n_term_fwd
        self.lyap_generate_sample_fwd = params.lyap_generate_sample_fwd
        self.lyap_find_domain = params.lyap_find_domain
        self.lyap_SOS_checker = params.lyap_SOS_checker
        self.lyap_SOS_fwd_gen = params.lyap_SOS_fwd_gen
        self.lyap_proba_fwd_gen = params.lyap_proba_fwd_gen
        self.lyap_proper_fwd = params.lyap_proper_fwd  # Whether we want a proper Lyapunov function when we generate in fwd
        self.lyap_multigen = params.lyap_multigen
        self.lyap_memory_nb = 0  # memory parameter, always start by this value
        self.lyap_memory_fun = None  # memory parameter, always start by this value

        assert self.lyap_pure_polynomial or (
            not self.lyap_SOS_checker and not self.lyap_SOS_fwd_gen
        )  # SOS checker only makes sense for pure polynomial

        self.lyap_only_2_norm = params.lyap_only_2_norm
        self.lyap_proba_diagonal = params.lyap_proba_diagonal
        self.lyap_proba_proper_composition = params.lyap_proba_proper_composition
        self.lyap_proba_proper_multiply = params.lyap_proba_proper_multiply
        self.lyap_proba_cross_composition = params.lyap_proba_cross_composition
        self.lyap_proba_cross_multiply = params.lyap_proba_cross_multiply
        self.lyap_basic_functions_num = params.lyap_basic_functions_num  # see parser metadata
        self.lyap_basic_functions_den = params.lyap_basic_functions_den
        assert (not self.lyap_pure_polynomial) or (self.lyap_polynomial_H and self.lyap_polynomial_V)
        assert (not self.lyap_only_2_norm) or self.lyap_proba_diagonal == 1
        if self.lyap_pure_polynomial:
            assert self.lyap_proba_cross_composition == 0
            assert self.lyap_proba_proper_composition == 0
            assert self.lyap_proba_cross_multiply == 0
            assert self.lyap_proba_proper_multiply == 0
        assert self.lyap_polynomial_V or self.lyap_basic_functions_num  # No need for p2 when we are in the more generique case

        self.lyap_stable = params.lyap_stable
        self.lyap_predict_stability = params.lyap_predict_stability
        assert (not self.lyap_stable) or (not self.lyap_predict_stability)

        self.lyap_float_resolution_poly = params.lyap_float_resolution_poly
        self.lyap_strict = (
            params.lyap_strict
        )  # whether we want a strict Lyapunov function, i.e. \nabla V decays along all coordinates and cannot be conserved on some hyperplan.
        self.lyap_local = params.lyap_local  # whether we want a local Lyapunov function
        self.lyap_proper = params.lyap_proper  # whether we want a proper Lyapunov function (i.e. grows to infinity at infinity) or not.
        self.lyap_debug = params.lyap_debug
        assert self.lyap_proper or self.lyap_cross_term

        self.np_positive = np.zeros(self.max_degree + 1, dtype=int)
        self.np_total = np.zeros(self.max_degree + 1, dtype=int)

        self.SYMPY_OPERATORS = {
            # Elementary functions
            sp.Add: "+",
            sp.Mul: "*",
            sp.Pow: "^",
            sp.exp: "exp",
            sp.log: "ln",
            sp.Abs: "Abs",
            # sp.sign: 'sign',
            # Trigonometric Functions
            sp.sin: "sin",
            sp.cos: "cos",
            sp.tan: "tan",
            # sp.cot: 'cot',
            # sp.sec: 'sec',
            # sp.csc: 'csc',
            # Trigonometric Inverses
            sp.asin: "asin",
            sp.acos: "acos",
            sp.atan: "atan",
            # sp.acot: 'acot',
            # sp.asec: 'asec',
            # sp.acsc: 'acsc',
            sp.DiracDelta: "delta0",
        }

        self.trig_ops = ["sin", "cos", "tan"]
        self.arctrig_ops = ["asin", "acos", "atan"]
        self.exp_ops = ["exp", "ln"]
        self.other_ops = ["sqrt"]

        op_set = {
            "+": 2,
            "-": 2,
            "*": 2,
            "/": 2,
            "^": 2,
            "sqrt": 1,
            "exp": 1,
            "ln": 1,
            "sin": 1,
            "cos": 1,
            "tan": 1,
            "asin": 1,
            "acos": 1,
            "atan": 1,
            "Abs": 1,
        }

        self.operators_lyap = params.operators_lyap or op_set
        assert all(op in op_set for op in self.operators_lyap), [k for k in self.operators_lyap if k not in op_set]

        # Sets of increasing, positive and bounded functions that will be used in composition and multiplication when generating a Lyapunov function candidate.
        self.increasing_functions_sp = [sp.exp, sqrt1, ln1]
        self.positive_functions_sp = [sp.exp, sin1, cos1]
        self.bounded_functions_sp = [[sp.cos, sp.pi], [sp.sin, -sp.pi / 2]]

        # Operators for the generation
        self.operators = self.operators_lyap
        self.unaries = [o for o in self.operators.keys() if self.operators[o] == 1]  # For generic generation (not in the paper)
        self.unaries_poly_lyap = ["id", "id", "pow2", "cos", "sin", "exp", "ln"]  # For generation of polynomial of non-polynomial operators
        self.binaries = [o for o in self.operators.keys() if self.operators[o] == 2]
        self.unary = len(self.unaries) > 0

        assert self.max_int >= 1
        assert self.precision >= 2
        assert self.eval_value == 0.0  # equilibrium is taken at x=0.

        # variables
        self.variables = OrderedDict({f"x{i}": sp.Symbol(f"x{i}") for i in range(2 * self.max_degree)})

        self.eval_point = OrderedDict({self.variables[f"x{i}"]: self.eval_value for i in range(2 * self.max_degree)})

        self.tree_parser = TreeParser(["+", "-"], ["*", "/"], self.unaries, list(self.variables.keys()), int_base=self.int_base)

        # symbols / elements
        self.constants = ["pi", "E"]

        self.symbols = ["I", "INT+", "INT-", "FLOAT+", "FLOAT-", ".", "10^"]
        self.elements = [str(i) for i in range(max(10, self.int_base))]

        # SymPy elements
        self.local_dict = {}
        for k, v in list(self.variables.items()):
            assert k not in self.local_dict
            self.local_dict[k] = v

        # vocabulary
        self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) + list(self.operators.keys()) + self.symbols + self.elements
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        self.func_separator = "<SPECIAL_3>"  # separate equations in a system
        logger.info(f"words: {self.word2id}")

        # initialize distribution for binary and unary-binary trees
        self.distrib = self.generate_dist(2 * self.max_ops)

    def get_integer(self, positive=False, max_int: Optional[int] = None):
        """
        Generate an integer (can also be a number depending on the option).
        Flag positive with override the class parameter if needed
        """
        if max_int is None:
            max_int = self.max_int
        if self.positive and self.nonnull or positive:
            return self.rng.randint(1, max_int + 1)
        if self.positive:
            return self.rng.randint(0, max_int + 1)
        if self.nonnull:
            s = self.rng.randint(1, 2 * max_int + 1)
            return s if s <= max_int else (max_int - s)

        return self.rng.randint(-max_int, max_int + 1)

    def generate_leaf(self, degree, index):
        """
        Generate a leaf at random. In a mathematical expression this corresponds to either a variable, a constant, or numerical value.
        This is by contrast with operators that are internal nodes.
        """
        if self.rng.rand() < self.prob_int:
            return self.get_integer()
        elif degree == 1:
            return self.variables[f"x{index}"]
        else:
            return self.variables[f"x{self.rng.randint(degree)}"]

    def generate_ops(self, arity):
        """
        Generate an operator at random.
        """
        if arity == 1:
            if self.custom_unary_probs:
                w = [self.prob_trigs, self.prob_arc_trigs, self.prob_logs, self.prob_others]
                s = [self.trig_ops, self.arctrig_ops, self.exp_ops, self.other_ops]
                return self.rng.choice(s, p=w)
            else:
                return self.rng.choice(self.unaries)

        else:
            return self.rng.choice(self.binaries)

    def generate_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D = []
        D.append([0] + ([1 for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        return D

    def sample_next_pos(self, nb_empty, nb_ops):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `nb_empty` - 1}.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])
        probs = [p / self.distrib[nb_ops][nb_empty] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = self.rng.choice(len(probs), p=probs)
        arity = 1 if self.unary and e < nb_empty else 2
        e %= nb_empty
        return e, arity

    def generate_tree(self, nb_ops, degree, index=0):
        """
        Generate a random mathematical expression
        """
        tree = Node(0)
        empty_nodes = [tree]
        next_en = 0
        nb_empty = 1
        while nb_ops > 0:
            next_pos, arity = self.sample_next_pos(nb_empty, nb_ops)
            for n in empty_nodes[next_en : next_en + next_pos]:
                n.value = self.generate_leaf(degree, index)
            next_en += next_pos
            empty_nodes[next_en].value = self.generate_ops(arity)
            for _ in range(arity):
                e = Node(0)
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            nb_empty += arity - 1 - next_pos
            nb_ops -= 1
            next_en += 1
        for n in empty_nodes[next_en:]:
            n.value = self.generate_leaf(degree, index)
        return tree

    def generate_polynomial(self, nterm, max_factor, degree, unaries, noconstant=True):
        """
        Generate a function that is polynomial with respect to the unary operators unaries
        """
        pol = set()
        for i in range(nterm):
            nfactor = self.rng.randint(1, max_factor + 1)
            vars = set()
            for j in range(nfactor):
                vars.add((self.rng.randint(0, degree), self.rng.randint(0, len(unaries))))
            pol.add(tuple(vars))
        for i in range(len(pol)):
            v = list(pol)[i]
            for j in range(len(v)):
                op = unaries[v[j][1]]
                var = Node(self.variables[f"x{v[j][0]}"])
                if op == "id":
                    term = var
                elif op == "ln":
                    term = Node("ln", [Node("+", [Node(1), var])])
                elif len(op) > 3 and op[:3] == "pow":
                    term = Node("^", [var, Node(int(op[3:]))])
                else:
                    term = Node(op, [var])
                p = term if j == 0 else Node("*", [p, term])
            coeff = self.get_integer()
            sgn = np.sign(coeff)
            if abs(coeff) != 1:
                p = Node("*", [Node(abs(coeff)), p])
            tree = p if i == 0 else Node("+" if coeff > 0 else "-", [tree, p])
        if not noconstant:
            coeff = self.get_integer()
            tree = Node("+" if coeff > 0 else "-", [tree, Node(abs(coeff))])
        return tree

    def generate_ipd_matrix(self, degree, max_integer: Optional[bool] = None, positive_definite: bool = True):
        """
        Generate a positive definite matrix with float resolution coefficients
        """
        max_int = max_integer or self.max_int
        A = self.rng.randint(-max_int, max_int, size=(degree, degree))
        P = A.astype(int) @ A.T.astype(int)
        if self.lyap_debug:
            print("P semidefinite", P)
        if positive_definite:
            P = P + np.eye(degree).astype(int)
            if self.lyap_debug:
                print("P positive definite", P)
        if self.lyap_float_resolution_poly == 1:
            # Find the greatest common divisor of all the coefficients
            gcd_val = np.gcd.reduce(P.flatten())
            # Divide all elements of the matrix by the greatest common divisor
            P = P // gcd_val
        else:
            # Divide all terms by max_int to normalize the value of the coefficients and be consistant with the cross-terms and the diagonal Lyapunov functions
            P = P / (degree * max_int)

        return P

    def get_linearized_system(self, sys_sp):
        """
        Takes a nonlinear system in sympy and output the linearized system around self.eval_point and the maximal value of the eigenvalues of the system.
        Input: system (List of sympy expression)
        Output: linearized system (list of sympy expression), max_value of eigenvalues
        """
        lin_sys = []
        dyn_matrix = []
        for eq_sp in sys_sp:
            eq_lin = 0
            vect_mat = []
            for i in range(len(sys_sp)):
                coeff = eq_sp.diff(self.variables[f"x{i}"]).subs(self.eval_point).evalf()
                eq_lin = eq_lin + coeff * self.variables[f"x{i}"]
                vect_mat.append(coeff)
            dyn_matrix.append(vect_mat)
            lin_sys.append(eq_lin)
        try:
            max_eigenval = max(complex(num).real for num in sp.Matrix(dyn_matrix).eigenvals().keys())
        except (TypeError, ValueError) as e:
            return None
        return lin_sys, max_eigenval

    def generate_polynomial_system(
        self,
        bounded_poly: bool = True,
        unaries_poly: List[str] = ["id", "pow2", "pow3", "pow4"],
        max_order: Optional[int] = None,
        max_n_term: Optional[int] = None,
    ):
        """
        Generate a system of polynomial equations (forward generation).
        Inputs:
            bounded_poly: True if we use the generate_bounded_polynomial method, False for the generate_polynomial_method
            unaries_poly: unary operator to be used in the method generate_polynomial if suitable.
        Output:
            x: an encoded input expression for the model
        """
        if max_order is None:
            max_order = self.lyap_max_degree + 2 * self.lyap_max_order_pure_poly - 1
        if max_n_term is None:
            max_n_term = self.lyap_max_n_term_fwd
        n_eq = self.rng.randint(self.min_degree, self.max_degree + 1)
        system = []
        for _ in range(n_eq):
            if bounded_poly:
                new_poly_expr = self.generate_bounded_polynomial(
                    n_eq, max_order=max_order, max_pow=max_order, max_number=min(2 * self.max_int, 10), max_n_term=max_n_term
                )
            else:
                n_ops = self.rng.randint(1, 2 * self.max_degree + 1)
                new_poly_expr = self.generate_polynomial(n_ops, 4, n_eq, unaries_poly, True)
            poly_sp = sp.S(str(new_poly_expr))
            poly_sp = sp.expand(poly_sp)
            system.append(poly_sp)
        try:
            x = self.write_int(n_eq)
            for s in system:
                x.append(self.func_separator)
                x.extend(self.sympy_to_prefix(s))
        except Exception:
            return None
        return x

    def generate_random_system(
        self,
        poly: bool = False,
        unaries: Optional[List[str]] = None,
        binaries: Optional[List[str]] = None,
        nb_term: Optional[int] = None,
        simplify_flag: bool = False,
        make_eq_compatible: bool = False,
        discard_non_eq: bool = False,
    ):
        """
        Generate a random system of equations (forward generation).
        Inputs:
            unaries: List of unary operators to be used, if None, uses self.unary.
            binaries: List of binary operators to be used, if None, uses self.binary
            poly: True if the generation is using a polynomial of potentially non-polynomial operators
        Outputs:
            x: an encoded expression for the model
        Warnings: if poly is False, then self.unary is used anyway (generation use generate_tree).
        """
        max_factor = 3
        nb_terms = nb_term or self.lyap_nb_ops_proper
        if unaries is None:
            unaries = self.unaries
        if binaries is None:
            binaries = self.binaries
        n_eq = self.rng.randint(self.min_degree, self.max_degree + 1)
        system = []
        i = 0
        non_eq_error = 0
        while i < n_eq:
            if poly:
                nb_term = self.rng.randint(1, nb_terms + 1)
                expr_eq = self.generate_polynomial(nterm=nb_terms, max_factor=max_factor, degree=n_eq, unaries=unaries, noconstant=True)
            else:
                nb_ops = self.rng.randint(1, max_factor * nb_terms + 1)
                expr_eq = self.generate_tree(nb_ops, n_eq)
            expr_sy = sp.S(str(expr_eq))
            if make_eq_compatible:
                try:
                    value_0 = expr_sy.subs(self.eval_point).evalf()
                    if value_0 != 0:
                        expr_sy = expr_sy - value_0
                except TypeError as er:
                    if er.args[0].startswith("cannot unpack non-iterable ComplexInfinity object"):
                        continue
                    else:
                        raise RuntimeError("Wrong evalf in forward generation") from er
            if discard_non_eq and not make_eq_compatible:
                try:
                    value_0 = expr_sy.subs(self.eval_point).evalf()
                    if value_0 != 0:
                        non_eq_error += 1
                        continue
                except TypeError as er:
                    if er.args[0].startswith("cannot unpack non-iterable ComplexInfinity object"):
                        continue
                    else:
                        raise RuntimeError("Wrong evalf in forward generation") from er
            if simplify_flag:
                expr_sy = simplify(expr_sy, 5)
            i += 1
            system.append(expr_sy)
        try:
            x = self.write_int(n_eq)
            for s in system:
                x.append(self.func_separator)
                x.extend(self.sympy_to_prefix(s))
        except UnknownSymPyOperator as e:
            if self.lyap_debug:
                print(f"error during forward generation {e.args[0]}")
            return None
        return x

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in base self.int_base.
        """
        res = []
        neg = val < 0
        val = -val if neg else val
        while True:
            rem = val % self.int_base
            val = val // self.int_base
            res.append(str(rem))
            if val == 0:
                break
        res.append("INT-" if neg else "INT+")
        return res[::-1]

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        if len(lst) < 2 or lst[0] not in ["INT+", "INT-"] or not lst[1].isdigit():
            return np.nan, 0
        val = int(lst[1])
        i = 1
        for x in lst[2:]:
            if not x.isdigit():
                break
            val = val * self.int_base + int(x)
            i += 1
        if lst[0] == "INT-":
            val = -val
        return val, i + 1

    def write_float(self, value, precision=None):
        """
        Write a float number.
        """
        precision = self.precision if precision is None else precision
        assert value not in [-np.inf, np.inf]
        res = ["FLOAT+"] if value >= 0.0 else ["FLOAT-"]
        m, e = (f"%.{precision}e" % abs(value)).split("e")
        i, f = m.split(".")
        assert e[0] in ["+", "-"]
        e = int(e[1:] if e[0] == "+" else e)
        i = i + f
        ipart = int(i)
        fpart = 0
        expon = int(e) - precision
        if ipart == 0 and fpart == 0:
            value = 0.0
            expon = 0
        res = res + self.write_int(ipart)[1:] + ["10^"] + self.write_int(expon)
        return res

    def parse_float(self, lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        if len(lst) < 2 or lst[0] not in ["FLOAT+", "FLOAT-"]:
            return np.nan, 0
        sign = -1 if lst[0] == "FLOAT-" else 1
        if not lst[1].isdigit():
            return np.nan, 1
        mant = 0.0
        i = 1
        for x in lst[1:]:
            if not (x.isdigit()):
                break
            mant = mant * self.int_base + int(x)
            i += 1
        mant *= sign
        if len(lst) > i and lst[i] == "10^":
            i += 1
            exp, offset = self.parse_int(lst[i:])
            if np.isnan(exp):
                return np.nan, i
            i += offset
        else:
            return np.nan, i
            # exp = 0
        return mant * float(10.0**exp), i

    def input_to_infix(self, lst):
        res = ""
        degree, offset = self.parse_int(lst)
        res = str(degree) + "|"

        offset += 1
        l1 = lst[offset:]
        nr_eqs = degree
        for i in range(nr_eqs):
            pti = self.prefix_to_infix(l1)
            if pti is None:
                return "incorrect"
            s, l2 = pti
            res = res + s + "|"
            l1 = l2[1:]
        return res[:-1]

    def output_to_infix(self, lst):
        val, _ = self.parse_float(lst)
        return str(val)

    def prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            return None
        t = expr[0]
        if t in self.operators.keys():
            args = []
            l1 = expr[1:]
            for _ in range(self.operators[t]):
                pti = self.prefix_to_infix(l1)
                if pti is None:
                    return None
                i1, l1 = pti
                args.append(i1)
            if self.operators[t] == 1:
                return f"{t}({args[0]})", l1
            return f"({args[0]}{t}{args[1]})", l1
        elif t in self.variables or t in self.constants or t == "I":
            return t, expr[1:]
        elif t == "FLOAT+" or t == "FLOAT-":
            val, i = self.parse_float(expr)
            if np.isnan(val):
                return None
        else:
            val, i = self.parse_int(expr)
            if np.isnan(val):
                return None
        return str(val), expr[i:]

    def prefix_to_node(self, tokens: List[str]) -> Node:
        """
        Convert a prefix list into the corresponding Node
        """
        assert isinstance(tokens, list)
        if len(tokens) == 0:
            raise RuntimeError("Empty list of tokens in prefix to node")

        def _tokens_to_node(offset: int) -> Tuple["Node", int]:
            if offset >= len(tokens):
                raise RuntimeError(f"Missing token, parsing {' '.join(tokens)} with offset {offset}")
            tok = tokens[offset]
            assert isinstance(tok, str)
            if tok in self.binaries:
                lhs, rhs_offset = _tokens_to_node(offset + 1)
                rhs, next_offset = _tokens_to_node(rhs_offset)
                return Node(tok, [lhs, rhs]), next_offset
            elif tok in self.unaries:
                term, next_offset = _tokens_to_node(offset + 1)
                return Node(tok, [term]), next_offset
            elif tok in self.variables or tok in self.constants:
                return Node(tok), offset + 1
            else:
                if tok not in self.symbols:
                    raise RuntimeError(f"Unexpected token {tok} in {tokens}")
                i, v, found = offset + 1, 0, False
                if tok in {"INT+", "INT-"}:
                    while i < len(tokens):
                        if tokens[i] not in self.elements:
                            break
                        v = v * self.int_base + int(tokens[i])
                        i += 1
                        found = True
                if tok in {"FLOAT+", "FLOAT-"}:
                    found1 = False
                    while i < len(tokens):
                        if tokens[i] not in self.elements:
                            break
                        v = v * self.int_base + int(tokens[i])
                        i += 1
                        found1 = True
                    assert tokens[i] == "10^"
                    expon_sign = tokens[i + 1]
                    assert expon_sign in {"INT+", "INT-"}
                    i += 2
                    expon = 0
                    while i < len(tokens):
                        if tokens[i] not in self.elements:
                            break
                        expon = expon * self.int_base + int(tokens[i])
                        i += 1
                        found = found1
                    if expon_sign == "INT-":
                        expon = -expon
                    v = v * 10 ^ (expon)
                if not found:
                    raise RuntimeError(f"Missing digits, parsing {' '.join(tokens)}")
                if tok == "INT-" or tok == "FLOAT-":
                    v *= -1
                return Node(v), i

        node, last_offset = _tokens_to_node(0)
        if last_offset != len(tokens):
            raise RuntimeError(f"prefix_to_node didn't parse everything: {' '.join(tokens)}. " f"Stopped at length {last_offset}")
        return node

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        assert (op == "+" or op == "*") and (n_args >= 2) or (op != "+" and op != "*") and (1 <= n_args <= 2)

        # square root
        if op == "^" and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
            return ["sqrt"] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Float):
            return self.write_float(float(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ["/"] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ["E"]
        elif expr == sp.pi:
            return ["pi"]
        elif expr == sp.I:
            raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def decode_src(self, x: str) -> List[str]:
        """
        decode the encoded input x (that is the mathematical system) and returns the system in infix version
        """
        src = x
        _, pos = self.parse_int(src)
        sys = []
        nx = src[pos:]
        while len(nx) > 0:
            b, nx = self.prefix_to_infix(nx[1:])
            sys.append(b)
        return sys

    def decode_lyap(self, y: str) -> Tuple[int, str]:
        """
        decode the encoded output y (the Lyapunov function) and returns the stability which can be None if it is not predicted by the transformer,
        and the expression of the Lyapunov function as a string.
        """
        nx = y
        stability = None
        b, nx = self.prefix_to_infix(nx)
        lyap = [b]
        while len(nx) > 0:
            b, nx = self.prefix_to_infix(nx[1:])
            lyap.append(b)
        if (not self.lyap_stable) and self.lyap_predict_stability:
            stability = lyap[0]
            lyap = lyap[1:]
        lyap_infix = lyap[0]
        for j in range(len(lyap[1:])):
            lyap_infix += "+" + lyap[j + 1]
        return (stability, lyap_infix)

    def generate_bounded_polynomial(
        self,
        degree,
        max_order=4,
        max_pow=4,
        p_crossed=0.5,
        max_n_term=5,
        max_factor=3,
        bias_monome=0.2,
        proba_add_factor=0.95,
        max_number=None,
        nonzero=True,
        positive_coeff=False,
        no_constant: bool = True,
    ):
        """
        Generate a polynomial in expanded form with a bounded order.
        float_resolution =1 corresponds to integer. nonzero correspond
        to whether or not we require the polynomial to be non (identically) zero
        """
        # Number of significant digits in the polynomial's coefficients
        float_resolution = self.lyap_float_resolution_poly

        # Number of terms of this polynomial
        min_term = 1 if nonzero else 0
        n_term = self.rng.randint(min_term, max_n_term + 1)

        # Max value for the coefficients
        if max_number is None:
            max_number = self.max_int

        # Maximal degree or the polynomial
        min_order = 1 if no_constant else 0
        max_order_this_poly = self.rng.randint(min_order, max_order + 1)

        # Initializing the polynomial
        poly = None
        i = 0

        # Building terms as sum of monomials
        while i < n_term or (poly is None and nonzero):
            order = 0
            prod = 0
            # coeff = self.rng.randint(-1000, 1000) / 100

            # Building coefficient of the monomial
            coeff = self.rng.randint(-max_number * float_resolution, max_number * float_resolution) / float_resolution
            if abs(coeff) < (1e-2) / float_resolution:
                i += 1
                continue
            if positive_coeff:
                coeff = abs(coeff)
            if float_resolution == 1:
                coeff = int(coeff)

            # Initialize the monomial
            poly_monome = Node(coeff)
            is_const = True

            # Maximal order of this monomial
            max_order_this_monome = self.rng.randint(min_order, max_order_this_poly + 1)

            # Building the monomial
            while order < max_order_this_monome and prod < max_factor and self.rng.rand() < proba_add_factor:
                this_ord = self.rng.randint(min_order, min(max_pow, max_order_this_monome - order) + 1)
                index = self.rng.randint(degree)
                if this_ord > 0:
                    if this_ord == 1:
                        new_node = Node(self.variables[f"x{index}"])
                    else:
                        # n_value = f"pow{this_ord}"
                        new_node = Node("^", [Node(self.variables[f"x{index}"]), Node(this_ord)])
                    poly_monome = Node("*", [poly_monome, new_node])
                    is_const = False
                order += this_ord
                prod += 1
            if poly is None and (not is_const or not no_constant):
                # if poly_monome is not a constant or if we allow constant in the polynomial
                poly = poly_monome
            elif not is_const or not no_constant:
                poly = Node("+", [poly, poly_monome])
            if self.rng.rand() < bias_monome:
                prod = max_factor
            i += 1
        if poly is None and not nonzero:
            poly = Node(0)
        return poly

    def to_matlab(
        self, n_examples=100, method=None, filename: str = os.getcwd() + "/SOSTOOLS/matlab_dataset.txt", gen_dataset: Optional[list] = None
    ):
        """
        Generate systems and write them in a file that can be used in MATLAB SOSTOOLS
        Optionally, if gen_dataset is not None, the function only converts gen_dataset into a format
        that can be used in MATLAB SOSTOOLS. gen_dataset must be a list of x where x,y is the output of gen_lyapunov().
        """
        matlab_dataset = []
        if gen_dataset is None:
            gen_dataset = []
            for _ in range(n_examples):
                if method is None:
                    src = self.generate_polynomial_system()
                else:
                    src = method()
                if src is None or isinstance(src, str):
                    continue
                gen_dataset.append(src)
        for src in gen_dataset:
            system = []
            _, pos_src = self.parse_int(src)
            nx = src[pos_src:]
            while len(nx) > 0:
                b, nx = self.prefix_to_infix(nx[1:])
                system.append(b)
            matlab_dataset.append(system)
        with open(filename, "w") as f:
            for lst in matlab_dataset:
                f.write(",".join(lst) + "\n")

    def preprocess_dataset(self,file_path):
        """
        Pre-process data for to_matlab.
        Input: file_path of an encoded dataset
        Output: a list of list that is in the expected format of gen_dataset for the function to_matlab
        """
        with open(file_path, "r") as f:
            lines = [line.rstrip().split('|') for line in f]
        data = [xy.split('\t') for _, xy in lines]
        data = [xy[0].split(" ") for xy in data if len(xy) == 2]
        for dat in data:
            if dat[-1] == "":
                dat.pop(-1)
        return data

    def sympy_to_encoding(self, system, lyap_function, stability: Optional[int] = None):
        """
        Encode a system with a Lyapunov function in a formatted sequence to be passed to the transformer.
        """
        # encode input
        degree = len(system)
        x = self.write_int(degree)
        for s in system:
            x.append(self.func_separator)
            x.extend(self.sympy_to_prefix(s))

        # encode output: f_lyap_tot
        f_lyap_tot = sp.simplify(lyap_function)
        if self.lyap_pure_polynomial:
            f_lyap_tot = sp.expand(f_lyap_tot)
        y = []
        if (not self.lyap_stable) and self.lyap_predict_stability:
            assert stability is not None
            y = self.write_int(stability)
            y.append(self.func_separator)
        y.extend(self.sympy_to_prefix(f_lyap_tot))

        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            if self.lyap_debug:
                print("Length error, returning None")
            return "error:length"
        if self.max_output_len > 0 and len(y) >= self.max_output_len:
            if self.lyap_debug:
                print("Length output error, returning None")
            return "error:outlength"
        return x, y

    def gen_SOS_fwd(self):
        """
        Generate an example of polynomial system with a Sum-Of-Square Lyapunov function with a forward distribution.
        This function generates a system at random and reject it if it does not have a SOS Lyapunov function.
        """
        assert self.lyap_pure_polynomial
        # WARNING: IF WANT TO ALIGN THE MAXIMAL POSSIBLE DEGREE WITH THE BWD (WHICH WILL NOT ALIGN THE DISTRIBUTION BUT MAKE THEM FURTHER) THEN UNCOMMENT THE LINE BELOW INSTEAD
        # max_degree = 2*self.lyap_max_degree
        max_degree = self.lyap_max_degree
        if self.lyap_float_resolution_poly == 1:
            rounded = 1
        else:
            rounded = int(np.floor(np.log10(self.lyap_float_resolution_poly)))
        while True:
            x = self.generate_polynomial_system()
            src = x
            system = self.decode_src(src)
            system_sp = [sp.S(sys) for sys in system]
            y = []
            try:
                maybe_lyap = findlyap(system_sp, max_degree, rounded=rounded, proper=self.lyap_proper_fwd, env_rng=self.rng, max_int=self.max_int)
                if maybe_lyap is None:
                    continue
                else:
                    V = maybe_lyap[0]
                    y.extend(self.sympy_to_prefix(V))

                    if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
                        if self.lyap_debug:
                            print("Length error, returning None")
                        return "error:length"
                    if self.max_output_len > 0 and len(y) >= self.max_output_len:
                        if self.lyap_debug:
                            print("Length output error, returning None")
                        return "error:outlength"
                    return (x, y)
            except (RuntimeError, MyTimeoutError) as error:
                print(error.args[0])

    def gen_lyap_fun(self, degree: int):
        """
        Generate a Lyapunov function candidate, according to the different possible modes
        """
        # Parameters of the Lyapunov function

        n_terms1 = self.rng.randint(1, self.lyap_n_terms + 1)
        unaries_poly_lyap = self.unaries_poly_lyap

        ### Generate h_lyap, the n dimensional component

        p1 = []
        p2 = []
        h_lyap = []

        if self.lyap_cross_term:
            if not self.lyap_proper:
                nb_cross_term = self.rng.randint(1, self.lyap_max_nb_cross_term + 1)
            else:
                nb_cross_term = self.rng.randint(self.lyap_max_nb_cross_term + 1)

            if self.lyap_pure_polynomial and not self.lyap_only_2_norm:  # No crossed term when the lyapunov function is only the 2 norm.
                # Polynomial case
                for _ in range(nb_cross_term):
                    p1.append(
                        self.generate_bounded_polynomial(
                            degree,
                            max_order=self.lyap_max_degree,
                            max_pow=self.lyap_max_degree,
                            max_n_term=n_terms1,
                            max_number=np.round(np.sqrt(self.max_int)),
                        )
                    )
                    if self.lyap_debug:
                        print("here is the generated p1")
                        print(sp.simplify(sp.S(str(p1[-1]))))
            elif self.lyap_basic_functions_num and not self.lyap_only_2_norm:
                # Polynomial of non-polynomial operators
                for _ in range(nb_cross_term):
                    n_term_poly = self.rng.randint(1, n_terms1 + 1)
                    p1.append(self.generate_polynomial(n_term_poly, 3, degree, unaries_poly_lyap, True))
            elif not self.lyap_only_2_norm:
                # Generic generation (not in the paper)
                nb_ops = self.rng.randint(1, self.lyap_nb_ops_lyap + 1)
                p1.append(self.generate_tree(nb_ops, degree))

            if self.lyap_polynomial_V:
                # The cross-term is not a fraction between two members
                h_lyap = p1
            else:
                assert self.lyap_basic_functions_num  # No need for a ratio when the generation is already generic
                n_terms2 = self.rng.randint(1, self.lyap_n_terms + 1)
                # if self.lyap_basic_functions_den:
                for _ in range(nb_cross_term):
                    if self.lyap_basic_functions_den:
                        # Denominator is polynomial
                        p2.append(self.generate_bounded_polynomial(degree, max_n_term=n_terms2, max_number=min(2 * self.max_int, 10)))
                    else:
                        n_term_poly = self.rng.randint(1, n_terms2 + 1)
                        p2.append(self.generate_polynomial(n_terms2, 3, degree, unaries_poly_lyap, False))
                h_lyap = [Node("/", [p1[i], p2[i]]) for i in range(nb_cross_term)]

        ### Generate f_lyap : the proper part of the Lyapunov function

        # f_lyap is the sympy expression, f_vector the n one-dimensional components
        f_lyap = 0
        f_vector = []
        P_matrix = None
        if self.lyap_proper:
            for i in range(degree):
                deg = self.rng.randint(1, 1 + self.lyap_max_degree)
                if self.lyap_only_2_norm:
                    deg = 1
                fi = Node("^", [Node(self.variables[f"x{i}"]), Node(deg)])  # deg = 1 corresponds to (x_i)^2 eventually
                f_vector.append(fi)
                if self.lyap_debug:
                    print("Individual components fi", fi)

            if self.rng.rand() < self.lyap_proba_diagonal:
                if self.lyap_only_2_norm:
                    coeff_vector = [1 for _ in range(degree)]
                else:
                    if self.lyap_float_resolution_poly == 1:
                        # integer coefficients only
                        coeff_vector = [self.get_integer(positive=True) for _ in range(degree)]
                    else:
                        coeff_vector = [
                            self.get_integer(positive=True, max_int=self.max_int * self.lyap_float_resolution_poly) / self.lyap_float_resolution_poly
                            for _ in range(degree)
                        ]
                f_lyap = sp.S(
                    str(sum(Node(coeff_vector[i]) * Node("^", [f_vector[i], Node(2)]) for i in range(degree))), locals=self.local_dict, evaluate=False
                )
                # this corresponds to the component coeff*(fi^2(xi)) in the Lyapunov function.
            else:
                if self.lyap_float_resolution_poly == 1:
                    # integer coefficients only
                    P_matrix = self.generate_ipd_matrix(degree)
                else:
                    P_matrix = (
                        self.generate_ipd_matrix(degree, max_integer=self.max_int * self.lyap_float_resolution_poly) / self.lyap_float_resolution_poly
                    )
                f_vector_sp = sp.Matrix([sp.S(str(f_vector[i]), locals=self.local_dict, evaluate=False) for i in range(degree)])
                f_lyap = P_matrix * f_vector_sp
                f_lyap = f_vector_sp.dot(f_lyap)
                if self.lyap_debug:
                    print("P_matrix, f_vector_sp, f_lyap")
                    print(P_matrix)
                    print(f_vector_sp)
                    print(f_lyap)

            f_lyap_0 = f_lyap.copy()
            if self.rng.rand() < self.lyap_proba_proper_composition:
                index_comp = self.rng.randint(len(self.increasing_functions_sp))
                f_lyap = self.increasing_functions_sp[index_comp](f_lyap).doit()
            if self.rng.rand() < self.lyap_proba_proper_multiply:
                index_mul = self.rng.randint(len(self.positive_functions_sp))
                subexprs = f_lyap_0.args
                subexpr = self.rng.choice(subexprs)
                f_lyap = (f_lyap - f_lyap.subs(self.eval_point)) * self.positive_functions_sp[index_mul](subexpr)
        else:
            # In this case, some of the variables may be unsed and should be removed.
            # Below we renumber the variables in h_lyap so that they start at 0
            # and there is no unused variable. Moreover degree becomes the new number of variables
            assert self.lyap_cross_term, "Empty Lyapunov function"
            used_vars = []
            for i in range(degree):
                for cross_node in h_lyap:
                    if f"x{i}" in cross_node.infix():
                        used_vars.append(f"x{i}")
            used_vars = sorted(list(set(used_vars)))
            new_degree = len(used_vars)
            assert new_degree != 0, "No variables in the cross terms"
            assert len(h_lyap) == len(p1)
            for i in range(new_degree):
                for j, cross_node in enumerate(h_lyap):
                    cross_node = cross_node.replace(Node(self.variables[used_vars[i]]), Node(self.variables[f"x{i}"]))
                    p1[j] = p1[j].replace(Node(self.variables[used_vars[i]]), Node(self.variables[f"x{i}"]))
                    if not self.lyap_polynomial_V:
                        p2[j] = p2[j].replace(Node(self.variables[used_vars[i]]), Node(self.variables[f"x{i}"]))
            degree = new_degree

        ### Getting the expression in sympy of the cross term

        h_lyap_sp = 0
        for cross_term in h_lyap:
            cross_term_sp = sp.S(str(cross_term), locals=self.local_dict, evaluate=False)
            cross_term_sp_value = cross_term_sp.subs(self.eval_point).evalf()
            if has_inf_nan(cross_term_sp_value):
                if self.lyap_debug:
                    print("Lyapunov function becomes infinite in 0, returning None")
                return "error:f"
            if self.lyap_debug:
                print("cross term", cross_term)
                print("cross term sp", cross_term_sp)
                print("abs cross_term_sp", abs(cross_term_sp_value))
                print(cross_term_sp_value)
            try:
                pos_cond = abs(cross_term_sp_value) >= EPSILON
                change_var = True if pos_cond else False
            except TypeError as e:
                if e.args[0].startswith("cannot determine truth value of Relational"):
                    return "error:sympy_pow"
                else:
                    raise RuntimeError("Unexpected error")
            if change_var:
                cross_term_sp = cross_term_sp - cross_term_sp_value
            assert abs(cross_term_sp.subs(self.eval_point).evalf()) < EPSILON
            if self.rng.rand() < self.lyap_proba_cross_composition:
                index_comp = self.rng.randint(len(self.bounded_functions_sp))
                if self.lyap_debug:
                    print("cross_term_sp", cross_term_sp)
                    print(f"going through composition with {self.bounded_functions_sp[index_comp]}")
                    print(cross_term_sp + self.bounded_functions_sp[index_comp][1])
                cross_term_sp = self.bounded_functions_sp[index_comp][0](cross_term_sp + self.bounded_functions_sp[index_comp][1])
                if self.lyap_debug:
                    print("new cross_term_sp", cross_term_sp)
            else:
                cross_term_sp = cross_term_sp**2
            if self.rng.rand() < self.lyap_proba_cross_multiply:
                logger.warning("Functionality cross_multiply is disabled for now")
            h_lyap_sp += cross_term_sp

        # debug
        if self.lyap_debug:
            print("degree:", degree)
            print("h lyap:", [str(el_cross) for el_cross in h_lyap])
            print("h_lyap_sp:", h_lyap_sp)
            print("f_lyap0:", f_lyap_0)
            print("f_lyap:", f_lyap)
            print("h sp:", [str(el_cross) for el_cross in h_lyap])
            print("f_lyap_tot", f_lyap + h_lyap_sp)
            print("")

        ### Total Lyapunov function

        f_lyap_tot = f_lyap + h_lyap_sp

        if len(f_lyap_tot.free_symbols) == 0 or has_inf_nan(f_lyap_tot) or f_lyap_tot.has(sp.I):
            if self.lyap_debug:
                print("invalid lyapunov function, returning None")
                print(f_lyap_tot)
            return "error:f"
        f_value = f_lyap_tot.subs(self.eval_point).evalf()
        if has_inf_nan(f_value) or f_lyap_tot.has(sp.I):
            if self.lyap_debug:
                print("invalid lyapunov function, returning None")
                print(f_value)
            return "error:f"
        for i in range(degree):
            g_value = simplify(f_lyap_tot.diff(self.variables[f"x{i}"]), 1).subs(self.eval_point).evalf()
            if self.lyap_debug:
                print("g_value", g_value)
                print("f_lyap_tot:", f_lyap_tot)
                print("f_lyap_tot gradient", f_lyap_tot.diff(self.variables[f"x{i}"]))
            if has_inf_nan(g_value) or g_value.has(sp.I):
                return "error:g"
            if np.isnan(float(g_value)):
                if self.lyap_debug:
                    print("nan lyapunov function gradient, returning None, error:g")
                if "sqrt(" in str(f_lyap_tot):
                    return "error:g_sqrt"
                return "error:g"
            if abs(g_value) >= EPSILON:
                if self.lyap_debug:
                    print("f_lyap:", f_lyap_tot)
                    print("g_value :", g_value)
                    print("self.eval_point :", self.eval_point)
                    print("sym expr:", sp.simplify(f_lyap_tot.diff(self.variables[f"x{i}"])))
                    print("Lyapunov function not minimal in evaluation point, returning None, error:g")
                return "error:g"

        ### Gradient of the Lyapunov function and base of the orthogonal hyperplan

        try:
            grad_sp = []
            for i in range(degree):
                g_diff = f_lyap_tot.diff(self.variables[f"x{i}"])
                if self.lyap_polynomial_V:
                    g_diff = sp.expand(g_diff)
                grad_sp.append(g_diff)
        except MyTimeoutError:
            raise
        except Exception as e:
            raise e
        if self.lyap_debug:
            print("gradient")
            for f in grad_sp:
                print(f)
            print("")

        return f_lyap_tot, grad_sp, degree

    def gen_lyap_system(self, grad_sympy, degree) -> Tuple[List, int]:
        """
        Generate a stable system (or possibly unstable if not self.lyap_stable) from the knowledge of \nabla V.
        INPUTS:
            grad_symp: \nabla V in sympy form
            degree: number of equations / variables
        OUTPUTS:
            system: list of equations in sympy format
            stability: whether the system is stable or not
        """
        ### Generate randomly n vectors from the hyperplan
        grad_sp = grad_sympy.copy()
        e = []
        e.append(grad_sp)
        while len(e) < degree:
            i = self.rng.randint(degree)
            j = self.rng.randint(degree)
            sgn = 1 - 2 * self.rng.randint(2)
            assert abs(sgn) == 1
            if i == j:
                continue
            g = [0 for __ in range(degree)]
            g[i] = sgn * grad_sp[j]
            g[j] = (-sgn) * grad_sp[i]
            e.append(g)

        # debug
        if self.lyap_debug:
            print("e:")
            for comp_e in e:
                print(comp_e)
            print("")

        stability = 1 if self.lyap_stable else 2 * self.rng.randint(2) - 1

        # generate functions,
        nb_ops = self.rng.randint(1, self.lyap_nb_ops_proper + 1, size=(2 * degree,))
        h0 = []
        h = []
        i = 0

        """
        Generation of the system
        """

        ###Generation of parameters

        # Number of components of the gradient vector used
        if self.lyap_strict:
            nb_components = degree
        else:
            if self.lyap_gen_weight > 0:
                weights = np.array([np.exp(-self.lyap_gen_weight * i) for i in range(degree)])
                weights = weights / np.sum(weights)
                nb_components = self.rng.choice(range(1, degree + 1), p=weights)
            else:
                nb_components = self.rng.randint(1, degree + 1)

        # Number of vectors to be used
        if self.lyap_gen_weight > 0:
            weights = np.array([np.exp(-self.lyap_gen_weight * i) for i in range(degree)])
            weights = weights / np.sum(weights)
            nb_vectors = self.rng.choice(range(degree), p=weights)
        else:
            nb_vectors = self.rng.randint(degree)

        ### Generation of function coefficients

        # first n components multiply the component of the vector gradient
        # then the following multiply each vector of the orthogonal hyperplan
        while i < nb_vectors + nb_components:
            try:
                if self.lyap_polynomial_H:
                    if self.lyap_pure_polynomial:
                        if i < nb_components:  # first n components are for the first vector
                            # to make sure the first vector is less easily identifiable (since later squared)
                            max_order = self.lyap_max_order_pure_poly
                            max_number = np.floor(np.sqrt(self.max_int))
                        else:
                            max_order = 2 * self.lyap_max_order_pure_poly
                            max_number = self.max_int

                        fi = self.generate_bounded_polynomial(
                            degree, max_order=max_order, max_n_term=nb_ops[i], max_number=max_number, nonzero=False, no_constant=False
                        )
                    else:
                        unaries_poly_sys = ["id", "cos", "sin", "exp"]
                        if i < nb_components:
                            nb_ops_eff = int(np.ceil(np.sqrt(nb_ops[i])))
                        else:
                            nb_ops_eff = nb_ops[i]
                        fi = self.generate_polynomial(nb_ops_eff, 2, degree, unaries_poly_sys, True)
                else:
                    if i < nb_components:
                        nb_ops_eff = int(np.ceil(np.sqrt(nb_ops[i])))
                        fi = self.generate_tree(nb_ops_eff, degree)
                        if "sqrt" in fi.prefix():
                            if self.lyap_debug:
                                print("fi with sqrt before replacement", fi)
                            fi = fi.replace_ops("sqrt", ["Abs", "sqrt"], except_exp=True)
                            if self.lyap_debug:
                                print("fi with sqrt after replacement", fi)
                    else:
                        nb_ops_eff = nb_ops[i]
                        fi = self.generate_tree(nb_ops_eff, degree)
                if self.lyap_debug and i < nb_components:
                    print("Here are the functions that are going to be squared")
                    print(fi)
                fi_sp = sp.S(str(fi), locals=self.local_dict, evaluate=False)
                f = fi_sp
                if len(f.free_symbols) == 0 or has_inf_nan(f) or f.has(sp.I):
                    continue
            except Exception as e:
                logger.error(
                    'An unknown exception of type {0} occurred in line {1} for expression "{2}". Arguments:{3!r}.'.format(
                        type(e).__name__, sys.exc_info()[-1].tb_lineno, fi, e.args
                    )
                )
                continue
            if i < nb_components:  #  first components represent the coefficient of the first vector
                f = -stability * f * f
            h.append(f)
            i += 1
        assert len(h) == nb_components + nb_vectors

        # debug
        if self.lyap_debug:
            print("hi:")
            for comp_h in h:
                print(comp_h)
            print("nb_vectors", nb_vectors)
            print("nb_components", nb_components)
            print("")

        ### Build coefficients for the gradient vectors

        if not self.lyap_generate_gradient_flow:
            order = list(self.rng.permutation(range(degree)))
            for i, index in enumerate(order):
                if i < nb_components:
                    e[0][index] = h[i] * e[0][index]
                else:
                    e[0][index] = 0
            if self.lyap_debug:
                print("after the choice of the components, the gradient contribution to the system is")
                print(e[0])
            # remove the components corresponding to the gradient vectors in h and insert a 1 instead
            # such that h is the vector of coefficient for each vector of the family e we are using
            h = h[nb_components:]
            h.insert(0, sp.S(1))
            assert len(h) == 1 + nb_vectors
        else:
            # Particular case of a gradient flow:
            # The gradient of the Lyapunov function stored in e[0] remains untouched
            # and we only store a -1 in h such that the system will be - \nabla V
            h = [-sp.S(1)]

        ### Build system from coefficients and basis of the hyperplan

        system = []
        try:
            # select randomly the vectors to be used,
            # except the first one that remains always the first one
            order = list(self.rng.permutation(range(1, len(h))))
            order.insert(0, 0)
            if self.lyap_debug:
                print("order system:", order)
            for i in range(degree):
                s = 0
                for j, index in enumerate(order):
                    s += h[j] * e[index][i]
                    if self.lyap_debug:
                        print("h[j]:", h[j])
                        print("e[index][i]:", e[index][i])
                s = simplify(s, 1)
                if self.lyap_pure_polynomial:
                    s = sp.expand(s)
                if self.lyap_debug:
                    print("s:", s)
                system.append(s)
        except Exception:
            raise RuntimeError("Exception when building the system")

        if self.lyap_debug:
            print("system")
            for el in system:
                print(el)
            print("")
            print("Scalar product with the gradient")
            grad0 = grad_sympy.copy()
            s_prod = 0
            for i, el in enumerate(system):
                s_prod += el * grad0[i]
                print(f"Component {i}", sp.simplify(el * grad0[i]))
            print("Total gradient")
            s_prod = sp.simplify(s_prod)
            print(s_prod)
            print("")

        # Check that the system has no forbidden symbols (especially for log(-1), etc.)
        for s in system:
            value_zero_s = s.subs(self.eval_point).evalf()
            if has_inf_nan(s) or has_inf_nan(value_zero_s) or s.has(sp.I):
                if self.lyap_debug:
                    print(f"invalid system equation {s}, returning None")
                return "error:h"
            if float(value_zero_s) >= EPSILON:
                print("here", value_zero_s)
                print(s)
                if self.lyap_debug:
                    print(f"invalid system equation {s}, returning None")
                return "error:h"
        if not self.lyap_polynomial_H and self.lyap_find_domain:
            for s in system:
                node_s = self.prefix_to_node(self.sympy_to_prefix(s))
                domain_s = node_s.domain()
                if len(domain_s) > 0:
                    # If some constraints, then check the gradient is well defined at the equilibrium
                    # Another way could be to check that the equilibrium is inside the whole domain
                    for i in range(degree):
                        grad_s = simplify(s.diff(self.variables[f"x{i}"]), 1)
                        value_zero_grad_s = grad_s.subs(self.eval_point).evalf()
                        if (
                            has_inf_nan(grad_s)
                            or has_inf_nan(value_zero_grad_s)
                            or grad_s.has(sp.I)
                            or value_zero_grad_s.has(sp.I)
                            or np.isnan(float(value_zero_grad_s))
                        ):
                            if self.lyap_debug:
                                print("value_zero_grad_s", value_zero_grad_s)
                                print(f"invalid system gradient in equation {s}, with evaluation point {self.eval_point} returning None")
                            return "error:hg"
        return system, stability

    @timeout(30)
    def gen_lyapunov(self):
        """
        Generate Lyapunov function, get its gradient, build a base of the orthogonal hyperplan, generate problem.
        In backward mode the Lyapunov function will be created following the formula in Appendix B Step 1c of the paper (see https://openreview.net/pdf?id=kOMrm4ZJ3m)
        The system is then generated such that \nabla f\cdot V < 0.
        Finally the system and Lyapunov functions are encoded.
        """
        # If generation is using a forward SOS prover
        if self.lyap_SOS_fwd_gen or self.rng.rand() < self.lyap_proba_fwd_gen:
            return self.gen_SOS_fwd()

        # If the generation is backward
        if self.lyap_memory_nb > 0:
            # If lyap_multigen is enabled and a Lyapunov function is currently kept in memory:
            assert self.lyap_memory_fun is not None
            f_lyap_tot, grad_sp, degree = self.lyap_memory_fun
            self.lyap_memory_nb -= 1
        else:
            # If no Lyapunov function kept in memory
            degree = self.rng.randint(self.min_degree, self.max_degree + 1)
            maybe_lyap = self.gen_lyap_fun(degree)
            if isinstance(maybe_lyap, str):
                return None
            f_lyap_tot, grad_sp, degree = maybe_lyap
            if self.lyap_multigen > 0:
                self.lyap_memory_nb = self.rng.randint(
                    0, self.lyap_multigen
                )  # parameters is at most self.lyap_multigen - 1 and so generator will generate at most self.lyap_multigen identical Lyapunov function.
                self.lyap_memory_fun = (f_lyap_tot.copy(), grad_sp, degree)

        # Generate the system
        maybe_system = self.gen_lyap_system(grad_sympy=grad_sp, degree=degree)
        if isinstance(maybe_system, str):
            return None
        system, stability = maybe_system

        ### Encoding the system (input) and the coefficients of the Lyapunov function (output) in tokens

        res_gen = self.sympy_to_encoding(system=system, lyap_function=f_lyap_tot, stability=stability)
        return res_gen

    def top_test(self, tgt, hyp):
        ny = hyp
        nt = tgt
        try:
            if self.lyap_predict_stability:
                v1, pos = self.parse_int(hyp)
                ny = hyp[pos + 1 :]
                v2, pos = self.parse_int(tgt)
                if np.isnan(v1) or np.isnan(v2):
                    return False
                if v1 != v2:
                    return False
                nt = tgt[pos + 1 :]
            pti = self.prefix_to_infix(ny)
            if pti is None:
                return False
            b, ny = pti
            s = sp.S(str(b), locals=self.local_dict, evaluate=False)
            sps = s  # first term is the crossed term #Edit: crossed term is now squared like the others
            while len(ny) > 0:
                pti = self.prefix_to_infix(ny[1:])
                if pti is None:
                    return False
                b, ny = pti
                s = sp.S(str(b), locals=self.local_dict, evaluate=False)
                sps = sps + s
            pti = self.prefix_to_infix(nt)
            if pti is None:
                return False
            b, nt = pti
            s = sp.S(str(b), locals=self.local_dict, evaluate=False)
            sps = sps - s  # first term is the crossed term  #Edit: crossed term is now squared like the others
            while len(nt) > 0:
                pti = self.prefix_to_infix(nt[1:])
                if pti is None:
                    return False
                b, nt = pti
                s = sp.S(str(b), locals=self.local_dict, evaluate=False)
                sps = sps - s
            sps = simplify(sps, 1)
        except MyTimeoutError:
            logger.info("TimeoutError in top_test")
            return False
        except Exception as e:
            logger.info(f"Exception {e} in top_test")
            return False
        return sps == 0

    @timeout(1200)
    def check_lyap_validity(self, src, hyp, tgt, check_components_minimum: bool = True):
        """
        Checks the validity of a Lyapunov function given a system.
        INPUTS:
            - encoded system
            - encoded lyapunov function predicted
            - encoded reference Lyapunov function in the dataset (only needed for a first comparison to stop the test early if equal to predicted)

        OUTPUTS:
            1: valid
            0: invalid
            -1: optim error
            -2: incorrect hyp
            -3: timeout
            -4: other error
            -5: input error (wtf)
        """
        # try:
        debug = self.lyap_debug
        if self.top_test(tgt, hyp) is True:
            # if predicted Lyapunov function equals reference Lyapunov function to this one, the test doesn't go further and check is valid
            is_valid = 1
        else:
            stability = 0  # 0 unknown, 1 stable, -1 unstable
            ny = hyp
            if self.lyap_stable:
                stability = 1
            elif self.lyap_predict_stability:
                stability, pos = self.parse_int(hyp)
                if np.isnan(stability):
                    return -2
                ny = hyp[pos + 1 :]
            y_infix = []
            pti = self.prefix_to_infix(ny)
            if pti is None:
                return -2
            b, ny = pti
            s = sp.S(str(b), locals=self.local_dict, evaluate=False)
            value_zero = s.subs(self.eval_point).evalf()
            y_infix.append(s)  # First term is crossed term, but now treated as others
            while len(ny) > 0:
                pti = self.prefix_to_infix(ny[1:])
                if pti is None:
                    return -2
                b, ny = pti
                s = sp.S(str(b), locals=self.local_dict, evaluate=False)
                y_infix.append(s)
            degree, pos_src = self.parse_int(src)
            assert len(y_infix) == 1, ([str(el) for el in y_infix], len(y_infix))
            if np.isnan(degree):
                return -5
            nx = src[pos_src:]

            # Total lyapunov gradient

            grad = []
            assert len(y_infix) == 1
            y = y_infix[0]
            for i in range(degree):
                dy = y.diff(self.variables[f"x{i}"])
                grad.append(dy)

            # check that the gradient is 0 in 0
            for grad_el in grad:
                grad_zero = sp.simplify(grad_el).subs(self.eval_point).evalf()
                if abs(grad_zero) > EPSILON:
                    if self.lyap_debug:
                        print(grad_zero, grad_el)
                    return -2

            pp = np.zeros(degree, dtype=float) + [
                0.01 * self.rng.rand() for _ in range(degree)
            ]  # Reduce chances of aiming directly as the 0 of a function.

            domain_V = []
            if not self.lyap_polynomial_H and self.lyap_find_domain:
                domain_V.extend(self.prefix_to_node(self.sympy_to_prefix(y)).domain())

            # To be fully rigorous, in the general case we should check that the Lyapunov function is
            # minimum in 0. This requires another call to test_V_positive which is the bottleneck so it will not be done if check_components_minimum is False.
            # We recommand to keep this option for test time.
            if check_components_minimum:
                if self.lyap_debug:
                    print("Checking that V is minimum in 0")
                rescale_lyap = y - value_zero
                if self.lyap_SOS_checker:
                    try:
                        check_sos = SOS_checker(rescale_lyap)
                    except MyTimeoutError:
                        return -3
                    if check_sos is None:
                        return -1
                    if check_sos is False:
                        return 0
                else:
                    min_value = test_V_positive(rescale_lyap, pp, domain_V, debug=debug)
                    if min_value != 1:
                        return min_value

            # read system and dot product
            dot = 0
            i = 0
            total_domain = []
            while len(nx) > 0:
                pti = self.prefix_to_infix(nx[1:])
                if pti is None:
                    return -5
                b, nx = pti
                s = sp.S(str(b), locals=self.local_dict, evaluate=False)
                if not self.lyap_polynomial_H and self.lyap_find_domain:
                    total_domain.extend(self.prefix_to_node(self.sympy_to_prefix(s)).domain())
                dot += s * grad[i]
                if self.lyap_debug:
                    print(f"dot after component {i}")
                    print(dot)
                i += 1
            dot = sp.simplify(dot)
            # check sign
            if stability == 0:  # determine sign at some point
                point = OrderedDict({self.variables[f"x{i}"]: 0.01 for i in range(self.max_degree)})
                f = dot.subs(point).evalf()
                if f < -EPSILON:
                    stability = 1
                if f > EPSILON:
                    stability = -1
            if stability == 0:
                is_valid = 0
            else:
                if stability == 1:
                    dot = -dot
                if self.lyap_SOS_checker:
                    try:
                        check_sos = SOS_checker(dot)
                    except MyTimeoutError:
                        return -3
                    if check_sos is None:
                        c = -1
                    elif check_sos is False:
                        c = 0
                    else:
                        c = 1
                else:
                    if self.lyap_debug:
                        print("Checking dV/dt")
                    c = test_V_positive(dot, pp, total_domain, debug=debug)

                is_valid = c
        return is_valid

    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(self, task, train=True, params=params, path=(None if data_path is None else data_path[task][0]))
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 86400),
            batch_size=params.batch_size,
            num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(self, data_type, task, data_path, data_path_idx, batch_size, params, size):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"]
        assert (data_type == "valid") == (data_path_idx == 1)
        assert (data_type == "test") == (data_path_idx >= 2)
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(self, task, train=False, params=params, path=(None if data_path is None else data_path[task][data_path_idx]), size=size)
        return DataLoader(dataset, timeout=0, batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=dataset.collate_fn)

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument("--int_base", type=int, default=1000, help="Encoding base for integers")
        parser.add_argument("--max_int", type=int, default=10, help="Maximum integer value")
        parser.add_argument("--precision", type=int, default=4, help="Float numbers precision")
        parser.add_argument("--positive", type=bool_flag, default=False, help="Do not sample negative numbers")
        parser.add_argument("--nonnull", type=bool_flag, default=True, help="Do not sample zeros")

        parser.add_argument("--prob_int", type=float, default=0.3, help="Probability of int vs variables")
        parser.add_argument("--min_degree", type=int, default=2, help="Minimum degree of ode / nb of variables")
        parser.add_argument("--max_degree", type=int, default=6, help="Maximum degree of ode / nb of variables")

        parser.add_argument(
            "--custom_unary_probs",
            type=bool_flag,
            default=False,
            help="generate_tree (used in the generation of a system when the system is not required to have a special form or be polynomial) use a custom set of unaries",
        )
        parser.add_argument("--prob_trigs", type=float, default=0.2, help="Probability of trig operators")
        parser.add_argument("--prob_arc_trigs", type=float, default=0.2, help="Probability of inverse trig operators")
        parser.add_argument("--prob_logs", type=float, default=0.2, help="Probability of logarithm and exponential operators")

        parser.add_argument("--lyap_max_degree", type=int, default=3, help="Maximum degree of lyapunov polynomial terms")
        parser.add_argument("--lyap_n_terms", type=int, default=4, help="Maximum number of terms in lyapunov polynomials")
        parser.add_argument("--eval_value", type=float, default=0.0, help="Evaluation point for all variables")
        parser.add_argument("--skip_zero_gradient", type=bool_flag, default=True, help="No gradient can be zero at evaluation point")
        parser.add_argument("--lyap_nb_ops_proper", type=int, default=2, help="Max number of operators for the proper term")
        parser.add_argument("--lyap_nb_ops_lyap", type=int, default=6, help="Max number of operators for the lyap term")
        parser.add_argument("--operators_lyap", type=Optional[List[str]], default=None, help="operators to be considered for Lyapunov tasks")

        parser.add_argument(
            "--lyap_polynomial_V",
            type=bool_flag,
            default=True,
            help="Lyapunov function is a polynomial potentially of more complicated operators. To be deprecated",
        )
        parser.add_argument(
            "--lyap_polynomial_H",
            type=bool_flag,
            default=True,
            help="The h used to generate the system with the gradient of the Lyapunov function are polynomes of unary operators (which may be non-polynomial themselves)",
        )
        parser.add_argument(
            "--lyap_basic_functions_num",
            type=bool_flag,
            default=True,
            help="Numerators of the cross term of the Lyapunov function are polynomial with potentially complicated operators",
        )
        parser.add_argument("--lyap_basic_functions_den", type=bool_flag, default=True, help="Denominators of the Lyapunov function are polynomials")
        parser.add_argument("--lyap_debug", type=bool_flag, default=False, help="print generated systems")
        parser.add_argument("--lyap_stable", type=bool_flag, default=True, help="only generate stable systems")
        parser.add_argument("--lyap_predict_stability", type=bool_flag, default=False, help="stability in output")
        parser.add_argument("--lyap_drop_last_equation", type=bool_flag, default=False, help="remove one equation in input")
        parser.add_argument(
            "--lyap_pure_polynomial",
            type=bool_flag,
            default=False,
            help="The h used to generate the system with the gradient of the Lyapunov function is purely polynomial",
        )
        parser.add_argument("--lyap_cross_term", type=bool_flag, default=True, help="cross term in lyapunov function")
        parser.add_argument("--lyap_max_nb_cross_term", type=int, default=2, help="cross term in lyapunov function")
        parser.add_argument(
            "--lyap_proba_diagonal", type=float, default=0.5, help="probability of the proper term of the Lyapunov function to be diagonal."
        )
        parser.add_argument(
            "--lyap_proba_proper_composition",
            type=float,
            default=0,
            help="probability of composing the proper term of the Lyapunov function by an increasing function. E.g. x^2 -> e^(x^2)",
        )
        parser.add_argument(
            "--lyap_proba_proper_multiply",
            type=float,
            default=0,
            help="probability of multiplying the proper term of the Lyapunov function by a positive function. E.g. |x|^2 -> (|x|^2)e^(x_1^2)",
        )
        parser.add_argument(
            "--lyap_proba_cross_composition",
            type=float,
            default=0,
            help="probability of composing the crossed term of the Lyapunov function by a bounded function instead of squaring it. E.g. x1*x2 -> cos(x1*x2+\pi)",
        )
        parser.add_argument(
            "--lyap_proba_cross_multiply",
            type=float,
            default=0,
            help="probability of multiplying the proper term of the Lyapunov function by a positive function. E.g. |x|^2 -> (|x|^2)e^(x_1)",
        )
        parser.add_argument("--lyap_only_2_norm", type=bool_flag, default=False, help="Lyapunov function is the 2-norm")
        parser.add_argument("--lyap_strict", type=bool_flag, default=False, help="The lyapunov function is strict (strict stability)")
        parser.add_argument("--lyap_proper", type=bool_flag, default=True, help="The lyapunov function is proper (i.e. goes to infinity at infinity)")
        parser.add_argument("--lyap_local", type=bool_flag, default=False, help="task is to predict local stability and not global stability")
        parser.add_argument(
            "--lyap_float_resolution_poly",
            type=int,
            default=1,
            help="float resolution of the polynomials generated by generate_bounded_polynomial, e.g. 100 corresponds to two significant digits. Default value is 1 and corresponds to integers.",
        )
        parser.add_argument(
            "--lyap_generate_gradient_flow",
            type=bool_flag,
            default=False,
            help="When set to True, the backward generation (see gen_Lyapunov) only generates gradient flows, a special case of stable dynamics",
        )
        parser.add_argument(
            "--lyap_gen_weight",
            type=float,
            default=1.5,
            help="Weight that biais the way the number of components of the system is selected. When 0, the number of components (nb_components and nb_vectors) are selected uniformly at random in [1,degree+1] and [0,degree] respectively. The higher the weight the more smaller number of components if favored",
        )
        parser.add_argument(
            "--lyap_max_order_pure_poly",
            type=int,
            default=2,
            help="Half of the max order of the components of the system that will be multiplied by components of the gradient, when using generate_bounded_polynomial",
        )
        parser.add_argument("--lyap_max_n_term_fwd", type=int, default=5, help="Maximal number of terms in each equations in the fwd generation")
        parser.add_argument(
            "--lyap_generate_sample_fwd",
            type=bool_flag,
            default=False,
            help="When True, generate_sample generates a system with a forward method and an empty Lyapunov function. Can only be used for (forward) generation, not for training.",
        )
        parser.add_argument(
            "--lyap_find_domain",
            type=bool_flag,
            default=False,
            help="When True and not self.lyap_polynomial_H, we keep track of the domain of validity of the system and the Lyapunov function, including for the checks",
        )
        parser.add_argument("--lyap_SOS_checker", type=bool_flag, default=False, help="When True the checker becomes a SOS (Sum Of Squares) checker")
        parser.add_argument(
            "--lyap_SOS_fwd_gen",
            type=bool_flag,
            default=False,
            help="When True the generation is forward and uses a SOS checker and rejection sampling. Caution: slow",
        )
        parser.add_argument(
            "--lyap_proba_fwd_gen",
            type=float,
            default=0.0,
            help="When > 0 then at each call of the generator we generate a forward example with probability lyap_SOS_gen_fwd. Caution: slow if probability is high",
        )
        parser.add_argument(
            "--lyap_proper_fwd",
            type=bool_flag,
            default=False,
            help="When > 0 then at each call of the generator we generate a forward example with probability lyap_SOS_gen_fwd. Caution: slow if probability is high",
        )
        parser.add_argument(
            "--lyap_multigen", type=int, default=0, help="When >0, each call to (generate) generates several examples with the same Lyapunov function"
        )
        parser.add_argument(
            "--prob_positive", type=float, default=-1.0, help="Proportion of positive convergence speed (for all degrees, -1.0 = no control)"
        )


class EnvDataset(Dataset):
    """
    Class for dataset environment used for generation and training
    """

    def __init__(self, env, task, train, params, path, size=None):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        assert task in ODEEnvironment.TRAINING_TASKS
        assert size is None or not self.train

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path)
            logger.info(f"Loading data from {path} ...")
            with io.open(path, mode="r", encoding="utf-8") as f:
                # either reload the entire file, or the first N lines (for the training set)
                if not train:
                    lines = [line.rstrip().split("|") for line in f]
                else:
                    lines = []
                    for i, line in enumerate(f):
                        if i == params.reload_size:
                            break
                        if i % params.n_gpu_per_node == params.local_rank:
                            lines.append(line.rstrip().split("|"))
            self.data = [xy.split("\t") for _, xy in lines]
            self.data = [xy for xy in self.data if len(xy) == 2]
            logger.info(f"Loaded {len(self.data)} equations from the disk.")

            if params.lyap_drop_last_equation and self.train:
                self.data = [(x[: last_index(x, env.func_separator)], y) for (x, y) in self.data]

        # dataset size: infinite iterator for train, finite for valid / test (default of 5000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 5000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        nb_eqs = [seq.count(self.env.func_separator) for seq in x]
        x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_eqs)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.env.rng = np.random.RandomState([worker_id, self.global_rank, self.env_base_seed])
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed})."
            )
        else:
            self.env.rng = np.random.RandomState(0)
            logger.info(f"Initialized a random generator with seed 0")

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        while True:
            if self.train:
                index = self.env.rng.randint(len(self.data))
            x, y = self.data[index]
            x = x.split()
            y = y.split()
            if (self.env.max_len > 0 and len(x) >= self.env.max_len) or (self.env.max_output_len > 0 and len(y) >= self.env.max_output_len):
                index += 1
                continue
            return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:
            try:
                if self.env.lyap_generate_sample_fwd:
                    if self.env.lyap_pure_polynomial:
                        x = self.env.generate_polynomial_system()
                    else:
                        x = self.env.generate_random_system()
                    if x is None or isinstance(x, str):
                        continue
                    xy = x, []
                else:
                    xy = self.env.gen_lyapunov()
                    if xy is None or isinstance(xy, str):
                        continue
                x, y = xy
                break
            except MyTimeoutError:
                continue
            except Exception as e:
                logger.error(
                    'An unknown exception of type {0} occurred for worker {4} in line {1} for expression "{2}". Arguments:{3!r}.'.format(
                        type(e).__name__, sys.exc_info()[-1].tb_lineno, "F", e.args, self.get_worker_id()
                    )
                )
                continue
        self.count += 1

        # clear SymPy cache periodically
        if CLEAR_SYMPY_CACHE_FREQ > 0 and self.count % CLEAR_SYMPY_CACHE_FREQ == 0:
            logger.warning(f"Clearing SymPy cache (worker {self.get_worker_id()})")
            clear_cache()

        return x, y
