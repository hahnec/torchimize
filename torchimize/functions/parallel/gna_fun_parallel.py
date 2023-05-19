__author__ = "Christopher Hahne"
__email__ = "inbox@christopherhahne.de"
__license__ = """
    Copyright (c) 2022 Christopher Hahne <inbox@christopherhahne.de>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import torch
from typing import Union, Callable, Tuple, List
import warnings

from torchimize.functions.jacobian import jacobian_approx_t
from torchimize.functions.parallel.newton_parallel import newton_step_parallel


def lsq_gna_parallel(
        p: torch.Tensor,
        function: Callable, 
        jac_function: Callable = None,
        args: Union[Tuple, List] = (),
        wvec: torch.Tensor = None,
        ftol: float = 1e-8,
        ptol: float = 1e-8,
        gtol: float = 1e-8,
        l: float = 1.,
        max_iter: int = 100,
    ) -> List[torch.Tensor]:
    """
    Gauss-Newton implementation for parallel least-squares fitting of non-linear functions with conditions.

    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param wvec: weights vector used in reduction of multiple costs
    :param ftol: relative change in cost function as stop condition
    :param ptol: relative change in independant variables as stop condition
    :param gtol: maximum gradient tolerance as stop condition
    :param l: step size damping parameter
    :param max_iter: maximum number of iterations
    :return: list of results
    """

    if len(args) > 0:
        # pass optional arguments to function
        fun = lambda p: function(p, *args)
    else:
        fun = function

    if jac_function is None:
        # use numerical Jacobian if analytical is not provided
        jac_fun = lambda p: jacobian_approx_t(p, f=fun)
    else:
        jac_fun = lambda p: jac_function(p, *args)

    assert len(p.shape) == 2, 'parameter tensor is supposed to have 2 dims, but has %s' % str(len(p.shape))

    wvec = torch.ones(1, device=p.device, dtype=p.dtype) if wvec is None else wvec

    p_list = []
    f_prev = torch.zeros(1, device=p.device, dtype=p.dtype)
    while len(p_list) < max_iter:
        p, f, g, h, H = gauss_newton_step(p, fun, jac_fun, wvec, l)
        p_list.append(p.clone())

        if torch.linalg.cond(H).isinf().sum() > 0: 
            warnings.warn('Condition number of at least one Hessian is infinite giving NaNs. Consider a step rate l<1.')

        # stop conditions
        gcon = torch.max(abs(g)) < gtol
        pcon = (h**2).sum()**.5 < ptol*(ptol + (p**2).sum()**.5)
        fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum() if f_prev.shape == f.shape else False
        f_prev = f.clone()
        
        if gcon or pcon or fcon:
            break

    return p_list


def lsq_gna_parallel_plain(
        p: torch.Tensor,
        function: Callable, 
        jac_function: Callable,
        wvec: torch.Tensor,
        l: float = 1.,
        max_iter: int = 100,
    ) -> torch.Tensor:
    """
    Gauss-Newton implementation for parallel least-squares fitting of non-linear functions without conditions.

    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param wvec: weights vector used in reduction of multiple costs
    :param l: step size damping parameter
    :param max_iter: maximum number of iterations
    :return: result
    """

    for _ in range(max_iter):
        p = gauss_newton_step(p, function=function, jac_function=jac_function, wvec=wvec, l=l)[0]

    return p


def gauss_newton_step(
        p: torch.Tensor,
        function: Callable,
        jac_function: Callable,
        wvec: torch.Tensor,
        l: float = 1.,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gauss-Newton step function for parallel least-squares fitting of non-linear functions

    :param p: current guess
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param wvec: weights vector used in reduction of multiple costs
    :param l: step size damping parameter
    :return: list of results
    """

    p, f, g, H = newton_step_parallel(p, function, jac_function, wvec)
    h = torch.linalg.lstsq(H.double(), g.double(), rcond=None, driver=None)[0].to(dtype=p.dtype)
    p -= l*h

    return p, f, g, h, H
