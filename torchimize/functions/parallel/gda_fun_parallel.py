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

from torchimize.functions.jacobian import jacobian_approx_t


def gradient_descent_parallel(
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
    Gradient Descent implementation for parallel least-squares fitting of non-linear functions with conditions.

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
        pn, f, h = newton_raphson_step(p, fun, jac_fun, wvec, l)
        g = h/l

        # batched stop conditions
        gcon = torch.max(abs(g), dim=-1)[0] < gtol
        pcon = (h**2).sum(-1)**.5 < ptol*(ptol + (p**2).sum(-1)**.5)
        fcon = ((f_prev-f)**2).sum((-1,-2)) < ((ftol*f)**2).sum((-1,-2)) if f_prev.shape == f.shape else torch.zeros_like(gcon)
        f_prev = f.clone()

        # update only parameters, which have not converged yet
        converged = gcon | pcon | fcon
        p[~converged] = pn[~converged]
        p_list.append(p.clone())
        
        if converged.all():
            break

    return p_list


def gradient_descent_parallel_plain(
        p: torch.Tensor,
        function: Callable, 
        jac_function: Callable,
        wvec: torch.Tensor,
        l: float = 1.,
        max_iter: int = 100,
    ) -> torch.Tensor:
    """
    Gradient Descent implementation for parallel least-squares fitting of non-linear functions without conditions.

    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param wvec: weights vector used in reduction of multiple costs
    :param l: step size damping parameter
    :param max_iter: maximum number of iterations
    :return: result
    """

    for _ in range(max_iter):
        p = newton_raphson_step(p, function=function, jac_function=jac_function, wvec=wvec, l=l)[0]

    return p


def newton_raphson_step(
        p: torch.Tensor,
        function: Callable,
        jac_function: Callable,
        wvec: torch.Tensor,
        l: float = 1.,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gradient Descent step function for parallel least-squares fitting of non-linear functions

    :param p: current guess
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param wvec: weights vector used in reduction of multiple costs
    :param l: step size damping parameter
    :return: tuple of results
    """

    fc = function(p)
    jc = jac_function(p)
    f = torch.einsum('bcp,c->bp', fc, wvec)
    j = torch.einsum('bcpi,c->bpi', jc, wvec)
    try:
        h = torch.linalg.lstsq(j.double(), f.double(), rcond=None, driver=None)[0].to(dtype=p.dtype)
    except torch._C._LinAlgError:
        jmin_rank = min(j.shape[1:])
        rank_mask = torch.linalg.matrix_rank(j) < jmin_rank
        j[rank_mask] = torch.eye(*j.shape[1:], dtype=j.dtype, device=j.device)
        h = torch.linalg.lstsq(j.double(), f.double(), rcond=None, driver=None)[0].to(dtype=p.dtype)

    p -= l*h

    return p, f, h
