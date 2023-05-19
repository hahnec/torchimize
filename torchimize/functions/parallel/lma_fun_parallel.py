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
from typing import Union, Callable, List, Tuple

from torchimize.functions.jacobian import jacobian_approx_t
from torchimize.functions.parallel.newton_parallel import newton_step_parallel


def lsq_lma_parallel(
        p: torch.Tensor,
        function: Callable, 
        jac_function: Callable = None,
        args: Union[Tuple, List] = (),
        wvec: torch.Tensor = None,
        ftol: float = 1e-8,
        ptol: float = 1e-8,
        gtol: float = 1e-8,
        tau: float = 1e-3, 
        meth: str = 'lev',
        rho1: float = .25, 
        rho2: float = .75, 
        beta: float = 2, 
        gama: float = 3, 
        max_iter: int = 100, 
    ) -> List[torch.Tensor]:
    """
    Levenberg-Marquardt implementation for parallel least-squares fitting of non-linear functions
    
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param wvec: weights vector used in reduction of multiple costs
    :param ftol: relative change in cost function as stop condition
    :param ptol: relative change in independant variables as stop condition
    :param gtol: maximum gradient tolerance as stop condition
    :param tau: factor to initialize damping parameter
    :param meth: method which is default 'lev' for Levenberg and otherwise Marquardt
    :param rho1: first gain factor threshold for damping parameter adjustment for Marquardt
    :param rho2: second gain factor threshold for damping parameter adjustment for Marquardt
    :param beta: multiplier for damping parameter adjustment for Marquardt
    :param gama: divisor for damping parameter adjustment for Marquardt
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
    D = torch.eye(p.shape[-1], dtype=p.dtype, device=p.device)[None, ...].repeat(p.shape[0], 1, 1)
    u = tau * torch.max(torch.diagonal(D, dim1=-2, dim2=-1), 1)[0]
    sinf = torch.tensor([-torch.inf, torch.inf], dtype=p.dtype, device=p.device)
    ones = torch.ones(p.shape[0], dtype=p.dtype, device=p.device)
    v = 2*ones

    if meth == 'lev':
        lm_uv_step = lambda rho, u, v: levenberg_uv(rho, u, v, ones=ones)
        lm_dg_step = lambda H, D: D * torch.ones_like(H)
    else:
        lm_uv_step = lambda rho, u, v=None: marquardt_uv(rho, u, v, rho1=rho1, rho2=rho2, beta=beta, gama=gama)
        lm_dg_step = lambda H, D: D * torch.max(torch.maximum(H.diagonal(dim1=2), D.diagonal(dim1=2)), dim=1)[0][..., None, None]

    p_list = []
    f_prev = torch.zeros(1, device=p.device, dtype=p.dtype)
    while len(p_list) < max_iter:

        # levenberg-marquardt step
        p, f, g, H = newton_step_parallel(p, fun, jac_fun, wvec)
        D = lm_dg_step(H, D)
        Hu = H+u[:, None, None]*D
        h = -torch.linalg.lstsq(Hu.double(), g.double(), rcond=None, driver=None)[0].to(dtype=p.dtype)
        f_h = fun(p+h)
        rho_nom = torch.einsum('bcp,bci->bc', f, f).sum(1) - torch.einsum('bcp,bci->bc', f_h, f_h).sum(1)
        rho_denom = torch.einsum('bnp,bni->bi', h[..., None], (u[:, None]*h-g)[..., None])[..., 0]
        rho = rho_nom / rho_denom
        rho[rho_denom==0] = sinf[(rho_nom > 0).type(torch.int64)][rho_denom==0]
        u, v = lm_uv_step(rho, u, v)
        p[rho>0, ...] += h[rho>0, ...]
        p_list.append(p.clone())

        # stop conditions
        gcon = torch.max(abs(g)) < gtol
        pcon = (h**2).sum()**.5 < ptol*(ptol + (p**2).sum()**.5)
        fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum() if (rho > 0).sum() > 0 and f_prev.shape == f.shape else False
        f_prev = f.clone()

        if gcon or pcon or fcon:
            break

    return p_list


def levenberg_uv(
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        ones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    u[rho>0] *= torch.maximum(ones/3, 1-(2*rho-1)**3)[rho>0]
    u[rho<0] *= v[rho<0]
    v[rho>0] = 2*ones[rho>0]
    v[rho<0] *= 2

    return u, v


def marquardt_uv(
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        rho1: float,
        rho2: float,
        beta: float,
        gama: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    u[rho < rho1] *= beta
    u[rho > rho2] /= gama

    return u, v
