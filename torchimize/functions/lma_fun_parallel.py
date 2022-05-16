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


def lsq_lma_parallel(
        p: torch.Tensor,
        function: Callable, 
        jac_function: Callable = None, 
        args: Union[Tuple, List] = (), 
        ftol: float = 1e-8,
        ptol: float = 1e-8,
        gtol: float = 1e-8,
        tau: float = 1e-3, 
        meth: str = 'lev',
        rho1: float = .25, 
        rho2: float = .75, 
        bet: float = 2, 
        gam: float = 3, 
        max_iter: int = 100, 
    ) -> List[torch.Tensor]:
    """
    Levenberg-Marquardt implementation for parallel least-squares fitting of non-linear functions
    
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param ftol: relative change in cost function as stop condition
    :param ptol: relative change in independant variables as stop condition
    :param gtol: maximum gradient tolerance as stop condition
    :param tau: factor to initialize damping parameter
    :param meth: method which is default 'lev' for Levenberg and otherwise Marquardt
    :param rho1: first gain factor threshold for damping parameter adjustment for Marquardt
    :param rho2: second gain factor threshold for damping parameter adjustment for Marquardt
    :param bet: multiplier for damping parameter adjustment for Marquardt
    :param gam: divisor for damping parameter adjustment for Marquardt
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

    f = fun(p)
    j = jac_fun(p)
    g = torch.bmm(j.transpose(-2, -1), f[..., None])[..., 0]
    H = torch.bmm(j.transpose(-2, -1), j)
    D = torch.eye(j.shape[-1], dtype=p.dtype, device=j.device)[None, ...].repeat(j.shape[0], 1, 1)
    u = tau * torch.max(torch.diagonal(H, dim1=-2, dim2=-1), 1)[0]
    sinf = torch.tensor([-torch.inf, torch.inf], dtype=p.dtype, device=p.device)
    ones = torch.ones(p.shape[0], dtype=p.dtype, device=p.device)
    v = 2*ones
    f_prev = f.clone()
    p_list = [p.clone().detach()]
    while len(p_list) < max_iter:
        D *= torch.ones_like(H) if meth == 'lev' else torch.max(torch.maximum(H.diagonal(dim1=2), D.diagonal(dim1=2)), dim=1)[0][..., None, None]

        h = -torch.linalg.lstsq(H+u[:, None, None]*D, g, rcond=None, driver=None)[0]
        f_h = fun(p+h)
        rho_denom = torch.bmm(h[:, None, :], (u[:, None]*h-g)[:, :, None])[:, 0, 0]
        rho_nom = torch.bmm(f[:, None, :], f[..., None]).flatten() - torch.bmm(f_h[:, None, :], f_h[..., None]).flatten()
        rho = torch.zeros(p.shape[0], dtype=p.dtype, device=p.device)
        rho[rho_denom>0] = (rho_nom / rho_denom)[rho_denom>0]
        rho[rho_denom<0] = sinf[(rho_nom > 0).type(torch.int64)][rho_denom<0]
        p[rho>0, ...] = p[rho>0, ...] + h[rho>0, ...]
        j[rho>0, ...] = jac_fun(p)[rho>0, ...]
        g[rho>0, ...] = torch.bmm(j.transpose(-2, -1), f[..., None])[rho>0, :, 0]
        H[rho>0, ...] = torch.bmm(j.transpose(-2, -1), j)[rho>0, ...]
        p_list.append(p.clone().detach())
        f_prev = f.clone()
        f = fun(p)
        if meth == 'lev':
            u[rho>0] *= torch.maximum(ones/3, 1-(2*rho-1)**3)[rho>0]
            u[rho<0] *= v[rho<0]
            v[rho>0] = 2*ones[rho>0]
            v[rho<0] *= 2
        else:
            u[rho < rho1] *= bet
            u[rho > rho2] /= gam

        # stop conditions
        gcon = torch.max(abs(g)) < gtol
        pcon = (h**2).sum()**.5 < ptol*(ptol + (p**2).sum()**.5)
        fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum() if (rho > 0).sum() > 0 else False
        if gcon or pcon or fcon:
            break
    
    [print(p[0].cpu().detach()) for p in p_list]

    return p_list
