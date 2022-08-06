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
from typing import Callable, Tuple


def newton_step_parallel(        
        p: torch.Tensor,
        function: Callable,
        jac_function: Callable,
        wvec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    Newton second order step function for parallel least-squares fitting of non-linear functions

    :param p: current guess
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param wvec: weights vector used in reduction of multiple costs
    :return: list of results
    """

    f = function(p)
    j = jac_function(p)
    gc = torch.einsum('bcnp,bcnp->bcp', j, f[..., None])
    Hc = torch.einsum('bcnp,bcni->bcpi', j, j)
    g = torch.einsum('bcp,c->bp', gc, wvec)
    H = torch.einsum('bcpi,c->bpi', Hc, wvec)

    return p, f, g, H
