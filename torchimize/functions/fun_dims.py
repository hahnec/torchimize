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
from typing import Callable, Union, Tuple, List

from torchimize.functions.jacobian import jacobian_approx_t


def test_fun_dims_parallel(        
        p: torch.Tensor,
        function: Callable, 
        jac_function: Callable = None,
        args: Union[Tuple, List] = (),
        wvec: torch.Tensor = None,
    ) -> bool:
    """
    Helper function that tests whether dimensionality of output tensors suits the herein provided optimization functions.

    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
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

    f = fun(p)
    assert len(f.shape) == 3, 'residual tensor is supposed to have 3 dims, but has %s' % str(len(f.shape))

    j = jac_fun(p)
    assert len(j.shape) == 4, 'jacobian tensor is supposed to have 4 dims, but has %s' % str(len(j.shape))

    # use weights of ones as default
    wvec = torch.ones(f.shape[1], dtype=p.dtype, device=p.device, requires_grad=False) if wvec is None else wvec
    assert len(wvec) == f.shape[1], 'weights vector length of %s is supposed to match cost number which is %s' % (str(len(wvec)), str(f.shape[1]))

    return True
