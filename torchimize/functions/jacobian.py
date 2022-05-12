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


def jacobian_approx_t(p, f):
    """
    Numerical approximation for the multivariate Jacobian
    :param p: initial value(s)
    :param f: function handle
    :return: jacobian
    """

    try:
        jac = torch.autograd.functional.jacobian(f, p, vectorize=True) # create_graph=True
    except RuntimeError:
        jac = torch.autograd.functional.jacobian(f, p, strict=True, vectorize=False)

    return jac


def jacobian_approx_loop(p, f, dp=1e-8, args=()):
    """
    Numerical approximation for the multivariate Jacobian
    :param p: initial value(s)
    :param f: function handle
    :param dp: delta p for approximation
    :param args: additional arguments passed to function handle
    :return: jacobian
    """

    n = len(p)
    if len(args) > 0:
        # pass optional arguments to function
        fun = lambda p: f(p, *args)
        jac = torch.zeros([n, len(args[0])], dtype=p.dtype, device=p.device)
    else:
        fun = f
        jac = torch.zeros(n, dtype=p.dtype, device=p.device)

    for j in range(n):
        dpj = abs(p[j]) * dp if p[j] != 0 else dp
        p_plus = torch.Tensor([(pi if k != j else pi + dpj) for k, pi in enumerate(p)]).to(p.device)
        jac[j] = (fun(p_plus) - fun(p)) / dpj

    return jac if len(args) == 0 else jac.T
