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


def jacobian_approx_t(p, f, create_graph=False):
    """
    Numerical approximation for the multivariate Jacobian
    :param p: initial value(s)
    :param f: function handle
    :param create_graph: If True, the Jacobian will be computed in a differentiable manner.
    :return: jacobian
    """

    try:
        jac = torch.autograd.functional.jacobian(f, p, create_graph=create_graph, vectorize=True)
    except RuntimeError:
        jac = torch.autograd.functional.jacobian(f, p, create_graph=create_graph, strict=True, vectorize=False)

    return jac

def batch_jacobian_approx_t(p, f, create_graph=False):
    """
    Numerical approximation for the multivariate Jacobian for inputs with batch dimension
    :param p: initial value(s)
    :param f: function handle
    :param create_graph: If True, the Jacobian will be computed in a differentiable manner.
    :return: jacobian
    """
    
    # reduce batch dim: https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/7
    f_sum = lambda x: torch.sum(f(x), axis=0)

    try:
        jac_sum = torch.autograd.functional.jacobian(f_sum, p, create_graph=create_graph, vectorize=True)
    except RuntimeError:
        jac_sum = torch.autograd.functional.jacobian(f_sum, p, create_graph=create_graph, strict=True, vectorize=False)

    # bring batch dimension to the front
    num_dims = len(jac_sum.shape)
    jac = jac_sum.permute(2, *range(0, 2), *range(3, num_dims))

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
        p_plus = torch.tensor([(pi if k != j else pi + dpj) for k, pi in enumerate(p)], device=p.device)
        jac[j] = (fun(p_plus) - fun(p)) / dpj

    return jac if len(args) == 0 else jac.T
