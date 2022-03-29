import torch
import functools

from torchimize.functions.jacobian import jacobian_approx_t


def lsq_gna(p, function, args=(), l=.1, tol=1e-7, max_iter=500):
    """
    Gauss-Newton implementation for least-squares fitting of non-linear functions
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param l: learning rate
    :param tol: tolerance for stop condition
    :param max_iter: maximum number of iterations
    :return: list of results, eps
    """

    if len(args) > 0:
        fun_args_pos_wrapper = lambda args, p: function(p, args)
        fun = functools.partial(fun_args_pos_wrapper, *args)
    else:
        fun = function

    j = jacobian_approx_t(p, fun)
    g = torch.matmul(j.T, fun(p))
    H = torch.matmul(j.T, j)
    eps = 1
    p_list = []
    while len(p_list) < max_iter:
        h = -l*torch.matmul(torch.linalg.inv(H), g)
        p = p + h
        p_list.append(p.detach())
        j = jacobian_approx_t(p, fun)
        g = torch.matmul(j.T, fun(p))
        H = torch.matmul(j.T, j)
        eps = max(abs(g))
        if eps < tol:
            break
    return p_list, eps
