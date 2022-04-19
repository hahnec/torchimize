import torch

from torchimize.functions.jacobian import jacobian_approx_t


def lsq_gna(p, function, jac_function=None, args=(), l=1., tol=1e-7, max_iter=50):
    """
    Gauss-Newton implementation for least-squares fitting of non-linear functions
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param l: learning rate
    :param tol: tolerance for stop condition
    :param max_iter: maximum number of iterations
    :return: list of results, eps
    """

    if len(args) > 0:
        # pass optional arguments to function
        fun = lambda p, args=args: function(p, *args)
    else:
        fun = function

    if jac_function is None:
        # use numerical Jacobian if analytical is not provided
        jac_fun = lambda p, f=fun: jacobian_approx_t(p, f=f)
    else:
        jac_fun = lambda p, args=args: jac_function(p, *args)

    j = jac_fun(p)
    g = torch.matmul(j.T, fun(p))
    H = torch.matmul(j.T, j)
    eps = 1
    p_list = []
    while len(p_list) < max_iter:
        h = -l*torch.matmul(torch.linalg.pinv(H), g)
        p = p + h
        p_list.append(p.detach())
        j = jac_fun(p)
        g = torch.matmul(j.T, fun(p))
        H = torch.matmul(j.T, j)
        eps = max(abs(g))
        if eps < tol:
            break
    return p_list, eps
