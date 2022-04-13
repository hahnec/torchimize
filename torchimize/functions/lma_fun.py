import torch 
import functools

from torchimize.functions.jacobian import jacobian_approx_t


def lsq_lma(p, function, jac_fun=None, args=(), tol=1e-7, tau=1e-3, meth='lev', rho1=.25, rho2=.75, bet=2, gam=3, max_iter=500):
    """
    Levenberg-Marquardt implementation for least-squares fitting of non-linear functions
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param tol: tolerance for stop condition
    :param tau: factor to initialize damping parameter
    :param meth: method which is default 'lev' for Levenberg and otherwise Marquardt
    :param rho1: first gain factor threshold for damping parameter adjustment for Marquardt
    :param rho2: second gain factor threshold for damping parameter adjustment for Marquardt
    :param bet: multiplier for damping parameter adjustment for Marquardt
    :param gam: divisor for damping parameter adjustment for Marquardt
    :param max_iter: maximum number of iterations
    :return: list of results, eps
    """

    if len(args) > 0:
        fun_args_pos_wrapper = lambda args, p: function(p, *args)
        fun = functools.partial(fun_args_pos_wrapper, args)
    else:
        fun = function

    # use numerical Jacobian if analytical is not provided
    if jac_fun is None:
        jac_fun = functools.partial(jacobian_approx_t, f=fun)
    else:
        jac_args_pos_wrapper = lambda args, p: jac_fun(p, *args)
        jac_fun = functools.partial(jac_args_pos_wrapper, args)

    j = jac_fun(p)
    g = torch.matmul(j.T, fun(p))
    H = torch.matmul(j.T, j)
    u = tau * torch.max(torch.diag(H.diagonal()))
    v = 2
    eps = 1
    p_list = []
    while len(p_list) < max_iter:
        D = torch.eye(j.shape[1], device=j.device)
        D *= 1 if meth == 'lev' else torch.max(torch.maximum(H.diagonal(), D.diagonal()))
        h = -torch.matmul(torch.linalg.inv(H+u*D), g)
        f = fun(p)
        f_h = fun(p+h)
        rho_denom = torch.matmul(.5*h.T, u*h-g)
        rho_nom = (torch.matmul(f.T, f) - torch.matmul(f_h.T, f_h))
        rho = rho_nom / rho_denom if rho_denom > 0 else torch.inf if rho_nom > 0 else -torch.inf
        if rho > 0:
            p = p + h
            p_list.append(p.detach())
            j = jac_fun(p)
            g = torch.matmul(j.T, fun(p))
            H = torch.matmul(j.T, j)
        if meth== 'lev':
            u, v = (u*torch.max([1/3, 1-(2*rho-1)**3]), 2) if rho > 0 else (u*v, v*2)
        else:
            u = u*bet if rho < rho1 else u/gam if rho > rho2 else u
        eps = max(abs(g))
        if eps < tol:
            break

    return p_list, eps
