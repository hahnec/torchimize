import torch
import functools


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

    if len(args) > 0:
        fun_args_pos_wrapper = lambda args, p: f(p, *args)
        fun = functools.partial(fun_args_pos_wrapper, args)
    else:
        fun = f

    n = len(p)
    jac = torch.zeros(n) if len(args) == 0 else torch.zeros([n, len(args[0])])
    for j in range(n):
        dpj = abs(p[j]) * dp if p[j] != 0 else dp
        p_plus = [(pi if k != j else pi + dpj) for k, pi in enumerate(p)]
        jac[j] = (fun(p_plus) - fun(p)) / dpj

    return jac if len(args) == 0 else jac.T
