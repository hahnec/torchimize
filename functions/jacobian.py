import numpy as np
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


def jacobian_approx_np(p, f, dp=1e-8, args=()):
    """
    Numerical approximation for the multivariate Jacobian
    :param p: initial value(s)
    :param f: function handle
    :param dp: delta p for approximation
    :param args: additional arguments passed to function handle
    :return: jacobian
    """

    n = len(p)
    jac = np.zeros(n) if len(args) == 0 else np.zeros([n, len(args[0])])
    for j in range(n):  # through columns to allow for vector addition
        dpj = abs(p[j]) * dp if p[j] != 0 else dp
        p_plus = [(pi if k != j else pi + dpj) for k, pi in enumerate(p)]
        # compute jacobian while passing optional arguments
        jac[j] = (f(*((p_plus,) + args)) - f(*((p,) + args))) / dpj

    return jac if len(args) == 0 else jac.T
