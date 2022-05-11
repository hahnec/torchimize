import torch

from torchimize.functions.jacobian import jacobian_approx_t


def lsq_gna(
        p, 
        function, 
        jac_function=None, 
        args=(), 
        ftol=1e-8,
        ptol=1e-8,
        gtol=1e-8,
        l=1.,
        max_iter=50,
    ):
    """
    Gauss-Newton implementation for least-squares fitting of non-linear functions
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param l: learning rate
    :param ftol: relative change in cost function as stop condition
    :param ptol: relative change in independant variables as stop condition
    :param gtol: maximum gradient tolerance as stop condition
    :param max_iter: maximum number of iterations
    :return: list of results, eps
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
    g = torch.matmul(j.T, )
    H = torch.matmul(j.T, j)
    eps = 1
    p_list = []
    while len(p_list) < max_iter:
        h = -l*torch.matmul(torch.linalg.pinv(H), g)
        p = p + h
        p_list.append(p.detach())
        f = fun(p)
        j = jac_fun(p)
        g = torch.matmul(j.T, f)
        H = torch.matmul(j.T, j)

        # stop conditions
        gcon = max(abs(g)) < gtol
        pcon = sum(h**2)**.5 < ptol*(ptol + sum(p**2)**.5)
        fcon = ((fun(p_list[-2])-fun(p_list[-1]))**2).sum() < ((ftol*f)**2).sum()
        if gcon or pcon or fcon:
            break

    return p_list, eps
