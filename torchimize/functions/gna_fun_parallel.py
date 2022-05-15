import torch
from typing import Union, Callable, Tuple, List

from torchimize.functions.jacobian import jacobian_approx_t

def lsq_gna_parallel(
        p: torch.Tensor,
        function: Callable, 
        jac_function: Callable = None,
        args: Union[Tuple, List] = (),
        ftol: float = 1e-8,
        ptol: float = 1e-8,
        gtol: float = 1e-8,
        l: float = 1.,
        max_iter: int = 100,
    ) -> List[torch.Tensor]:
    """
    Gauss-Newton implementation for least-squares fitting of non-linear functions
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param ftol: relative change in cost function as stop condition
    :param ptol: relative change in independant variables as stop condition
    :param gtol: maximum gradient tolerance as stop condition
    :param l: learning rate
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
    g = torch.bmm(j.transpose(-2, -1), f[..., None])[..., 0]
    H = torch.bmm(j.transpose(-2, -1), j)
    p_list = [p]
    while len(p_list) < max_iter:
        h = -l*torch.linalg.lstsq(H, g, rcond=None, driver=None)[0]#-l*torch.bmm(torch.pinverse(H), g)[..., 0]#
        p = p + h
        p_list.append(p.detach())
        f_prev = f.clone()
        f = fun(p)
        j = jac_fun(p)
        g = torch.bmm(j.transpose(-2, -1), f[..., None])[..., 0]
        H = torch.bmm(j.transpose(-2, -1), j)

        # stop conditions
        gcon = torch.max(abs(g)) < gtol
        pcon = (h**2).sum()**.5 < ptol*(ptol + (p**2).sum()**.5)
        fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum()
        if gcon or pcon or fcon:
            break

    return p_list
