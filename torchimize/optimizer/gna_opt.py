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
from torch import Tensor
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.nn.utils import _stateless
from typing import List


class GNA(Optimizer):
    r"""Implements Gauss-Newton.
    """

    def __init__(self, params, lr: float, model: nn.Module):

        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)

        super(GNA, self).__init__(params, defaults)

        self._model = model
        self._params = self.param_groups[0]['params']
        self._modules = [m for m in model.modules()][1:]
        self._j_list = []
        self._h_list = []

    def __setstate__(self, state):
        super(GNA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, x: torch.Tensor, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            parameters = dict(self._model.named_parameters())
            keys, values = zip(*parameters.items())

            # vectorized jacobian (https://github.com/pytorch/pytorch/issues/49171)
            def func(*params: torch.Tensor):
                out = _stateless.functional_call(self._model, {n: p for n, p in zip(keys, params)}, x)
                return out
            self._j_list: tuple[torch.Tensor] = torch.autograd.functional.jacobian(func, values, create_graph=True)
            self._j_list = [j.squeeze(1).flatten(start_dim=1) for j in self._j_list] # remove hidden and predict bias dimensions

            # vectorized hessian (https://github.com/pytorch/pytorch/issues/49171)
            def loss(*params):
                out: torch.Tensor = _stateless.functional_call(self._model, {n: p for n, p in zip(keys, params)}, x)
                return out.square().sum()
            self._h_list: tuple[torch.Tensor] = torch.autograd.functional.hessian(loss, tuple(self._model.parameters()), create_graph=True)
            #self._h_list = [torch.squeeze(h) for i, h in enumerate(self._h_list)] # filter hessian and remove hidden and predict bias dimensions

            self.gna_update(
                params_with_grad,
                d_p_list,
                lr=lr,
            )

        return loss

    def gna_update(
            self,
            params: List[Tensor],
            d_p_list: List[Tensor],
            lr: float,
                 ):
        r"""Functional API that performs Gauss-Newton algorithm computation.

        """
        # 4 list elements in raw_train_fit.py for hidden and predict weights and biases
        for i, param in enumerate(params):

            d_p = d_p_list[i]
            j = self._j_list[i]
            h = j.T.matmul(j)
            #h = self._h_list[i][i]
            h = h + torch.eye(j.shape[1])*torch.finfo(h.dtype).eps      # prevent zeros along hessian diagonal
            h_i = h.inverse()
            if h.shape[-1] == d_p.shape[0]:
                param.add_(h_i.matmul(d_p), alpha=-lr)
            elif h.shape[-1] == d_p.shape[::-1][0]:
                param.add_(h_i.matmul(d_p.T).T, alpha=-lr)
            else:
                raise Exception('Tensor dimension mismatch')