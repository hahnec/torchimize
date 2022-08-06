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

import unittest
import torch

from torchimize.functions import lsq_gna_parallel, lsq_gna_parallel_plain
from tests.emg import *

class ParallelOptimizationTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ParallelOptimizationTest, self).__init__(*args, **kwargs)

    def setUp(self):
        
        torch.manual_seed(3008)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # alpha, mu, sigma, eta
        gt_params_list = [[10, -1, 80, 5], [5, -2, 60, 5], [8, 0, 90, 3], [10, -1, 80, 5], [5, -2, 60, 5], [8, 0, 90, 3]]
        initials_list = [[7.5, -.75, 40, 3], [6, -1, 50, 4], [7, -1, 80, 4], [7.5, -.75, 40, 3], [6, -1, 50, 4], [7, -1, 80, 4]]
        channel_num = len(gt_params_list)
        self.gt_params = torch.tensor(gt_params_list, dtype=torch.float64, device=self.device, requires_grad=False)
        self.initials = torch.tensor(initials_list, dtype=torch.float64, device=self.device, requires_grad=False)
        
        self.cost_batch = lambda p_batch, t, y: (y-self.emg_model_batch(p_batch, t))**2

        self.t = torch.linspace(-1e3, 1e3, int(2e3), device=self.device, requires_grad=False)
        self.data_channels = []
        self.initials_list = []
        for i in range(channel_num):
            torch.manual_seed(3008+i)
            data = self.emg_model(self.gt_params[i, ...], t=self.t)
            self.data_channels.append(data + .01 * torch.randn(len(data), dtype=torch.float64, device=self.device, requires_grad=False))
            self.initials_list.append(self.initials[i, ...])
        self.batch_data_channels = torch.stack(self.data_channels, dim=0)
        self.batch_initials = torch.stack(self.initials_list, dim=0)

    def multi_cost_batch(self, p_batch, t, y):

        cost_a = (y - self.emg_model_batch(p_batch, t))**2
        cost_b = (y - self.emg_model_batch(p_batch, t))**2

        return torch.stack([cost_a, cost_b], dim=1)

    def multi_jaco_batch(self, p_batch, t=None, data=None):

        jac_a = self.emg_jac_batch(p_batch, t=t, data=data)
        jac_b = self.emg_jac_batch(p_batch, t=t, data=data)

        return torch.stack([jac_a, jac_b], dim=1)

    def emg_model_batch(
            self,
            p_batch,
            t: torch.Tensor = None,
        ):

        emg_batch = []
        for i in range(p_batch.shape[0]):

            emg = self.emg_model(p_batch[i, ...], t=t)
            emg_batch.append(emg)

        return torch.stack(emg_batch, dim=0)

    @staticmethod
    def emg_model(
            p,
            t: torch.Tensor = None,
        ):

        alpha, mu, sigma, eta = p

        alpha = 1 if alpha is None else alpha
        gauss = gauss_term(t, mu, sigma)
        asymm = asymm_term(t, mu, sigma, eta)

        return alpha * gauss * asymm

    def emg_jac_batch(self, p_batch, t=None, data=None):

        emg_jac_batch = []
        for i in range(p_batch.shape[0]):

            emg_jac = self.emg_jac(p_batch[i, ...], t=t, data=data[i, ...])
            emg_jac_batch.append(emg_jac)

        return torch.stack(emg_jac_batch, dim=0)

    def emg_jac(self, p, t=None, data=None):

        alpha, mu, sigma, eta = p

        c = 2*(data-self.emg_model([alpha, mu, sigma, eta], t=t))[..., None]

        jacobian = c * torch.stack([
            pd_emg_wrt_alpha(t, mu, sigma, eta),
            alpha*pd_emg_wrt_mu(t, mu, sigma, eta),
            alpha*pd_emg_wrt_sigma(t, mu, sigma, eta),
            alpha*pd_emg_wrt_eta(t, mu, sigma, eta),
        ]).T

        return jacobian

    def test_gna_emg_conditions(self):

        for p in [self.batch_initials.clone().float(), self.batch_initials.clone().double()]:

            coeffs = lsq_gna_parallel(
                p = p,
                function = self.multi_cost_batch,
                jac_function = self.multi_jaco_batch,
                args = (self.t.to(dtype=p.dtype), self.batch_data_channels.to(dtype=p.dtype)),
                wvec = torch.ones(2, device=self.device, dtype=p.dtype, requires_grad=False),
                l = 1,
                ftol = 1e-8,
                ptol = 1e-8,
                gtol = 1e-8,
                max_iter = 99,
            )

            # assertion
            self.iter_gna_cond = len(coeffs)
            self.assertTrue(self.iter_gna_cond < 100, 'Number of iterations exceeded 100')
            ret_params = torch.allclose(coeffs[-1].double(), self.gt_params, atol=1e-1)
            self.assertTrue(ret_params, 'Coefficients deviate')
            eps = torch.sum(self.cost_batch(coeffs[-1].double(), t=self.t, y=self.batch_data_channels)).cpu()
            self.eps_gna_cond = torch.round(eps, decimals=4)
            self.assertTrue(self.eps_gna_cond/len(self.gt_params) < 1, 'Error exceeded 1')

    def test_gna_emg_plain(self):

        for p in [self.batch_initials.clone().float(), self.batch_initials.clone().double()]:

            multi_cost_batch_args = lambda p: self.multi_cost_batch(p_batch=p, t=self.t.to(dtype=p.dtype), y=self.batch_data_channels.to(dtype=p.dtype))
            multi_jaco_batch_args = lambda p: self.multi_jaco_batch(p_batch=p, t=self.t.to(dtype=p.dtype), data=self.batch_data_channels.to(dtype=p.dtype))
            self.iter_gna = 199

            coeffs = lsq_gna_parallel_plain(
                p = p,
                function = multi_cost_batch_args,
                jac_function = multi_jaco_batch_args,
                wvec = torch.ones(2, device=self.device, dtype=p.dtype, requires_grad=False),
                l = 1,
                max_iter = self.iter_gna,
            )

            # assertion
            ret_params = torch.allclose(coeffs.double(), self.gt_params, atol=1e-1)
            self.assertTrue(ret_params, 'Coefficients deviate')
            eps = torch.sum(multi_cost_batch_args(coeffs.double())).cpu()
            self.eps_gna = torch.round(eps, decimals=4)
            self.assertTrue(self.eps_gna/len(self.gt_params) < 1, 'Error exceeded 1')

    def test_lma_emg_conditions(self):

        from torchimize.functions.parallel.lma_fun_parallel import lsq_lma_parallel

        for p in [self.batch_initials.clone().float(), self.batch_initials.clone().double()]:
            
            for m in ['lev']:

                coeffs = lsq_lma_parallel(
                    p = p,
                    function = self.multi_cost_batch,
                    jac_function = self.multi_jaco_batch,
                    args = (self.t.to(dtype=p.dtype), self.batch_data_channels.to(dtype=p.dtype)),
                    wvec = torch.ones(2, device=self.device, dtype=p.dtype, requires_grad=False),
                    meth = m,
                    ftol = 1e-8,
                    ptol = 1e-8,
                    gtol = 1e-8,
                    max_iter = 99,
                )

                # assertion
                self.iter_lma_cond = len(coeffs)
                self.assertTrue(self.iter_lma_cond < 100, 'Number of iterations exceeded 100')
                ret_params = torch.allclose(coeffs[-1].double(), self.gt_params, atol=1e-1)
                self.assertTrue(ret_params, 'Coefficients deviate')
                eps = torch.sum(self.cost_batch(coeffs[-1].double(), t=self.t, y=self.batch_data_channels)).cpu()
                self.eps_lma_cond = torch.round(eps, decimals=4)
                self.assertTrue(self.eps_lma_cond/len(self.gt_params) < 1, 'Error exceeded 1')

    def scipy_fit(self):

        from scipy.optimize import least_squares
        
        i = 0
        p = self.batch_initials[i, ...].clone().float().cpu()
        t = self.t.clone().cpu()
        data = self.batch_data_channels[i, ...].to(dtype=p.dtype).cpu()

        emg_model_args = lambda p: self.emg_model(p, t=t)
        emg_jacob_args = lambda p: self.emg_jac(p, t=t, data=data)

        coeffs = least_squares(
                fun = emg_model_args,
                x0 = p,
                jac = emg_jacob_args,
                method = 'lm',
            ).x

        return coeffs

    def test_fun_dims(self):

        from torchimize.functions import test_fun_dims_parallel

        for p in [self.batch_initials.float(), self.batch_initials.double()]:

            ret = test_fun_dims_parallel(
                p = p,
                function = self.multi_cost_batch,
                jac_function = self.multi_jaco_batch,
                args = (self.t.to(dtype=p.dtype), self.batch_data_channels.to(dtype=p.dtype)),
                wvec = torch.ones(2, device=self.device, dtype=p.dtype, requires_grad=False),
            )

            self.assertTrue(ret, 'torch tensor dimensionality test failed')

    def test_all(self):

        self.test_fun_dims()
        self.test_gna_emg_plain()
        self.test_gna_emg_conditions()
        self.test_lma_emg_conditions()

        # assert LMA performance against GNA
        self.assertTrue(self.eps_lma_cond <= self.eps_gna, 'LMA error is larger than that of GNA')
        self.assertTrue(self.eps_lma_cond <= self.eps_gna_cond, 'LMA error is larger than that of GNA')
        self.assertTrue(self.iter_lma_cond <= self.iter_gna, 'LMA required more iterations than GNA')
        self.assertTrue(self.iter_lma_cond <= self.iter_gna_cond, 'LMA required more iterations than GNA')


if __name__ == '__main__':
    unittest.main()
