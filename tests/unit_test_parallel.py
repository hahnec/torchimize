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
from tests.emg_mm import *

class ParallelOptimizationTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ParallelOptimizationTest, self).__init__(*args, **kwargs)

    def setUp(self):
        
        torch.manual_seed(3008)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # alpha, mu, sigma, eta
        gt_params_list = [[10, -1, 80, 5], [5, -2, 60, 5], [8, 0, 90, 3], [10, -1, 80, 5], [5, -2, 60, 5], [8, 0, 90, 3]]
        initials_list = [[7.5, -.75, 10, 3], [6, -1, 50, 4], [7, -1, 80, 4], [7.5, -.75, 10, 3], [6, -1, 50, 4], [7, -1, 80, 4]]
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

        coeffs = lsq_gna_parallel(
            p = self.batch_initials,
            function = self.multi_cost_batch,
            jac_function = self.multi_jaco_batch,
            args = (self.t, self.batch_data_channels),
            l = .1,
            gtol = 1e-6,
            max_iter = 199,
        )

        # assertion
        ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Coefficients deviate')
        eps = torch.sum(self.cost_batch(coeffs[-1], t=self.t, y=self.batch_data_channels))
        self.assertTrue(eps.cpu()/len(self.gt_params) < 1, 'Error exceeded 1')
        self.assertTrue(len(coeffs) < 200, 'Number of iterations exceeded 200')

    def test_gna_emg_plain(self):

        multi_cost_batch_args = lambda p, t=self.t, y=self.batch_data_channels: self.multi_cost_batch(p_batch=p, t=t, y=y)
        multi_jaco_batch_args = lambda p, t=self.t, y=self.batch_data_channels: self.multi_jaco_batch(p_batch=p, t=t, data=y)

        coeffs = lsq_gna_parallel_plain(
            p = self.batch_initials,
            function = multi_cost_batch_args,
            jac_function = multi_jaco_batch_args,
            wvec = torch.ones(2, device=self.device, dtype=torch.float64, requires_grad=False),
            l = .1,
            max_iter = 199,
        )

        # assertion
        ret_params = torch.allclose(coeffs, self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Coefficients deviate')
        eps = torch.sum(multi_cost_batch_args(coeffs))
        self.assertTrue(eps.cpu()/len(self.gt_params) < 1, 'Error exceeded 1')
        self.assertTrue(len(coeffs) < 200, 'Number of iterations exceeded 200')

    def test_lma_emg_conditions(self):

        from torchimize.functions.lma_fun_parallel import lsq_lma_parallel

        for m in ['lev', 'marq']:

            coeffs = lsq_lma_parallel(
                self.batch_initials,
                function = self.multi_cost_batch,
                jac_function = self.multi_jaco_batch,
                args = (self.t, self.batch_data_channels),
                wvec = torch.ones(2, device=self.device, dtype=torch.float64, requires_grad=False),
                meth = m,
                max_iter = 199,
            )

            # assertion
            ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
            self.assertTrue(ret_params, 'Coefficients deviate')
            eps = torch.sum(self.cost_batch(coeffs[-1], t=self.t, y=self.batch_data_channels))
            self.assertTrue(eps.cpu()/len(self.gt_params) < 1, 'Error exceeded 1')
            self.assertTrue(len(coeffs) < 40, 'Number of iterations exceeded 40')

    def test_all(self):

        self.test_lma_emg_conditions()
        self.test_gna_emg_conditions()
        self.test_gna_emg_plain()


if __name__ == '__main__':
    unittest.main()
