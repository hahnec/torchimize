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

from torchimize.functions.single.lma_fun_single import lsq_lma
from torchimize.functions.single.gna_fun_single import lsq_gna
from torchimize.functions.jacobian import jacobian_approx_loop


class SkewedGaussianTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SkewedGaussianTest, self).__init__(*args, **kwargs)

    def setUp(self):
        
        torch.manual_seed(3006)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.initials = torch.tensor([7.5, -.5, .5, 2.5], dtype=torch.float64, device=self.device, requires_grad=True)
        self.cost_fun = lambda p, y, placeholder: (y-self.skewed_gaussian(p))**2

        norm, mean, sigm, skew = 10, -1, 2, 5
        self.gt_params = torch.tensor([norm, mean, sigm, skew], dtype=torch.float64, device=self.device)
        self.data = self.skewed_gaussian(self.gt_params)
        self.data_raw = self.data + .01 * torch.randn(len(self.data), dtype=torch.float64, device=self.device)

    @staticmethod
    def skewed_gaussian(p):

        x = torch.linspace(-5, 5, 1000, dtype=p.dtype, device=p.device)

        amp = p[0] / (p[2] * (2 * torch.pi)**.5)
        spread = torch.exp((-(x - p[1]) ** 2.0) / (2 * p[2] ** 2.0))
        skew = 1 + torch.erf((p[3] * (x - p[1])) / (p[2] * 2**.5))
        return (amp * spread * skew).float()

    def test_gna_skewed_gaussian(self):

        coeffs = lsq_gna(self.initials, self.cost_fun, args=(self.data_raw, None), gtol=1e-6, max_iter=199)

        # assertion
        ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Skewed Gaussian coefficients deviate')
        eps = torch.sum(self.cost_fun(coeffs[-1], y=self.data_raw, placeholder=None))
        self.assertTrue(eps.cpu() < .2, 'Error exceeded')
        self.assertTrue(len(coeffs) < 200, 'Number of skewed Gaussian fit iterations exceeded 200')

    def test_lma_skewed_gaussian(self):

        for m in ['marq', 'lev']:
            coeffs = lsq_lma(self.initials, self.cost_fun, args=(self.data_raw, None), meth=m, gtol=1e-6, max_iter=49)

            # assertion
            ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
            self.assertTrue(ret_params, 'Skewed Gaussian coefficients deviate')
            eps = torch.sum(self.cost_fun(coeffs[-1], y=self.data_raw, placeholder=None))
            self.assertTrue(eps.cpu() < .2, 'Error exceeded')
            self.assertTrue(len(coeffs) < 50, 'Number of skewed Gaussian fit iterations exceeded 50')

    def test_jac_skewed_gaussian(self):

        # attach arguments to cost function and numerical jacobian
        wrapped_cost_fun = lambda p, y: self.cost_fun(p, y, placeholder=None)
        jac_fun = lambda p, args: jacobian_approx_loop(p, f=wrapped_cost_fun, dp=1e-6, args=(args,))

        jac_mat = jac_fun(p=self.initials, args=self.data_raw)

        self.assertEqual(jac_mat.shape, torch.Size([len(self.data_raw), len(self.gt_params)]), 'Numerical Jacobian is of wrong dimensions')
        self.assertTrue(torch.sum(jac_mat.isnan()) == 0, 'NaNs in Jacobian')

        coeffs = lsq_lma(self.initials, wrapped_cost_fun, jac_function=jac_fun, args=(self.data_raw,), meth='marq', gtol=1e-6, max_iter=39)

        # assertion
        ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Skewed Gaussian coefficients deviate')
        eps = torch.sum(self.cost_fun(coeffs[-1], y=self.data_raw, placeholder=None))
        self.assertTrue(eps.cpu() < .2, 'Error exceeded 1')
        self.assertTrue(len(coeffs) < 40, 'Number of skewed Gaussian fit iterations exceeded 40')

    def test_all(self):
        self.test_gna_skewed_gaussian()
        self.test_lma_skewed_gaussian()
        self.test_jac_skewed_gaussian()


if __name__ == '__main__':
    unittest.main()
