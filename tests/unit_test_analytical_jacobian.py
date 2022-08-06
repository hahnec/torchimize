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
from tests.emg import *

class JacobianFunctionTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(JacobianFunctionTest, self).__init__(*args, **kwargs)

    def setUp(self):

        torch.seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # alpha, mu, sigma, eta
        norm, mean, sigm, skew = 10, -1, 80, 5
        self.initials = torch.tensor([7.5, -.75, 10, 3], dtype=torch.float64, device=self.device, requires_grad=True)
        self.cost_fun = lambda p, t, y: (y-self.emg_model(p, t))**2

        self.gt_params = torch.tensor([norm, mean, sigm, skew], dtype=torch.float64, device=self.device)
        self.t = torch.linspace(-1e3, 1e3, int(2e3)).to(self.device)
        self.data = self.emg_model(self.gt_params, t=self.t)
        self.data_raw = self.data + .01 * torch.randn(len(self.data), dtype=torch.float64, device=self.device)

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

    def test_gna_emg(self):

        coeffs = lsq_gna(self.initials, self.cost_fun, jac_function=self.emg_jac, args=(self.t, self.data_raw), l=.1, gtol=1e-6, max_iter=199)

        # assertion
        ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Coefficients deviate')
        eps = torch.sum(self.cost_fun(coeffs[-1], t=self.t, y=self.data_raw))
        self.assertTrue(eps.cpu() < 1, 'Error exceeded')
        self.assertTrue(len(coeffs) < 200, 'Number of iterations exceeded 200')

    def test_lma_emg(self):
        
        for m in ['lev', 'marq']:
            coeffs = lsq_lma(self.initials, self.cost_fun, jac_function=self.emg_jac, args=(self.t, self.data_raw), meth=m, gtol=1e-6, max_iter=39)

            # assertion
            ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
            self.assertTrue(ret_params, 'Coefficients deviate')
            eps = torch.sum(self.cost_fun(coeffs[-1], t=self.t, y=self.data_raw))
            self.assertTrue(eps.cpu() < 1, 'Error exceeded')
            self.assertTrue(len(coeffs) < 40, 'Number of iterations exceeded 40')

    def test_all(self):

        self.test_gna_emg()
        self.test_lma_emg()


if __name__ == '__main__':
    unittest.main()
