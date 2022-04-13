import unittest
import torch

from torchimize.functions.lma_fun import lsq_lma
from torchimize.functions.gna_fun import lsq_gna


class SkewedGaussianTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SkewedGaussianTest, self).__init__(*args, **kwargs)

    def setUp(self):
        
        torch.seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.initials = torch.tensor([7.5, -.5, .5, 2.5], dtype=torch.float64, device=self.device, requires_grad=True)
        self.cost_fun = lambda p, y: (y-self.skewed_gaussian(p))**2

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

        coeffs, eps = lsq_gna(self.initials, self.cost_fun, args=(self.data_raw,), tol=1e-6)

        # assertion
        ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Skewed Gaussian coefficients deviate')
        self.assertTrue(eps.cpu() < 1, 'Error exceeded 1')
        self.assertTrue(len(coeffs) < 200, 'Number of skewed Gaussian fit iterations exceeded 200')

    def test_lma_skewed_gaussian(self):

        coeffs, eps = lsq_lma(self.initials, self.cost_fun, args=(self.data_raw,), meth='marq', tol=1e-6)

        # assertion
        ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Skewed Gaussian coefficients deviate')
        self.assertTrue(eps.cpu() < 1, 'Error exceeded 1')
        self.assertTrue(len(coeffs) < 40, 'Number of skewed Gaussian fit iterations exceeded 40')

    def test_all(self):
        self.test_lma_skewed_gaussian()


if __name__ == '__main__':
    unittest.main()
