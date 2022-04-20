import unittest
import torch

from torchimize.functions.lma_fun import lsq_lma
from torchimize.functions.gna_fun import lsq_gna

alpha_term = lambda sigma: 1 / (sigma * 2**.5 * torch.pi)
gauss_term = lambda t, mu, sigma: torch.exp(-.5 * (t-mu)**2 / sigma**2)
asymm_term = lambda t, mu, sigma, eta: 1 + torch.erf(eta * (t-mu) / (sigma * 2**.5))

pd_gauss_wrt_mu = lambda t, mu, sigma: gauss_term(t, mu, sigma) * (t-mu)/sigma**2
pd_gauss_wrt_sigma = lambda t, mu, sigma: gauss_term(t, mu, sigma) * (t-mu)**2/sigma**3

erf_term = lambda t, mu, sigma, eta: torch.exp(-.5*((t-mu)**2/ sigma**2) * eta**2)
pd_asymm_wrt_mu = lambda t, mu, sigma, eta: erf_term(t, mu, sigma, eta) * -1 * (2/torch.pi)**.5 * eta/sigma
pd_asymm_wrt_sigma = lambda t, mu, sigma, eta: erf_term(t, mu, sigma, eta) * -1 * (2/torch.pi)**.5 * eta*(t-mu)/sigma**2
pd_asymm_wrt_eta = lambda t, mu, sigma, eta: erf_term(t, mu, sigma, eta) * (2/torch.pi)**.5 * (t-mu)/sigma

def pd_emg_wrt_mu(t, mu, sigma, eta):

    product_rule_term_a = gauss_term(t, mu, sigma) * pd_asymm_wrt_mu(t, mu, sigma, eta)
    product_rule_term_b = pd_gauss_wrt_mu(t, mu, sigma) * asymm_term(t, mu, sigma, eta)

    return -1 * (product_rule_term_a + product_rule_term_b)

def pd_emg_wrt_sigma(t, mu, sigma, eta):

    product_rule_term_a = gauss_term(t, mu, sigma) * pd_asymm_wrt_sigma(t, mu, sigma, eta)
    product_rule_term_b = pd_gauss_wrt_sigma(t, mu, sigma) * asymm_term(t, mu, sigma, eta)

    return -1 * (product_rule_term_a + product_rule_term_b)

def pd_emg_wrt_eta(t, mu, sigma, eta):
    return -1 * gauss_term(t, mu, sigma) * pd_asymm_wrt_eta(t, mu, sigma, eta)

def pd_emg_wrt_alpha(t, mu, sigma, eta):
    return -1 * gauss_term(t, mu, sigma) * asymm_term(t, mu, sigma, eta)

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

        coeffs, eps = lsq_gna(self.initials, self.cost_fun, jac_function=self.emg_jac, args=(self.t, self.data_raw), l=.1, tol=1e-6, max_iter=199)

        # assertion
        ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Coefficients deviate')
        self.assertTrue(eps.cpu() < 1, 'Error exceeded 1')
        self.assertTrue(len(coeffs) < 200, 'Number of iterations exceeded 200')

    def test_lma_emg(self):

        coeffs, eps = lsq_lma(self.initials, self.cost_fun, jac_function=self.emg_jac, args=(self.t, self.data_raw), meth='lev', tol=1e-6, max_iter=39)

        # assertion
        ret_params = torch.allclose(coeffs[-1], self.gt_params, atol=1e-1)
        self.assertTrue(ret_params, 'Coefficients deviate')
        self.assertTrue(eps.cpu() < 1, 'Error exceeded 1')
        self.assertTrue(len(coeffs) < 40, 'Number of iterations exceeded 40')

    def test_all(self):
        self.test_lma_emg()
        self.test_gna_emg()


if __name__ == '__main__':
    unittest.main()
