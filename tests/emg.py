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
