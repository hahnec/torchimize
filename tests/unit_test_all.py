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

from tests.unit_test_raw_fit import TorchimizerTest
from tests.unit_test_skewed_gaussian import SkewedGaussianTest
from tests.unit_test_analytical_jacobian import JacobianFunctionTest
from tests.unit_test_parallel import ParallelOptimizationTest

test_classes = [
    SkewedGaussianTest,
    TorchimizerTest,
    JacobianFunctionTest,
    ParallelOptimizationTest,
                ]

for test_class in test_classes:

    # instantiate test object
    obj = test_class()
    obj.setUp()

    # switch off plots for headless
    obj.plt_opt = False

    obj.test_all()

    del obj
