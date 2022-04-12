from tests.unit_test_raw_fit import TorchimizerTest
from tests.unit_test_skewed_gaussian import SkewedGaussianTest

test_classes = [
    TorchimizerTest, SkewedGaussianTest,
                ]

for test_class in test_classes:

    # instantiate test object
    obj = test_class()
    obj.setUp()

    # switch off plots for headless
    obj.plt_opt = False

    obj.test_all()

    del obj
