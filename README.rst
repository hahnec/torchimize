==========
torchimize
==========

..

Description
===========

*torchimize* contains implementations of the Gauss-Newton and Levenberg-Marquardt optimization algorithms using the PyTorch library. The main motivation for this project is to enable convex optimization on GPUs based on the torch.Tensor class, which (as of 2022) is widely used in the deep learning field. This enables to minimize several least-squares optimization problems per loop iteration simultaneously.

|coverage| |tests_develop| |tests_master| |license|

Functional API Usage
====================

.. code-block:: python

    # gauss-newton
    from torchimize.functions import lsq_gna
    coeffs_gna, eps_gna = lsq_gna(initials, cost_fun, args=(other_inputs,), tol=1e-6)

    # levenberg-marquardt
    from torchimize.functions import lsq_lma
    coeffs_lma, eps_lma = lsq_lma(initials, cost_fun, args=(other_inputs,), tol=1e-6)
    

.. substitutions

.. |coverage| image:: https://coveralls.io/repos/github/hahnec/torchimize/badge.svg?branch=master
    :target: https://coveralls.io/github/hahnec/torchimize

.. |tests_develop| image:: https://img.shields.io/github/workflow/status/hahnec/torchimize/torchimize%20unit%20tests/develop?label=tests%20on%20develop
    :target: https://github.com/hahnec/torchimize/actions/

.. |tests_master| image:: https://img.shields.io/github/workflow/status/hahnec/torchimize/torchimize%20unit%20tests/master?label=tests%20on%20master
    :target: https://github.com/hahnec/torchimize/actions/

.. |license| image:: https://img.shields.io/badge/License-GPL%20v3.0-orange.svg?style=flat-square
    :target: https://www.gnu.org/licenses/gpl-3.0.en.html
    :alt: License