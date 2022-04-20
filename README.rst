==========
torchimize
==========

*torchimize* contains implementations of the Gauss-Newton and Levenberg-Marquardt optimization algorithms using the PyTorch library. The main motivation for this project is to enable convex optimization on GPUs based on the torch.Tensor class, which (as of April 2022) is widely used in the deep learning field.

|coverage| |tests_develop| |tests_master|

Functional API Usage
--------------------

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