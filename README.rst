==========
torchimize
==========

..

Description
===========

*torchimize* contains implementations of the Gauss-Newton and Levenberg-Marquardt optimization algorithms using the PyTorch library. The main motivation for this project is to enable convex optimization on GPUs based on the torch.Tensor class, which (as of 2022) is widely used in the deep learning field. This package features the capability to minimize several least-squares optimization problems at each loop iteration simultaneously.

|coverage| |tests_develop| |tests_master| |pypi| |license|

Installation
============

``$ python3 -m pip install torchimize``

Functional API Usage
====================

.. code-block:: python

    # gauss-newton
    from torchimize.functions import lsq_gna
    coeffs_list = lsq_gna(initials, cost_fun, args=(other_args,))

    # levenberg-marquardt
    from torchimize.functions import lsq_lma
    coeffs_list = lsq_lma(initials, function=cost_fun, jac_function=jac_fun, args=(other_args,))

    # parallel gauss-newton using batches
    from torchimize.functions import lsq_gna_parallel
    coeffs_list = lsq_gna_parallel(initials_batch, function=cost_fun_batch, jac_function=jac_fun_batch, args=(other_args,))


.. substitutions

.. |coverage| image:: https://coveralls.io/repos/github/hahnec/torchimize/badge.svg?branch=master
    :target: https://coveralls.io/github/hahnec/torchimize

.. |tests_develop| image:: https://img.shields.io/github/workflow/status/hahnec/torchimize/torchimize%20unit%20tests/develop?label=tests%20on%20develop
    :target: https://github.com/hahnec/torchimize/actions/

.. |tests_master| image:: https://img.shields.io/github/workflow/status/hahnec/torchimize/torchimize%20unit%20tests/master?label=tests%20on%20master
    :target: https://github.com/hahnec/torchimize/actions/

.. |license| image:: https://img.shields.io/badge/License-GPL%20v3.0-orange.svg
    :target: https://www.gnu.org/licenses/gpl-3.0.en.html
    :alt: License

.. |pypi| image:: https://img.shields.io/pypi/dm/torchimize?label=PyPI%20downloads
    :target: https://pypi.org/project/torchimize/
    :alt: PyPI Downloads