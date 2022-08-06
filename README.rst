|logo|

..

Description
===========

*torchimize* contains implementations of the Gauss-Newton and Levenberg-Marquardt optimization algorithms using the PyTorch library. The main motivation for this project is to enable convex optimization on GPUs based on the torch.Tensor class, which (as of 2022) is widely used in the deep learning field. This package features the capability to minimize several least-squares optimization problems at each loop iteration in parallel.

|coverage| |tests_develop| |tests_master| |pypi| |license|

Installation
============

``$ python3 -m pip install torchimize``

Kick-Start
==========

Cost Optimization
-----------------

.. code-block:: python

    # single gauss-newton
    from torchimize.functions import lsq_gna
    coeffs_list = lsq_gna(initials, cost_fun, args=(other_args,))

    # single levenberg-marquardt
    from torchimize.functions import lsq_lma
    coeffs_list = lsq_lma(initials, function=cost_fun, jac_function=jac_fun, args=(other_args,))

    # parallel gauss-newton for several optimization problems at multiple costs
    from torchimize.functions import lsq_gna_parallel
    coeffs_list = lsq_gna_parallel(
                        p = initials_batch,
                        function = multi_cost_fun_batch,
                        jac_function = multi_jac_fun_batch,
                        args = (other_args,),
                        wvec = torch.ones(5, device='cuda', dtype=initials_batch.dtype),
                        ftol = 1e-8,
                        ptol = 1e-8,
                        gtol = 1e-8,
                        l = 1.,
                        max_iter = 80,
                    )

    # parallel levenberg-marquardt for several optimization problems at multiple costs
    from torchimize.functions import lsq_lma_parallel
    coeffs_list = lsq_lma_parallel(
                        p = initials_batch,
                        function = multi_cost_fun_batch,
                        jac_function = multi_jac_fun_batch,
                        args = (other_args,),
                        wvec = torch.ones(5, device='cuda', dtype=initials_batch.dtype),
                        ftol = 1e-8,
                        ptol = 1e-8,
                        gtol = 1e-8,
                        meth = 'marq',
                        max_iter = 40,
                    )

    # validate that your provided functions return correct tensor dimensionality
    from torchimize.functions import test_fun_dims_parallel
    ret = test_fun_dims_parallel(
        p = initials_batch,
        function = multi_cost_fun_batch,
        jac_function = multi_jac_fun_batch,
        args = (other_args,),
        wvec = torch.ones(5, device='cuda', dtype=initials_batch.dtype),
    )

.. note::
    For simultaneous minimization of ``B`` optimization problems at a multiple of ``C`` costs, the ``function`` and ``jac_function`` arguments require to return a torch.Tensor type of ``B x C x N`` and ``B x C x N x P``, respectively. Here, ``N`` is the residual dimension and ``P`` represents the sought parameter number in each ``B x C``.

For further details, see the |apidoc|_.

.. substitutions

.. |logo| image:: https://github.com/hahnec/torchimize/blob/develop/docs/torchimize_logo_font.svg
    :target: https://hahnec.github.io/torchimize/
    :width: 400 px
    :scale: 100 %
    :alt: torchimize

.. |coverage| image:: https://coveralls.io/repos/github/hahnec/torchimize/badge.svg?branch=master
    :target: https://coveralls.io/github/hahnec/torchimize
    :width: 98

.. |tests_develop| image:: https://img.shields.io/github/workflow/status/hahnec/torchimize/torchimize%20unit%20tests/develop?label=tests%20on%20develop
    :target: https://github.com/hahnec/torchimize/actions/
    :width: 150

.. |tests_master| image:: https://img.shields.io/github/workflow/status/hahnec/torchimize/torchimize%20unit%20tests/master?label=tests%20on%20master
    :target: https://github.com/hahnec/torchimize/actions/
    :width: 150

.. |license| image:: https://img.shields.io/badge/License-GPL%20v3.0-orange.svg?logoWidth=40
    :target: https://www.gnu.org/licenses/gpl-3.0.en.html
    :alt: License
    :width: 150

.. |pypi| image:: https://img.shields.io/pypi/dm/torchimize?label=PyPI%20downloads
    :target: https://pypi.org/project/torchimize/
    :alt: PyPI Downloads
    :width: 162

.. |apidoc| replace:: **API documentation**
.. _apidoc: https://hahnec.github.io/torchimize/build/html/apidoc.html

Citation
========

.. code-block:: BibTeX

    @misc{torchimize,
        title={torchimize},
        author={Hahne, Christopher and Hayoz, Michel},
        year={2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/hahnec/torchimize}}
    }
