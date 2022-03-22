==========
Torchimize
==========

Functional API Usage
--------------------

.. code-block:: python

    # gauss-newton
    from torchimize.functions import lsq_lma
    coeffs_gna, eps_gna = lsq_gna(initials, cost_fun, args=(other_inputs,), tol=1e-6)

    # levenberg-marquardt
    from torchimize.functions import lsq_lma
    coeffs_lma, eps_lma = lsq_lma(initials, cost_fun, args=(other_inputs,), tol=1e-6)
    