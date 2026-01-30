import torch

add_docstr(torch.abs, r"""
abs(input, *, out=None) -> Tensor

Computes the absolute value of each element in :attr:`input`.

.. math::
    \text{out}_{i} = |\text{input}_{i}|
""" + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
""".format(**common_args))

add_docstr(torch.absolute,
           r"""
absolute(input, *, out=None) -> Tensor

Alias for :func:`torch.abs`
""")

add_docstr(
    torch.abs,
    r"""
abs(input, *, out=None) -> Tensor

Computes the absolute value of each element in :attr:`input`.

.. math::
    \text{out}_{i} = |\text{input}_{i}|
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
""".format(
        **common_args
    ),
)

add_docstr(
    torch.absolute,
    r"""
absolute(input, *, out=None) -> Tensor

Alias for :func:`torch.abs`
""",
)