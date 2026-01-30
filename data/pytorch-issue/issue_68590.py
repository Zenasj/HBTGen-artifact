def serialize(coo):
    return coo._indices(), coo._values()

self.assertEqual(serialize(actual), serialize(expected))

import torch
from torch.testing._internal.common_utils import TestCase

assertEqual = TestCase().assertEqual

indices = (
    (0, 1),
    (1, 0),
)
values = (1, 2)
actual = torch.sparse_coo_tensor(indices, values, size=(2, 2))
expected = actual.clone()

assertEqual(actual, expected)


def serialize(sparse_coo_tensor):
    return sparse_coo_tensor._indices(), sparse_coo_tensor._values()


assertEqual(serialize(actual), serialize(expected))

indices = (
    (0, 1),
    (1, 0),
)
actual_values = (1, 2)
actual = torch.sparse_coo_tensor(indices, actual_values, size=(2, 2))

expected_values = (1, 3)
expected = torch.sparse_coo_tensor(indices, expected_values, size=(2, 2))

try:
    assertEqual(actual, expected)
except AssertionError as error:
    print(error)

try:
    assertEqual(serialize(actual), serialize(expected))
except AssertionError as error:
    print(error)

import torch
from torch.testing._internal.common_utils import TestCase

assertEqual = TestCase().assertEqual

actual_indices = (
    (0, 1, 1),
    (1, 0, 0),
)
actual_values = (1, 1, 1)
actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))

expected_indices = (
    (0, 1),
    (1, 0),
)
expected_values = (1, 2)
expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))

assertEqual(actual, expected)

assertEqual(actual.coalesce(), expected.coalesce())

x._indices().shape == (x.sparse_dim(), x._nnz())
x._values().shape == (x._nnz(), x.dense_dim())
len(x.shape) == x.sparse_dim() + x.dense_dim()

x.shape == y.shape
x.sparse_dim() == y.sparse_dim()
x._nnz() == y._nnz()
x._indices() == y._indices()
x._values() == y._values()