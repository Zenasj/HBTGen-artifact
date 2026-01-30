import numpy as np
import random

a = np.random.rand(0, 4)
print([b.shape[0] for b in a])

a[:, 2:] += a[:, :2]
print(a[:, 2])

def torch_inter_coord_distance(a: "Tensor[n,3]", b: "Tensor[m, 3]") -> "Tensor[n, m]":
    return (a.unsqueeze(1) - b.unsqueeze(0)).norm(dim=-1)

import numpy

def numpy_inter_coord_distance(a: "ndarray[n,3]", b: "ndarray[m, 3]") -> "ndarray[n, m]":
    return numpy.linalg.norm((numpy.expand_dims(a, 1) - numpy.expand_dims(b, 0)), axis=-1)

coords = numpy.arange(12).reshape(4, 3)
print("numpy:")
print("4x4:")
print(numpy_inter_coord_distance(coords, coords))
print(numpy_inter_coord_distance(coords, coords).shape)
print()

print("2x4:")
print(numpy_inter_coord_distance(coords[0:2], coords))
print(numpy_inter_coord_distance(coords[0:2], coords).shape)
print()

print("0x4:")
print(numpy_inter_coord_distance(coords[0:0], coords))
print(numpy_inter_coord_distance(coords[0:0], coords).shape)
print()

import torch

def torch_inter_coord_distance(a: "Tensor[n,3]", b: "Tensor[m, 3]") -> "Tensor[n, m]":
    return (a.unsqueeze(1) - b.unsqueeze(0)).norm(dim=-1)

coords = torch.arange(12).reshape(4, 3)

print("torch:")
print("4x4:")
print(torch_inter_coord_distance(coords, coords))
print(torch_inter_coord_distance(coords, coords).shape)
print()

print("2x4:")
print(torch_inter_coord_distance(coords[0:2], coords))
print(torch_inter_coord_distance(coords[0:2], coords).shape)
print()

print("0x4:")
print(torch_inter_coord_distance(coords[0:0], coords))
print(torch_inter_coord_distance(coords[0:0], coords).shape)
print()