import torch.nn as nn
import numpy as np

nn.Sequential(
  nn.Linear(num_input_features, int(np.prod(feature_shape))),
  View((-1, *feature_shape)),
  nn.BatchNorm2d(feature_shape[0]),
)