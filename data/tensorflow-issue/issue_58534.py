import numpy as np

import flax
from flax import linen as nn
from flax.training import train_state
from flax import jax_utils
from flax.core.frozen_dict import freeze, unfreeze
import jax
import jax.numpy as jnp
from jax import random, device_get
from typing import Sequence, Tuple, Optional, Callable, Any

ModuleDef = Any

class NN(nn.Module):
    features: Sequence[Sequence[int]]
    kernel_sizes: Sequence[Sequence[Tuple[int, int]]]
    used_rounds: Optional[int] = None
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        n_rounds = len(self.kernel_sizes) if self.used_rounds == None else self.used_rounds
        if n_rounds < 0:
            round_range = range(len(self.kernel_sizes) + n_rounds)
        else:
            round_range = range(n_rounds)
        
        y_rounds = jnp.zeros((len(list(round_range)), x.shape[0], 1))
        for boosting_round in round_range:
            y = None
            for depth in range(len(self.features[boosting_round])):
                if depth == (len(self.features[boosting_round]) - 1) and boosting_round == (len(list(round_range)) - 1):
                    conv_name = "conv_head"
                    dense_name = "dense_head"
                else:
                    conv_name = f"conv_{boosting_round}_{depth}"
                    dense_name = f"dense_{boosting_round}_{depth}"
                if y is not None:
                    y = jnp.concatenate((x, y), axis=-1)
                else:
                    y = x
                f = self.features[boosting_round][depth]
                k = self.kernel_sizes[boosting_round][depth]
                y = nn.Conv(features=f, kernel_size=k, name=conv_name)(y)
                y = self.act(y)
            
            y_flat = y.reshape((y.shape[0], -1))  # flatten
            y_dense = nn.Dense(features=1, name=dense_name)(y_flat)
            y_rounds = y_rounds.at[boosting_round].set(y_dense)
        return jnp.sum(y_rounds, axis=0)

tree_trial = NN([[2, 3, 4], [2, 3]], [[(2,), (4,), (5,)], [(2,), (4,)]])
params = tree_trial.init(random.PRNGKey(0), jnp.ones([33, 8]))['params']
print(jax.tree_map(lambda x: x.shape, params))

os.environ["XLA_FLAGS"]="--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["XLA_FLAGS"]="--xla_gpu_autotune_level=0"

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="1"