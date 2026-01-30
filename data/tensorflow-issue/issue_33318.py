import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

def K_eval(x):
    try:
        return K.get_value(K.to_dense(x))
    except:
        eval_fn = K.function([], [x])
        return eval_fn([])[0]

import tensorflow as tf
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution()

var = K.variable([2.], name='var')

try:
    print("K.get_value", K.get_value(var))
except:
    try:
        print("K.eval", K.eval(var))
    except:
        try:
            print("K.eager(K.get_value)", K.eager(K.get_value)(var))
        except:
            try:
                print("K.eager(K.eval)", K.eager(K.eval)(var))
            except:
                print("K_eval", K_eval(var))

K_eval [2.]

from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

ipt   = Input(shape=(4,))
out   = Dense(1,  activation='sigmoid')(ipt)
model = Model(ipt, out)
model.compile(SGD(lr=1e-2), loss='binary_crossentropy')

X = np.random.randn(32,4)
Y = np.random.randint(0,3,(32,1))
model.train_on_batch(X,Y)

model.save("path.h5")

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False,
                 total_iterations=100, eta_min=0, eta_max=1,
                 t_cur=0, init_verbose=True, **kwargs):
        eta_t = kwargs.pop('eta_t', 1.)
        super(SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.initial_decay = decay
        self.nesterov = nesterov
        self.total_iterations = total_iterations

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]
        self.updates.append(state_ops.assign_add(self.t_cur, 1))

        lr = self.lr

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        self.eta_t = _compute_eta_t(self)

        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(state_ops.assign(m, v))

            if self.nesterov:
                p_t = p + self.momentum * v - lr * g
            else:
                p_t = p + v

            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov,
            'total_iterations': int(self.total_iterations),
            'eta_t': int(K.eval(self.eta_t)),
            't_cur': int(K.get_value(self.t_cur)),
            'eta_min': int(K.get_value(self.eta_min)),
            'eta_max': int(K.get_value(self.eta_max)),
        }
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def _compute_eta_t(cls):
    PI = 3.141592653589793
    t_frac = K.cast(cls.t_cur / cls.total_iterations , 'float32')
    eta_t = cls.eta_min + 0.5 * (cls.eta_max - cls.eta_min) * \
        (1 + K.cos(PI * t_frac))
    return eta_t

import tensorflow as tf
import tensorflow.keras.backend as K
tf.compat.v1.disable_eager_execution()

var = K.variable([2.], name='var')

try:
    print("K.get_value", K.get_value(var))
except:
    try:
        print("K.eval", K.eval(var))
    except:
        try:
            print("K.eager(K.get_value)", K.eager(K.get_value)(var))
        except:
            try:
                print("K.eager(K.eval)", K.eager(K.eval)(var))
            except:
                print("K_eval", K_eval(var))

K_eval [2.]

# in __init__
self.step = K.variable(0, dtype='int64', name='step')

# in call
self.step = self.step + 1