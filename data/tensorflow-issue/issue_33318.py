# tf.random.uniform((None, 4), dtype=tf.float32) ← inferred input shape from minimal reproducible example input X shape (32,4)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.ops import state_ops

def K_eval(x):
    # Custom fallback evaluation function compatible with both eager and graph mode,
    # inspired by the issue discussion for handling ResourceVariables and Graph tensors.
    try:
        return K.get_value(K.to_dense(x))
    except Exception:
        eval_fn = K.function([], [x])
        return eval_fn([])[0]

def _compute_eta_t(cls):
    # Cosine annealing computation for updating eta_t in the optimizer state.
    PI = 3.141592653589793
    t_frac = K.cast(cls.t_cur / cls.total_iterations, 'float32')
    eta_t = cls.eta_min + 0.5 * (cls.eta_max - cls.eta_min) * (1 + K.cos(PI * t_frac))
    return eta_t

class CustomSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False,
                 total_iterations=100, eta_min=0, eta_max=1,
                 t_cur=0, eta_t=1., **kwargs):
        super(CustomSGD, self).__init__(**kwargs)
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

        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        # Compute the updated eta_t using cosine annealing schedule
        self.eta_t = _compute_eta_t(self)

        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity update
            self.updates.append(state_ops.assign(m, v))

            if self.nesterov:
                p_t = p + self.momentum * v - lr * g
            else:
                p_t = p + v

            new_p = p_t

            # Apply constraint if any
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        # Use K_eval instead of K.eval for compatibility in graph mode with resource variables
        # Cast numeric values to standard Python types for config serialization
        config = {
            'lr': float(K.get_value(self.lr)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov,
            'total_iterations': int(self.total_iterations),
            'eta_t': float(K_eval(self.eta_t)),  # safer evaluation fallback
            't_cur': int(K.get_value(self.t_cur)),
            'eta_min': float(K.get_value(self.eta_min)),
            'eta_max': float(K.get_value(self.eta_max)),
        }
        base_config = super(CustomSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Minimal model as per the example: input shape (4,)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
        # Use our custom SGD optimizer to illustrate the fix context
        self.optimizer = CustomSGD(lr=1e-2)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # Forward pass with the dense layer
        return self.dense(inputs)

    @tf.function(jit_compile=True)
    def train_step(self, data):
        # A basic train step using custom optimizer showing usage
        x, y = data
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss = self.loss_fn(y, preds)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

def my_model_function():
    # Return an instance of MyModel with default initialization of layers and optimizer
    return MyModel()

def GetInput():
    # Generate a random input tensor compatible with MyModel:
    # Shape (batch_size=32, 4 features), float32 as common for inputs
    return tf.random.uniform((32, 4), dtype=tf.float32)

