import math
from tensorflow import keras
from tensorflow.keras import optimizers

import tensorflow as tf

class MyAdamOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name="MyAdamOptimizer", **kwargs):
        super(MyAdamOptimizer, self).__init__(name, **kwargs)
        
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("epsilon", epsilon)
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            
    def _resource_apply_dense(self, grad, var):
        lr = self._get_hyper("learning_rate", var_dtype=var.dtype.base_dtype)
        beta_1 = self._get_hyper("beta_1", var_dtype=var.dtype.base_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype=var.dtype.base_dtype)
        epsilon = self._get_hyper("epsilon", var_dtype=var.dtype.base_dtype)
        
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        m.assign_add((1 - beta_1) * (grad - m))
        v.assign_add((1 - beta_2) * (tf.square(grad) - v))
        
        m_hat = m / (1 - tf.math.pow(beta_1, tf.cast(self.iterations + 1, tf.float32)))
        v_hat = v / (1 - tf.math.pow(beta_2, tf.cast(self.iterations + 1, tf.float32)))
        
        var_update = lr * m_hat / (tf.sqrt(v_hat) + epsilon)
        
        var.assign_sub(var_update)
        
        return var_update
        
    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    
optimizer = MyAdamOptimizer(learning_rate=0.001)