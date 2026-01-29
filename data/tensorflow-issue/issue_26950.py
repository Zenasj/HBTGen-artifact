# tf.random.uniform((B, 96, 96, 3), dtype=tf.float32) ← inferred input shape from TensorSpec logs in issue, batch size unknown at class-level 

import tensorflow as tf
import re

class MyModel(tf.keras.Model):
    # A dummy model to match the example issue context, since original model "res_model" is undefined.
    # This model accepts input shape (batch, 96, 96, 3) and outputs a scalar per example.
    def __init__(self):
        super().__init__()
        # Simple conv + global pooling + dense layer for binary output
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        return self.dense(x)

class AdaBoundOptimizer(tf.train.Optimizer):
    """AdaBound optimizer implementation compatible with TensorFlow 1.x style.
    Adapted and fixed to properly create slot variables (_create_slots) 
    instead of creating variables inside apply_gradients as in the reported issue.
    """

    def __init__(self,
                 learning_rate=0.001,
                 final_lr=0.1,
                 beta1=0.9,
                 beta2=0.999,
                 gamma=1e-3,
                 epsilon=1e-8,
                 amsbound=False,
                 decay=0.,
                 weight_decay=0.,
                 exclude_from_weight_decay=None,
                 use_locking=False, name="AdaBound"):
        super(AdaBoundOptimizer, self).__init__(use_locking, name)

        if final_lr <= 0.:
            raise ValueError("Invalid final learning rate : {}".format(final_lr))
        if not 0. <= beta1 < 1.:
            raise ValueError("Invalid beta1 value : {}".format(beta1))
        if not 0. <= beta2 < 1.:
            raise ValueError("Invalid beta2 value : {}".format(beta2))
        if not 0. <= gamma < 1.:
            raise ValueError("Invalid gamma value : {}".format(gamma))
        if epsilon <= 0.:
            raise ValueError("Invalid epsilon value : {}".format(epsilon))

        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._final_lr = final_lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._amsbound = amsbound
        self._decay = decay
        self._weight_decay = weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay
        self._base_lr = learning_rate

    def _create_slots(self, var_list):
        # Create optimizer slots (m, v, v_hat if amsbound)
        for var in var_list:
            param_name = self._get_variable_name(var.name)
            self._zeros_slot(var, "adabound_m", self._name)
            self._zeros_slot(var, "adabound_v", self._name)
            if self._amsbound:
                self._zeros_slot(var, "adabound_v_hat", self._name)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        lr = self._lr
        t = tf.cast(global_step, dtype=tf.float32)

        if self._decay > 0.:
            lr *= (1. / (1. + self._decay * t))

        t += 1

        bias_correction1 = 1. - tf.pow(self._beta1, t)
        bias_correction2 = 1. - tf.pow(self._beta2, t)
        step_size = (lr * tf.sqrt(bias_correction2) / bias_correction1)

        final_lr = self._final_lr * lr / self._base_lr
        lower_bound = final_lr * (1. - 1. / (self._gamma * t + 1.))
        upper_bound = final_lr * (1. + 1. / (self._gamma * t))

        assignments = []
        for grad, param in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = self.get_slot(param, "adabound_m")
            v = self.get_slot(param, "adabound_v")
            if self._amsbound:
                v_hat = self.get_slot(param, "adabound_v_hat")

            m_t = self._beta1 * m + (1. - self._beta1) * grad
            v_t = self._beta2 * v + (1. - self._beta2) * tf.square(grad)

            if self._amsbound:
                v_hat_t = tf.maximum(v_hat, v_t)
                denom = tf.sqrt(v_hat_t) + self._epsilon
            else:
                denom = tf.sqrt(v_t) + self._epsilon

            step_size_p = step_size * tf.ones_like(denom)
            step_size_p_bound = step_size_p / denom

            lr_t = m_t * tf.clip_by_value(step_size_p_bound,
                                          clip_value_min=lower_bound,
                                          clip_value_max=upper_bound)
            p_t = param - lr_t

            if self._do_use_weight_decay(param_name):
                p_t += self._weight_decay * param

            update_list = [param.assign(p_t), m.assign(m_t), v.assign(v_t)]
            if self._amsbound:
                update_list.append(v_hat.assign(v_hat_t))

            assignments.extend(update_list)

        if global_step is not None:
            assignments.append(global_step.assign_add(1))

        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self._weight_decay:
            return False
        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    @staticmethod
    def _get_variable_name(param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Build the model to create weights (e.g. call once)
    example_input = GetInput()
    model(example_input)
    return model

def GetInput():
    # Return a random tensor input that matches expected input shape.
    # Assumed batch size 8 for example; H=96, W=96, C=3 as per issue logs.
    return tf.random.uniform((8, 96, 96, 3), dtype=tf.float32)

