from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops

class DropConnectDense(Dense):
    def __init__(self, *args, **kwargs):
        self.rate = kwargs.pop('rate', 0.5)
        if 0. < self.rate < 1.:
            self.uses_learning_phase = True

        super(DropConnectDense, self).__init__(*args, **kwargs)

    def call(self, inputs):

        if 0. < self.rate < 1.:
            kernel = K.in_train_phase(nn_ops.dropout(self.kernel, rate = self.rate), self.kernel)
            bias = K.in_train_phase(nn_ops.dropout(self.bias, rate = self.rate), self.bias)
        else:
            kernel = self.kernel
            bias = self.bias
            self.kernel = kernel
            self.bias = bias

        outputs = gen_math_ops.MatMul(a = inputs, b = kernel)
        if self.use_bias:
          outputs = nn_ops.bias_add(outputs, bias)

        if self.activation is not None:
          outputs = self.activation(outputs)
        return outputs


    def get_config(self):
        config = super(DropConnectDense, self).get_config()
        config.update({
            'rate': self.rate,
            'uses_learning_phase': self.uses_learning_phase,
        })

        return config