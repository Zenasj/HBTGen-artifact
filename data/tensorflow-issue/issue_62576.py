# tf.random.uniform((BS, 10, 2), dtype=tf.float32)  ← Inferred input shape from dataShape=(10,2) and batch size BS=10

import tensorflow as tf

maxValue = 1 - tf.keras.backend.epsilon()
BS = 10
dataShape = (10, 2)
outShape = 1


class ConstMul(tf.keras.layers.Layer):
    def __init__(self, const_val, **kwargs):
        super().__init__(**kwargs)
        # Use a tf.Variable to allow mutation at runtime for enabling/disabling gradient flow
        self.const = tf.Variable(const_val, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        # Multiply inputs by the current value of const
        return inputs * self.const


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # GRU layer as shared 'body'
        self.gru = tf.keras.layers.GRU(units=10, unroll=True, name="gru")

        # Two heads with controllable gradient flow via ConstMul layers
        self.head1_ctrl = ConstMul(1.0, name="head1")
        self.head2_ctrl = ConstMul(0.0, name="head2")

        # Output dense layers without bias
        self.out1 = tf.keras.layers.Dense(outShape, use_bias=False)
        self.out2 = tf.keras.layers.Dense(outShape, use_bias=False)

    def call(self, inputs, training=False):
        x = inputs
        x = self.gru(x, training=training)
        # Apply the controllable multipliers before each head
        x1 = self.head1_ctrl(x)
        x2 = self.head2_ctrl(x)
        out1 = self.out1(x1)
        out2 = self.out2(x2)
        return [out1, out2]

    def set_head_active(self, head_index):
        """
        Utility method to switch active head.
        head_index: 1 or 2, activates that head (multiplier=1) and disables the other (multiplier=0)
        """
        if head_index == 1:
            self.head1_ctrl.const.assign(maxValue)
            self.head2_ctrl.const.assign(0.0)
        elif head_index == 2:
            self.head1_ctrl.const.assign(0.0)
            self.head2_ctrl.const.assign(maxValue)
        else:
            raise ValueError("head_index must be 1 or 2")

def my_model_function():
    model = MyModel()
    # Build the model by calling it once on input shape (None, 10, 2)
    model(tf.random.uniform((1, *dataShape)))
    # Compile with binary crossentropy and Adam optimizer as in examples
    model.compile(
        loss="binary_crossentropy",
        optimizer='Adam',
    )
    return model


def GetInput():
    """
    Returns a random tensor input matching the input expected by MyModel.
    Shape: (BS, 10, 2), dtype float32
    """
    return tf.random.uniform((BS, *dataShape), dtype=tf.float32)

