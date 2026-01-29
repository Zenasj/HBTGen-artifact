# tf.random.uniform((B, 6*6*6*32, 16), dtype=tf.float32)  ← Assumed input shape to DigitCapsuleLayer: batch size B, capsules=6*6*6*32, dim=16

import tensorflow as tf

def squash(vectors, axis=-1):
    """Squashing function corresponding to Eq. 1 from the Capsule Network paper.
    Args:
        vectors: some vectors to be squashed, N-D tensor
        axis: the axis to squash
    Returns:
        a tensor with same shape as input vectors
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    safe_norm = tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    squashed = scale * vectors / safe_norm
    return squashed

class DigitCapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules=2, input_capsules=6*6*6*32, input_dim=16, output_dim=8, routing_iters=2, **kwargs):
        super(DigitCapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules  # 2 capsules in this example
        self.input_capsules = input_capsules  # 6*6*6*32 capsules from previous layer
        self.input_dim = input_dim  # 16 dims per capsule input
        self.output_dim = output_dim  # 8 dims per output capsule
        self.routing_iters = routing_iters

    def build(self, input_shape):
        # Weight matrix W to transform input capsules to output capsules.
        # Shape: [num_capsules, input_capsules, input_dim, output_dim]
        # This weight shape is fixed as per code: [2, 6*6*6*32, 16, 8]
        self.W = self.add_weight(
            shape=(self.num_capsules, self.input_capsules, self.input_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights')
        super(DigitCapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: [batch_size, input_capsules, input_dim]
        batch_size = tf.shape(inputs)[0]
        # Expand dims to [batch_size, 1, input_capsules, input_dim]
        inputs_expanded = tf.expand_dims(inputs, 1)
        # Tile num_capsules times on axis=1: [batch_size, num_capsules, input_capsules, input_dim]
        inputs_tiled = tf.tile(inputs_expanded, [1, self.num_capsules, 1, 1])
        # Perform batch matrix multiplication between inputs and W
        # Using tf.map_fn to batch multiply each sample's inputs with weights
        # map_fn over batch dimension
        def matmul_per_example(x):
            # x shape: [num_capsules, input_capsules, input_dim]
            # self.W shape: [num_capsules, input_capsules, input_dim, output_dim]
            # Result shape: [num_capsules, input_capsules, output_dim]
            return tf.linalg.matmul(x, self.W)
        inputs_hat = tf.map_fn(matmul_per_example, elems=inputs_tiled)

        # Initialize routing logits b to zeros: shape [batch_size, num_capsules, input_capsules]
        b = tf.zeros(shape=[batch_size, self.num_capsules, self.input_capsules], dtype=tf.float32)

        for i in range(self.routing_iters):
            # Coupling coefficients c by softmax over num_capsules dim=1
            c = tf.nn.softmax(b, axis=1)
            # s: weighted sum of inputs_hat over input_capsules dim=2
            # c shape: [batch_size, num_capsules, input_capsules]
            # inputs_hat shape: [batch_size, num_capsules, input_capsules, output_dim]
            # multiply and sum over input_capsules:
            s = tf.reduce_sum(tf.expand_dims(c, -1) * inputs_hat, axis=2)  # shape: [batch_size, num_capsules, output_dim]
            # squash s to get output vectors v
            v = squash(s)
            if i < self.routing_iters - 1:
                # Update b by adding scalar product between inputs_hat and v
                # inputs_hat shape: [batch_size, num_capsules, input_capsules, output_dim]
                # v shape: [batch_size, num_capsules, output_dim]
                # we want to do batch dot product over output_dim between inputs_hat and v
                # Result shape of batch_dot: [batch_size, num_capsules, input_capsules]
                v_expanded = tf.expand_dims(v, 2)  # [batch_size, num_capsules, 1, output_dim]
                b += tf.reduce_sum(inputs_hat * v_expanded, axis=-1)

        return v  # output shape: [batch_size, num_capsules, output_dim]

def output_layer(inputs):
    # inputs shape is expected [batch_size, num_capsules, output_dim]
    # output is length (L2 norm) along last dim, sqrt(sum(square(x)) + epsilon)
    return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1) + tf.keras.backend.epsilon())

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Assuming input shape is [batch_size, 6*6*6*32, 16]
        # These dimensions are inferred from provided code snippet
        self.digit_capsule_layer = DigitCapsuleLayer()
        self.output_lambda = tf.keras.layers.Lambda(output_layer)

    def call(self, inputs):
        digit_caps = self.digit_capsule_layer(inputs)
        outputs = self.output_lambda(digit_caps)
        return outputs  # shape: [batch_size, num_capsules]

def my_model_function():
    return MyModel()

def GetInput():
    # Return random input tensor of shape [batch_size, input_capsules, input_dim]
    # batch size arbitrarily chosen as 4 for example
    batch_size = 4
    input_capsules = 6 * 6 * 6 * 32  # 6*6*6*32 = 6912
    input_dim = 16
    return tf.random.uniform(
        (batch_size, input_capsules, input_dim), dtype=tf.float32)

