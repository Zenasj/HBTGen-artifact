# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape inferred from the Input layer of shape=(2,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_outputs=1):
        super(MyModel, self).__init__()
        self.num_outputs = num_outputs

        # Variables will be created in build() once input shape is known.
        self.W_ = None
        self.M_ = None
        self.GAM = None
        self.GM = None

    def build(self, input_shape):
        # input_shape is a TensorShape (batch_size, features)
        # Convert any Dimensions to int explicitly to avoid TypeErrors
        input_dim = int(input_shape[-1])
        output_dim = self.num_outputs
        shape = (input_dim, output_dim)

        initializer = tf.keras.initializers.GlorotUniform()

        # Following TensorFlow best practices, use add_weight instead of deprecated add_variable
        self.W_ = self.add_weight(
            name="W_",
            shape=shape,
            initializer=initializer,
            trainable=True,
            dtype=tf.float32,
        )
        self.M_ = self.add_weight(
            name="M_",
            shape=shape,
            initializer=initializer,
            trainable=True,
            dtype=tf.float32,
        )
        self.GAM = self.add_weight(
            name="GAM",
            shape=shape,
            initializer=initializer,
            trainable=True,
            dtype=tf.float32,
        )
        self.GM = self.add_weight(
            name="GM",
            shape=shape,
            initializer=initializer,
            trainable=True,
            dtype=tf.float32,
        )

        super(MyModel, self).build(input_shape)

    def call(self, x):
        # Gate activations
        gam = tf.sigmoid(tf.matmul(x, self.GAM))
        gm = tf.sigmoid(tf.matmul(x, self.GM))

        # Learned weights (bounded)
        W = tf.tanh(self.W_) * tf.sigmoid(self.M_)

        # Additive part
        add = tf.matmul(x, W)

        # Multiplicative part: exponentiate the log of abs(x)
        # Add small epsilon to avoid log(0)
        m = tf.exp(tf.matmul(tf.math.log(tf.abs(x) + 1e-14), W))

        # Mix multiplicative components with gating term gm
        mul = (-1.0 * m) * (1.0 - gm) + gm * m

        # Final output mixes additive and multiplicative parts gated by gam
        y = gam * add + (1.0 - gam) * mul
        return y

    def compute_output_shape(self, input_shape):
        # Output shape replaces last dimension with num_outputs
        shape = tf.TensorShape(input_shape).as_list()  # force to list of ints
        shape[-1] = self.num_outputs
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyModel, self).get_config()
        base_config['num_outputs'] = self.num_outputs
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def my_model_function():
    # Return an instance of MyModel with default output dimension 1
    return MyModel(num_outputs=1)


def GetInput():
    # Return a random input tensor matching the expected input shape (batch, 2)
    # Here batch size is chosen as 8 for example, dtype float32 as typical input
    return tf.random.uniform((8, 2), dtype=tf.float32)

