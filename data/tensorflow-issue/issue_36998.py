# tf.random.uniform((None, 784), dtype=tf.float32) ← Assuming MNIST-like input with batch size None and 784 features (28x28 flattened)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Composite model containing two submodels:
      - keras_model: Keras subclassed Model with recommended API usage,
      - custom_model: User-implemented custom Model demonstrating the issues from the reported bug.

    The call method runs both models on the same input and returns a dictionary
    including predictions, losses, and diagnostics comparing the two models.

    This encapsulates the issue described: 
    - Keras model trains well with or without @tf.function
    - Custom model fails to train properly without @tf.function
    - Difference potentially due to softmax+cross-entropy usage and weight initialization
    """

    def __init__(self):
        super().__init__()
        # Use consistent weight initializer similar to Keras default GlorotUniform for comparison
        kernel_init = tf.keras.initializers.GlorotUniform()

        # Keras Model Submodule (recommended usage)
        class KerasModelInner(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=kernel_init)
                self.dense2 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=kernel_init)
                self.out_layer = tf.keras.layers.Dense(10, kernel_initializer=kernel_init)  # no softmax here

            def call(self, x):
                x = self.dense1(x)
                x = self.dense2(x)
                return self.out_layer(x)  # logits output

        # Custom Model Submodule mimicking the problematic user custom model (fixed issues applied)
        class CustomModelInner(tf.keras.Model):
            def __init__(self):
                super().__init__()
                # Initialize weights like Keras (GlorotUniform) to match performance better
                self.w1 = tf.Variable(kernel_init(shape=(784, 512)), trainable=True, name='w1')
                self.b1 = tf.Variable(tf.zeros([512]), trainable=True, name='b1')
                self.w2 = tf.Variable(kernel_init(shape=(512, 512)), trainable=True, name='w2')
                self.b2 = tf.Variable(tf.zeros([512]), trainable=True, name='b2')
                self.w3 = tf.Variable(kernel_init(shape=(512, 10)), trainable=True, name='w3')
                self.b3 = tf.Variable(tf.zeros([10]), trainable=True, name='b3')

            @property
            def trainable_vars(self):
                return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

            def call(self, x):
                z1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
                z2 = tf.nn.relu(tf.matmul(z1, self.w2) + self.b2)
                logits = tf.matmul(z2, self.w3) + self.b3  # no softmax (important: use logits directly!)
                return logits

        self.keras_model = KerasModelInner()
        self.custom_model = CustomModelInner()

        # Loss Function:
        # Use tf.nn.sparse_softmax_cross_entropy_with_logits which combines softmax + cross entropy properly
        self.loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    def call(self, inputs, labels=None, training=False):
        """
        Forward pass:
         - Compute logits for both models
         - Optionally compute losses if labels provided
         - Outputs a dict with logits, losses, and difference metrics

        Args:
          inputs: batch of input samples, shape = (batch_size, 784)
          labels: (optional) integer class labels for calculation of loss
          training: (optional) flag for training mode (passed through, no dropout here)

        Returns:
          A dict with keys:
            'logits_keras': Keras model logits (batch,10)
            'logits_custom': Custom model logits (batch,10)
            'loss_keras': scalar loss tensor (if labels provided)
            'loss_custom': scalar loss tensor (if labels provided)
            'logit_diff_norm': L2 norm difference between logits
        """
        logits_keras = self.keras_model(inputs)
        logits_custom = self.custom_model(inputs)

        loss_keras = None
        loss_custom = None
        if labels is not None:
            loss_keras = tf.reduce_mean(self.loss_fn(labels, logits_keras))
            loss_custom = tf.reduce_mean(self.loss_fn(labels, logits_custom))

        logit_diff_norm = tf.norm(logits_keras - logits_custom)  # L2 norm to measure difference

        return {
            'logits_keras': logits_keras,
            'logits_custom': logits_custom,
            'loss_keras': loss_keras,
            'loss_custom': loss_custom,
            'logit_diff_norm': logit_diff_norm,
        }


def my_model_function():
    # Instantiate and return the composed model
    return MyModel()


def GetInput():
    # Return a random input tensor shaped like flattened MNIST images
    # Batch size arbitrarily chosen as 32 here
    B = 32
    H = 1
    W = 1
    C = 784
    # Return float32 tensor emulating batch of flattened 28x28 images
    return tf.random.uniform((B, C), minval=0.0, maxval=1.0, dtype=tf.float32)

