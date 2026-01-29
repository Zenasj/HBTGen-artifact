# tf.random.uniform((B, 100), dtype=tf.float32) ← Input shape is (batch_size, 100) as seen in original code placeholders and inputs

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model internally implements two versions of a simple dense + softmax classification:
    1) A native tf.keras model using tf.keras layers and Adam optimizer.
    2) A low-level TensorFlow model constructing the same architecture using tf layers and manual Adam optimizer.
    
    The forward pass runs both models' forward computations and returns:
    - The weights updates delta (after one-step gradient application) from each model for comparison,
    - The gradients calculated for each,
    - The boolean tensor indicating if weights updates and gradients are close within a tolerance.
    
    Assumptions and Notes:
    - Input shape: (batch_size, 100)
    - Output units: 2 (binary classification)
    - Loss: categorical crossentropy / softmax cross entropy
    - Adam hyperparameters are fixed according to the original snippet
    - One training step is applied internally for both models with the same data input
    - This fused model is helpful for comparing gradient/weights update consistency.
    """
    def __init__(self):
        super().__init__()
        self.input_dim = 100
        self.output_units = 2
        self.lr = 1e-3
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.seed = 10
        
        # Initialize the keras-style model (tf.keras.Model subclass components)
        initializer = tf.keras.initializers.RandomNormal(seed=self.seed)
        self.keras_dense = tf.keras.layers.Dense(
            units=self.output_units,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            name='keras_dense'
        )
        self.softmax = tf.keras.layers.Activation('softmax')
        # Keras Adam optimizer
        self.keras_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon
        )
        
        # Initialize variables for the TF "native" model part
        # These will be tf.Variable matching the keras_dense layer weights for direct comparison
        # Kernel and bias initialized same as keras_dense
        # This simulates the low-level TF model's weights
        # Use seed to initialize similarly
        tf.random.set_seed(self.seed)
        kernel_init = tf.random.normal([self.input_dim, self.output_units], seed=self.seed)
        bias_init = tf.random.normal([self.output_units], seed=self.seed + 1)  # different seed for bias
        
        self.tf_kernel = tf.Variable(kernel_init, dtype=tf.float32, name='tf_kernel')
        self.tf_bias = tf.Variable(bias_init, dtype=tf.float32, name='tf_bias')
        
        # Adam optimizer variables for tf native part
        self.tf_optimizer = tf.optimizers.Adam(
            learning_rate=self.lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon
        )
        
        # Small tolerance for floating point close checks
        self.tolerance = 1e-6
        
    def call(self, inputs, training=False):
        """
        Forward pass:
        - inputs: float32 tensor of shape (batch_size, 100)
        
        Returns:
        A dict with keys:
        'keras_weight_update', 'keras_gradients',
        'tf_weight_update', 'tf_gradients',
        'weights_close' (bool tensor), 'grads_close' (bool tensor).
        
        This runs one step of gradient calculation and one optimization step on both models.
        """
        
        # ---- Run keras submodel ----
        with tf.GradientTape() as keras_tape:
            keras_logits = self.keras_dense(inputs)
            keras_pred = self.softmax(keras_logits)
            keras_loss = tf.keras.losses.categorical_crossentropy(
                tf.ones_like(keras_pred),  # dummy to get shape; real label used below
                keras_pred,
                from_logits=False
            )
            # We will override loss by manually computing the categorical crossentropy with actual labels later
            # So here just use placeholder; calculation will be repeated properly below
        
        # But we need the actual loss and gradients using labels: we will do it shortly...
        # Since no labels are passed to call, we assume the user will pass labels as inputs[1] (tuple)
        # To keep the interface, assume inputs provided as (features, labels) for training comparison.
        # If inputs is a tuple, unpack. Else raise error
        # This aligns with the original script which uses datas as x, labels as y.
        
        if isinstance(inputs, tuple) and len(inputs) == 2:
            x_input, labels = inputs
        else:
            raise ValueError(
                "Expected input as tuple (features, labels) for training comparison; got: {}".format(type(inputs))
            )
        
        # Keras model forward + loss
        keras_logits = self.keras_dense(x_input)
        keras_pred = self.softmax(keras_logits)
        keras_loss_value = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(labels, keras_pred)
        )
        # Compute gradients w.r.t trainable variables
        keras_gradients = keras_tape.gradient(keras_loss_value, self.keras_dense.trainable_variables)
        
        # Save old weights
        keras_weights_before = [tf.identity(w) for w in self.keras_dense.trainable_variables]
        # Apply one step update
        self.keras_optimizer.apply_gradients(zip(keras_gradients, self.keras_dense.trainable_variables))
        keras_weights_after = self.keras_dense.trainable_variables
        
        keras_weight_update = [after - before for before, after in zip(keras_weights_before, keras_weights_after)]
        
        # ---- Run TF native submodel ----
        # Forward pass using tf kernel and bias variables
        with tf.GradientTape() as tf_tape:
            tf_logits = tf.matmul(x_input, self.tf_kernel) + self.tf_bias
            tf_pred = tf.nn.softmax(tf_logits)
            tf_loss_value = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=tf_logits)
            )
        # Compute gradients w.r.t tf_kernel, tf_bias
        tf_gradients = tf_tape.gradient(tf_loss_value, [self.tf_kernel, self.tf_bias])
        
        # Save old weights
        tf_weights_before = [tf.identity(self.tf_kernel), tf.identity(self.tf_bias)]
        # Apply one step update
        self.tf_optimizer.apply_gradients(zip(tf_gradients, [self.tf_kernel, self.tf_bias]))
        tf_weights_after = [self.tf_kernel, self.tf_bias]
        
        tf_weight_update = [after - before for before, after in zip(tf_weights_before, tf_weights_after)]
        
        # Check closeness of weight updates and gradients for debugging
        weights_close = [tf.reduce_all(tf.abs(k_update - t_update) < self.tolerance)
                         for k_update, t_update in zip(keras_weight_update, tf_weight_update)]
        grads_close = [tf.reduce_all(tf.abs(k_grad - t_grad) < self.tolerance)
                       for k_grad, t_grad in zip(keras_gradients, tf_gradients)]
        
        # Aggregate boolean results into single tensors (for weights and grads)
        weights_close_all = tf.reduce_all(weights_close)
        grads_close_all = tf.reduce_all(grads_close)
        
        return {
            'keras_weight_update': keras_weight_update,
            'keras_gradients': keras_gradients,
            'tf_weight_update': tf_weight_update,
            'tf_gradients': tf_gradients,
            'weights_close': weights_close_all,
            'grads_close': grads_close_all,
        }

def my_model_function():
    # Returns an instance of the fused MyModel
    return MyModel()

def GetInput():
    # Return a tuple (features, labels) matching the model input expectations
    
    # According to original code:
    # - features shape: (batch_size, 100), batch_size inferred as 2 to cover two samples as in user data
    # - labels shape: (batch_size, 2), one-hot labels
    
    batch_size = 2
    features = tf.ones((batch_size, 100), dtype=tf.float32)
    labels = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    return (features, labels)

