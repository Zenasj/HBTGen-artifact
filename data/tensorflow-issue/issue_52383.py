# tf.random.uniform((B, 8), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers as described in the issue's example model
        self.dense1 = tf.keras.layers.Dense(12, activation='relu', input_shape=(8,))
        self.dense2 = tf.keras.layers.Dense(12, activation='relu')
        self.dense3 = tf.keras.layers.Dense(12, activation='relu')
        self.dense4 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def custom_loss_factory(model):
    """
    Creates a loss function that uses GradientTape to compute the gradient of 
    model output w.r.t. inputs, then computes a PDE residual-like loss term.
    This mirrors the user's goal from the issue:
    - derivative of output wrt input dimension 5 (index 5)
    - mean squared residual of that derivative
    
    Note: This loss must be passed to model.compile after model creation.
    """

    def customLoss(yTrue, yPred):
        # yPred is model(input) normally; yTrue is ground truth labels.
        # To get gradients w.r.t input, we must have access to input tensor.
        # The input tensor here is inside the tape watch.
        # However, Keras passes yTrue, yPred only - input tensor is detached.
        # Typical workaround is to capture inputs within the model or build wrapped function.
        # Here, we assume the input is passed additionally or use tf.Variable trick.
        #
        # Since Keras symbolic tensors are incompatible with GradientTape as watchables,
        # The input to gradient must be a real tensor.
        #
        # We rely on a captured input placeholder from closure or will pass input alongside yTrue/yPred.
        #
        # To keep a closure, this loss requires inputs to be passed via `yPred` substitution or using subclassed training loop.
        #
        # For demonstration:
        #
        # We'll assume yTrue is a tuple: (inputs, true_labels)
        # and yPred is model output.
        # So yTrue[0] holds inputs tensor.
        #
        # If user wants to use this loss with model.fit, they must feed inputs via yTrue accordingly.
        #
        # To avoid changing code logic deeply, we'll just show core logic that should be adapted in actual use.

        x_tensor = yTrue[0]  # inputs tensor
        y_true_labels = yTrue[1]  # original true labels

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            output = model(x_tensor)
        DyDX = tape.gradient(output, x_tensor)  # shape (batch_size, input_dim=8)

        # Extract derivative w.r.t. 6th input dimension (index 5, per user code)
        dy_t = DyDX[:, 5:6]  # shape (batch_size, 1)

        # PDE residual loss: mean squared of dy/dx6
        loss_PDE = tf.reduce_mean(tf.square(dy_t))

        # Optionally classical data loss (commented out as per user's minimal example):
        # loss_data = tf.reduce_mean(tf.square(y_true_labels - yPred))

        # Return just PDE loss as in their stripped down example
        return loss_PDE

    return customLoss

def GetInput():
    # Produce a random float32 input tensor matching model input shape (batch_size, 8)
    # Using batch size of 10, as per example fit(batch_size=10)
    return tf.random.uniform((10, 8), dtype=tf.float32)

