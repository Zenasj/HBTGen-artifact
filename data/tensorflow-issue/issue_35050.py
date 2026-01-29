# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Inputs
        self.clf_input = tf.keras.layers.Input(shape=(32, 32, 3), name="model/input")
        self.label_ref = tf.keras.layers.Input(shape=(10,), name='label_ref')

        # Base pretrained model without weights, will be trained from scratch
        # Using ResNet50V2 with max pooling and 10 output classes
        self.base_model = tf.keras.applications.ResNet50V2(
            include_top=True,
            weights=None,
            input_tensor=self.clf_input,
            pooling='max',
            classes=10)

        # Forward pass from input tensor to logits
        self.clf_out = self.base_model(self.clf_input)

        # Optimizer: TF1-style Adam optimizer via compat.v1
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)

        # In TF1 graph mode (eager disabled), variables and operations are created 
        # as part of the graph, use tf.function to wrap forward+loss+train step for TF2.

        # Loss function: categorical crossentropy mean
        def c_loss():
            loss = tf.keras.losses.categorical_crossentropy(self.label_ref, self.clf_out)
            return tf.math.reduce_mean(loss)

        # Accuracy metric function: categorical accuracy mean
        def acc_metric():
            acc = tf.keras.metrics.categorical_accuracy(self.label_ref, self.clf_out)
            return tf.math.reduce_mean(acc)

        self.c_loss = c_loss
        self.op_acc = acc_metric

    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        # inputs: Tensor with shape (B,32,32,3)
        # We manually build model via base_model which uses the input tensor directly.
        # To enable compatibility with tf.function jit_compile=True, replicate forward pass here.
        # The original base_model was built on self.clf_input tensor. Here we run model layers directly.
        # So we can do:
        return self.base_model(inputs, training=training)

    @tf.function(jit_compile=True)
    def train_step(self, x, y):
        # Gradient tape for training step
        with tf.GradientTape() as tape:
            logits = self.call(x, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y, logits)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        # Compute accuracy
        accuracies = tf.keras.metrics.categorical_accuracy(y, logits)
        acc = tf.reduce_mean(accuracies)
        return loss, acc

def my_model_function():
    # Create a new instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor input compatible with MyModel's expected input shape
    # Batch size 100 (inferred from issue), 32x32 RGB input, float32 between 0 and 1
    batch_size = 100
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

