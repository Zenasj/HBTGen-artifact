# tf.random.uniform((B, 32), dtype=tf.float32) ← inferred input shape (batch size B, features 32)
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two simple submodels, each with one Dense layer of 32 units
        self.net1 = CustomModel1()
        self.net2 = CustomModel2()

    def call(self, inputs):
        # Forward pass calls net1 then net2, returns both outputs as in original code
        z = self.net1(inputs)
        x = self.net2(z)
        return z, x

    def train_step(self, data):
        # Custom train step illustrating the original problematic behavior:
        # Calling submodels explicitly instead of self.call to reproduce the issue
        x, y = data

        with tf.GradientTape() as tape:
            # Calling submodels directly triggers save/load error in original issue
            y_pred = self.net2(self.net1(x))  
            loss = self.compiled_loss(y, y_pred)

        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

@tf.keras.utils.register_keras_serializable()
class CustomModel1(Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(32)

    def call(self, inputs):
        x = self.dense(inputs)
        return x

@tf.keras.utils.register_keras_serializable()
class CustomModel2(Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(32)

    def call(self, inputs):
        x = self.dense(inputs)
        return x

def my_model_function():
    # Returns an instance of MyModel, ready for compilation/training
    return MyModel()

def GetInput():
    # Generate a batch of random inputs with shape (batch_size, 32)
    # Assuming typical batch size of 8 for example, dtype float32
    batch_size = 8
    return tf.random.uniform((batch_size, 32), dtype=tf.float32)

"""
Notes / Assumptions:
- Input shape: (batch_size, 32), as seen from the Dense layer input dimension and training example.
- Two submodels, each one Dense(32), composed inside MyModel as net1 and net2.
- train_step overridden to call net1 and net2 explicitly inside, mimicking the original code's issue.
- Registered the CustomModel1 and CustomModel2 classes with @register_keras_serializable for loading/saving support.
- Did not modify behavior to fix saving/loading, as the original issue is about the bug when calling submodels in train_step.
- The primary goal is capturing the model structure and usage pattern that leads to the error,
  not fixing the bug or changing the logic.
- No external dependencies other than TensorFlow 2.15+.
"""

