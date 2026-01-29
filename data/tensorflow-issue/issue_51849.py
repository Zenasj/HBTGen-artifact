# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← input shape from the original model's input signature (batch_size, IMG_SIZE, IMG_SIZE)

import tensorflow as tf

IMG_SIZE = 28
NUM_CLASSES = 10

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # replicate the sequential model from original code
        self.flatten = tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        # Loss and optimizer instances (for training method)
        self._LOSS_FN = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self._OPTIM = tf.optimizers.SGD()

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32),
    ])
    def train(self, x, y):
        # Custom training logic matching original train function
        with tf.GradientTape() as tape:
            prediction = self.call(x)
            loss = self._LOSS_FN(y, prediction)  # Correct order in tf2: loss(y_true, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self._OPTIM.apply_gradients(zip(gradients, self.trainable_variables))
        # Return loss and gradients keyed by name for compatibility with original signature
        result = {"loss": loss}
        for grad, var in zip(gradients, self.trainable_variables):
            # grad.name does not exist, use var.name as key (gradients don't have .name)
            result[var.name] = grad
        return result

    @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32)])
    def predict(self, x):
        # Return dictionary containing prediction (softmax probabilities)
        return {
            "output": self.call(x)
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        # Save the model weights to the given checkpoint path using raw_ops
        tensor_names = [weight.name for weight in self.weights]
        tensors_to_save = [weight.read_value() for weight in self.weights]
        tf.raw_ops.Save(filename=checkpoint_path,
                        tensor_names=tensor_names,
                        data=tensors_to_save,
                        name='save')
        return {"checkpoint_path": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        # Restore model weights from the given checkpoint path
        restored_tensors = {}
        for var in self.weights:
            restored = tf.raw_ops.Restore(file_pattern=checkpoint_path,
                                          tensor_name=var.name,
                                          dt=var.dtype,
                                          name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return a new instance of MyModel
    return MyModel()

def GetInput():
    # Generate a sample input tensor with shape (batch_size, 28, 28)
    # Use batch_size = 1 as a reasonable default for testing
    return tf.random.uniform((1, IMG_SIZE, IMG_SIZE), dtype=tf.float32)

