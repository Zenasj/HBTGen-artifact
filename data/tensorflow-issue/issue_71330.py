# tf.random.uniform((32, 224, 224, 3), dtype=tf.float32) ← Input batch size 32, height 224, width 224, 3 channels

import tensorflow as tf
import numpy as np

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_SAMPLES = 1000

class TestMeanIou(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Override to convert y_true to int32 and use argmax on y_pred
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(NUM_CLASSES, kernel_size=1, activation='softmax', name='output')
        self.metric = TestMeanIou(num_classes=NUM_CLASSES)

    def call(self, inputs, training=False):
        # Run a forward pass through conv layer.
        preds = self.conv(inputs)
        return preds

    @tf.function
    def train_step(self, data):
        # Custom training step to incorporate metric update and compute loss manually,
        # as we want to ensure metric sees batch tensors properly.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metric explicitly
        self.metric.update_state(y, y_pred)

        # Update other compilation metrics if any
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current values
        results = {'loss': loss}
        results.update({m.name: m.result() for m in self.metrics})
        return results

    def reset_metrics(self):
        # Reset metric state at start of each epoch or when needed
        self.metric.reset_state()
        for m in self.metrics:
            m.reset_state()

def my_model_function():
    # Instantiate and compile the model with loss and metrics
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[model.metric]
    )
    return model

def generate_dataset(num_samples):
    # Use numpy random to generate fixed random inputs and masks to match the example
    # This mimics the generator from the issue.
    np.random.seed(1)
    input_images = np.random.rand(num_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
    masks = np.random.randint(0, NUM_CLASSES, size=(num_samples, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.int32)
    for i in range(num_samples):
        yield input_images[i], masks[i]

def GetInput():
    """
    Returns a batch of random input tensors matching the model input shape, 
    and a batch of random integer masks as used in the issue's generator.
    This is a tuple of (inputs, masks) matching model inputs and expected labels.
    """
    # Generate a batch of inputs and masks using numpy to keep consistent with example
    np.random.seed(1)
    input_batch = np.random.rand(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
    mask_batch = np.random.randint(
        0, NUM_CLASSES, size=(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.int32
    )
    return tf.convert_to_tensor(input_batch), tf.convert_to_tensor(mask_batch)

