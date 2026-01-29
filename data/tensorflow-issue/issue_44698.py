# tf.random.uniform((GLOBAL_BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np
import os

# Assumptions and reconstruction notes:
# - The main model is a simple CNN for classification on image data shaped (28,28,1).
# - Input batch size is GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * number_of_replicas.
# - Strategy is MirroredStrategy across all detected GPUs.
# - Loss scaling performed via tf.nn.compute_average_loss with proper global batch size.
# - Distributed train/test steps wrapped with tf.function with strategy.experimental_run_v2 (TF 2.3 style).
#   (In TF 2.20, experimental_run_v2 is renamed to run(), but we keep as in examples.)
# - The example aligns with the official TensorFlow custom training loop with distribution strategy.
# - The code reconstructs the core model, training, loss, and distributed logic, suitable for XLA compilation.
# - No dataset loading or checkpointing included here, only model + distributed training logic and inputs.
# - The input tensor shape and dtype are documented in the comment as requested.

strategy = tf.distribute.MirroredStrategy()

# Parameters for dataset and model training
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10
BUFFER_SIZE = 60000  # Approximate, assuming MNIST-like train set size

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Instantiate model, optimizer, loss, metrics under strategy scope
    with strategy.scope():
        model = MyModel()
        optimizer = tf.keras.optimizers.Adam()

        # Use SparseCategoricalCrossentropy with no reduction (to control loss scaling)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
        
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            # Scale loss by global batch size when training on multiple GPUs
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        @tf.function
        def train_step(inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_accuracy.update_state(labels, predictions)
            return loss

        @tf.function
        def test_step(inputs):
            images, labels = inputs
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
            scaled_loss = tf.nn.compute_average_loss(t_loss, global_batch_size=GLOBAL_BATCH_SIZE)
            test_loss.update_state(scaled_loss)
            test_accuracy.update_state(labels, predictions)

        @tf.function
        def distributed_train_step(dataset_inputs):
            # strategy.experimental_run_v2 runs train_step on each replica input and returns per replica losses
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            # Reduce the per replica losses across replicas by summation
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

        # Return all relevant components so they can be used outside
        class TrainHelper:
            pass

        h = TrainHelper()
        h.model = model
        h.optimizer = optimizer
        h.compute_loss = compute_loss
        h.train_accuracy = train_accuracy
        h.test_accuracy = test_accuracy
        h.test_loss = test_loss
        h.train_step = train_step
        h.test_step = test_step
        h.distributed_train_step = distributed_train_step
        h.distributed_test_step = distributed_test_step

        return h

def GetInput():
    # Produce a random tensor matching shape of input expected by MyModel
    # Input shape: (GLOBAL_BATCH_SIZE, 28, 28, 1), dtype float32 in [0,1] range
    return tf.random.uniform((GLOBAL_BATCH_SIZE, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

