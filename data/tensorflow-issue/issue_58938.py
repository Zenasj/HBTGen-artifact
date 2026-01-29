# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST dataset (batch size B, 28x28 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Base CNN model for MNIST classification
        self.base_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)  # logits output for 10 classes
        ])
        # Tracker for averaged loss metric
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        # Sparse categorical accuracy metric
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        
        # Obtain current distribution replica context for batch size info in distributed training
        replica_context = tf.distribute.get_replica_context()
        # Compute global batch size as per-replica batch size * number of replicas
        GLOBAL_BATCH_SIZE = replica_context.num_replicas_in_sync * tf.shape(y)[0]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Compute per-example loss (no reduction)
            per_example_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)(y, y_pred)
            # Compute average loss normalized by global batch size
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
            # If the model has any regularization losses, add those scaled losses
            if self.losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(self.losses))

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {'loss': self.loss_tracker.result(), 'accuracy': self.accuracy.result()}

    @property
    def metrics(self):
        # Return a list of metrics for resetting at epoch boundaries etc.
        return [self.loss_tracker, self.accuracy]

def my_model_function():
    # Instantiate and compile the model
    model = MyModel()
    # Compile with SGD optimizer with learning rate 0.001, no loss here since loss is computed in train_step
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))
    return model

def GetInput():
    # Generate a batch of random inputs corresponding to MNIST images
    # Batch size is chosen as 32 (typical per-replica batch size)
    batch_size = 32
    # Random normal values (or uniform) in [0,1), shape (batch_size, 28, 28)
    x = tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)
    # Random integer labels in [0, 9], shape (batch_size,)
    y = tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.int64)
    return (x, y)

