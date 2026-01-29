# tf.random.uniform((B, 48, 48, 1), dtype=tf.float32)
import tensorflow as tf

# Assumptions & notes:
# - Original code is TF 1.x with eager execution and tf.layers, but for modern TF 2.20.0 compatibility,
#   tf.keras.layers is used instead of tf.layers (since tf.layers is deprecated).
# - The input shape is inferred as (batch, 48, 48, 1) from dummy_input in restore_model().
# - The model is a CNN for emotion recognition with 7 classes.
# - Dropout layers use rate and training argument for conditional dropout.
# - BatchNormalization layers instantiated using tf.keras.layers.BatchNormalization as in chunk 2/5.
# - Predictor returns flattened logits (for loss and softmax cross entropy).
# - For XLA compatibility, avoid eager-only constructs such as tfe.* (TensorFlow 1.x eager APIs),
#   use native TF 2.x APIs.
# - Save/load weights replaced by tf.train.Checkpoint to avoid .meta file issues.
# - The device string like "gpu:0" is handled via tf.device context.
# - Optimizer is passed externally to fit function.
# - The metrics and iterators adapted to tf.data.Dataset usage.
# - To keep the same interface, method names and signatures preserved where applicable.
# - The loss function is sparse_softmax_cross_entropy with logits.
# - For clarity and compatibility, predict renamed from predict() to call() special method.

class MyModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        # Define layers similarly to provided code, but use tf.keras.layers
        self.conv1 = tf.keras.layers.Conv2D(16, 5, padding='same', activation=None)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(16, 5, strides=2, padding='same', activation=None)
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(32, 5, padding='same', activation=None)
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation=None)
        self.batch4 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv2D(64, 3, padding='same', activation=None)
        self.batch5 = tf.keras.layers.BatchNormalization()
        self.conv6 = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation=None)
        self.batch6 = tf.keras.layers.BatchNormalization()
        self.conv7 = tf.keras.layers.Conv2D(64, 1, padding='same', activation=None)
        self.batch7 = tf.keras.layers.BatchNormalization()
        self.conv8 = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation=None)
        self.batch8 = tf.keras.layers.BatchNormalization()
        self.conv9 = tf.keras.layers.Conv2D(256, 1, padding='same', activation=None)
        self.batch9 = tf.keras.layers.BatchNormalization()
        self.conv10 = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation=None)
        self.conv11 = tf.keras.layers.Conv2D(256, 1, padding='same', activation=None)
        self.batch11 = tf.keras.layers.BatchNormalization()
        self.conv12 = tf.keras.layers.Conv2D(num_classes, 3, strides=2, padding='same', activation=None)

        # Dropout rates from original: 0.4, then 0.3 multiple times
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.dropout4 = tf.keras.layers.Dropout(0.3)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, images, training=False):
        # Forward pass reflecting predict function from issue
        x = self.conv1(images)
        x = self.batch1(x, training=training)
        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        x = self.conv3(x)
        x = self.batch3(x, training=training)
        x = self.conv4(x)
        x = self.batch4(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)
        x = self.conv5(x)
        x = self.batch5(x, training=training)
        x = self.conv6(x)
        x = self.batch6(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout3(x, training=training)
        x = self.conv7(x)
        x = self.batch7(x, training=training)
        x = self.conv8(x)
        x = self.batch8(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout4(x, training=training)
        x = self.conv9(x)
        x = self.batch9(x, training=training)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.batch11(x, training=training)
        x = self.conv12(x)
        # Output flattened logits for classification (num_classes)
        x = self.flatten(x)
        return x

    def loss_fn(self, images, targets, training=False):
        logits = self.call(images, training=training)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
        return loss

    def grads_fn(self, images, targets, training=True):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(images, targets, training)
        return tape.gradient(loss, self.trainable_variables), loss

    def compute_accuracy(self, dataset):
        # dataset expected as tf.data.Dataset yielding (images, targets)
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for images, targets in dataset:
            logits = self.call(images, training=False)
            accuracy.update_state(targets, logits)
        return accuracy.result().numpy()

    def fit(self, train_dataset, eval_dataset, optimizer,
            num_epochs=500, early_stopping_rounds=10, verbose=10, train_from_scratch=False,
            checkpoint_dir=None):
        # Adapted fit method with tf.train.Checkpoint usage and early stopping

        # Prepare checkpoint manager if checkpoint_dir provided
        if checkpoint_dir:
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
            if not train_from_scratch:
                checkpoint.restore(manager.latest_checkpoint)
                if manager.latest_checkpoint:
                    print(f"Restored from {manager.latest_checkpoint}")
                else:
                    print("No checkpoint found, training from scratch")
        else:
            checkpoint = None
            manager = None

        best_loss = float('inf')
        count = early_stopping_rounds

        train_loss_metric = tf.keras.metrics.Mean()
        eval_loss_metric = tf.keras.metrics.Mean()

        for epoch in range(num_epochs):
            # Training loop
            for images, targets in train_dataset:
                grads, loss = self.grads_fn(images, targets, training=True)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                train_loss_metric.update_state(loss)

            # Reset metrics at end of epoch
            train_loss = train_loss_metric.result().numpy()
            train_loss_metric.reset_states()

            # Evaluation loop
            for images, targets in eval_dataset:
                loss = self.loss_fn(images, targets, training=False)
                eval_loss_metric.update_state(loss)
            eval_loss = eval_loss_metric.result().numpy()
            eval_loss_metric.reset_states()

            if (epoch == 0) or ((epoch + 1) % verbose == 0):
                print(f"Epoch {epoch + 1}: Train Loss={train_loss:.6f}, Eval Loss={eval_loss:.6f}")

            # Early stopping logic
            if eval_loss < best_loss:
                best_loss = eval_loss
                count = early_stopping_rounds
                # Save checkpoint if requested
                if manager:
                    save_path = manager.save()
                    print(f"Checkpoint saved at {save_path}")
            else:
                count -= 1
                if count == 0:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def save_model(self, checkpoint_dir):
        # Save weights & optimizer state using tf.train.Checkpoint
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        save_path = manager.save()
        print(f"Model checkpoint saved to {save_path}")

    def restore_model(self, checkpoint_dir):
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            print(f"Model restored from {manager.latest_checkpoint}")
        else:
            print("No checkpoint found to restore.")

def my_model_function():
    # Instantiate the model with 7 classes (as per original)
    return MyModel(num_classes=7)

def GetInput():
    # Return a random tensor input matching (batch=1, height=48, width=48, channels=1)
    return tf.random.uniform((1, 48, 48, 1), dtype=tf.float32)

