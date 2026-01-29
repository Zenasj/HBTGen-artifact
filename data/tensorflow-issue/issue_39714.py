# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input is batch of 28x28 grayscale images

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Flatten layer for 28x28 inputs
        self.flat = tf.keras.layers.Flatten(input_shape=(28, 28))

    def call(self, inputs, training=False, **kwargs):
        x = self.flat(inputs)
        # The original model returns a tuple of three identical flattened tensors
        # Here we replicate this behavior as per the original issue
        out = (x, x, x)
        return out

    def train_step(self, data):
        # Custom train step to handle tuple output and multiple losses
        # We use the same pattern shown in the issue
        # Unpack data (features, labels, sample_weights)
        data = tf.keras.utils.tf_utils.expand_1d(data)
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # forward pass
            loss0 = tf.reduce_sum(self.losses)  # any additional losses (e.g., regularization)
            # Custom loss function returns a tuple of three loss values
            loss1, loss2, loss3 = self.loss(y, y_pred)
            total_loss = tf.reduce_sum([loss0, loss1, loss2, loss3])

        # compute gradients and apply
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # update metrics according to the compiled metrics and outputs
        self.compiled_metrics.update_state(y, y_pred)
        # return dict of metric results
        return {m.name: m.result() for m in self.metrics}


class MultiTaskLoss(tf.keras.losses.Loss):
    def __init__(self):
        # We set reduction NONE because the call returns multiple losses as a tuple
        super(MultiTaskLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # y_pred is a tuple of 3 tensors (all are flattened inputs here)
        # Original issue prints shapes to debug; replicate behavior
        tf.print("Shapes in loss call:", y_pred[0].shape, y_pred[1].shape, y_pred[2].shape)
        # Losses are simply sum of each tensor contents,
        # cast to float32 explicitly (likely unnecessary but done for clarity here)
        loss1 = tf.reduce_sum(y_pred[0])
        loss2 = tf.reduce_sum(y_pred[1])
        loss3 = tf.reduce_sum(y_pred[2])
        return tf.cast(loss1, tf.float32), tf.cast(loss2, tf.float32), tf.cast(loss3, tf.float32)


def my_model_function():
    # Instantiate model and assign the custom loss as model.loss to be used inside train_step
    model = MyModel()
    model.loss = MultiTaskLoss()
    return model


def GetInput():
    # Return a random batch of 8 images with shape (8, 28, 28)
    # Matches the input expected by MyModel (flatten layer with input 28x28)
    batch_size = 8
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

