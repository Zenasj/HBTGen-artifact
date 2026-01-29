# tf.random.uniform((batch_size, 1), dtype=tf.float32) ← Input is a batch of scalar floats shaped (B, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(num_classes)
        self.first_step = True  # Control flow in call, per original code

    def call(self, inputs, training=None, mask=None):
        hidden = self.dense1(inputs)
        if training and not self.first_step:
            # According to original code, on training and after first step,
            # it returns hidden for loss computation only, no logits
            return None, hidden
        else:
            logits = self.dense2(hidden)
            return logits, hidden


class SampledSoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, decoder_obj=None, num_classes=0):
        super().__init__()
        self.decoder_obj = decoder_obj  # The final Dense layer to get weights/biases from
        self.num_classes = num_classes
        self.num_sampled = 5  # Number of negative samples

    def call(self, labels, hidden):
        labels = tf.cast(tf.expand_dims(labels, -1), tf.int64)  # Shape (batch_size, 1)
        # Get layer weights and biases from Dense layer (kernel and bias)
        weights = tf.transpose(self.decoder_obj.get_weights()[0])  # Shape (num_classes, units)
        biases = self.decoder_obj.get_weights()[1]  # Shape (num_classes,)

        # Sampled softmax requires sampled values for true and sampled classes
        sampled_values = tf.random.uniform_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=self.num_sampled,
            unique=False,
            range_max=self.num_classes,
        )

        # Compute sampled softmax loss
        loss_val = tf.nn.sampled_softmax_loss(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=hidden,
            num_sampled=self.num_sampled,
            num_classes=self.num_classes,
            sampled_values=sampled_values
        )
        return tf.reduce_mean(loss_val)  # Return mean scalar loss


def my_model_function():
    # Create the MyModel instance with fixed number of classes
    num_classes = 500
    return MyModel(num_classes=num_classes)


def GetInput():
    # Create random input tensor shaped (batch_size, 1) with float32 dtype,
    # matching the original input format (x was expanded dims of integers in float32).
    batch_size = 10  # consistent with original test batch size
    # Per original code, x was np.expand_dims(y.astype(np.float32), -1), so input floats scalar
    return tf.random.uniform((batch_size, 1), minval=0, maxval=500, dtype=tf.float32)

