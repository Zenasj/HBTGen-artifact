# tf.random.normal((B, 2048), dtype=tf.float32) ← Input shape inferred from issue examples (B=batch, 2048 features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a shared architecture repeated for both mixed precision and fp32 models
        # The core model: 6 dense layers with relu activations and final logits layer + softmax output
        # Feature size and number of units is 2048 (as per final code chunk)
        num_units = 2048
        num_classes = 16

        # Layers for mixed precision model
        # These will run under mixed precision policy when used after setting policy externally
        self.mp_dense1 = tf.keras.layers.Dense(num_units, activation='relu', name='mp_dense_1')
        self.mp_dense2 = tf.keras.layers.Dense(num_units, activation='relu', name='mp_dense_2')
        self.mp_dense3 = tf.keras.layers.Dense(num_units, activation='relu', name='mp_dense_3')
        self.mp_dense4 = tf.keras.layers.Dense(num_units, activation='relu', name='mp_dense_4')
        self.mp_dense5 = tf.keras.layers.Dense(num_units, activation='relu', name='mp_dense_5')
        self.mp_dense6 = tf.keras.layers.Dense(num_units, activation='relu', name='mp_dense_6')
        self.mp_logits = tf.keras.layers.Dense(num_classes, name='mp_dense_logits')
        self.mp_softmax = tf.keras.layers.Activation('softmax', dtype='float32', name='mp_predictions')

        # Layers for fp32 model (float32 only)
        self.fp_dense1 = tf.keras.layers.Dense(num_units, activation='relu', name='fp_dense_1')
        self.fp_dense2 = tf.keras.layers.Dense(num_units, activation='relu', name='fp_dense_2')
        self.fp_dense3 = tf.keras.layers.Dense(num_units, activation='relu', name='fp_dense_3')
        self.fp_dense4 = tf.keras.layers.Dense(num_units, activation='relu', name='fp_dense_4')
        self.fp_dense5 = tf.keras.layers.Dense(num_units, activation='relu', name='fp_dense_5')
        self.fp_dense6 = tf.keras.layers.Dense(num_units, activation='relu', name='fp_dense_6')
        self.fp_logits = tf.keras.layers.Dense(num_classes, name='fp_dense_logits')
        self.fp_softmax = tf.keras.layers.Activation('softmax', dtype='float32', name='fp_predictions')

    def call(self, inputs, training=False):
        """
        Run inputs through both mixed precision and FP32 models and compare outputs.
        Returns a dict with both outputs and the numeric difference (l2 norm) between predictions.
        """
        # Mixed precision forward path:
        # Because the policy should be set outside this class, layers will run accordingly.
        x_mp = self.mp_dense1(inputs)
        x_mp = self.mp_dense2(x_mp)
        x_mp = self.mp_dense3(x_mp)
        x_mp = self.mp_dense4(x_mp)
        x_mp = self.mp_dense5(x_mp)
        x_mp = self.mp_dense6(x_mp)
        logits_mp = self.mp_logits(x_mp)
        preds_mp = self.mp_softmax(logits_mp)

        # FP32 forward path:
        # Force float32 casting to ensure fp32 path
        x_fp = tf.cast(inputs, tf.float32)
        x_fp = self.fp_dense1(x_fp)
        x_fp = self.fp_dense2(x_fp)
        x_fp = self.fp_dense3(x_fp)
        x_fp = self.fp_dense4(x_fp)
        x_fp = self.fp_dense5(x_fp)
        x_fp = self.fp_dense6(x_fp)
        logits_fp = self.fp_logits(x_fp)
        preds_fp = self.fp_softmax(logits_fp)

        # Compute difference between mixed precision and fp32 predictions for comparison
        diff = tf.norm(preds_mp - preds_fp, ord='euclidean', axis=-1)

        return {
            'mp_predictions': preds_mp,
            'fp_predictions': preds_fp,
            'diff_l2': diff
        }

def my_model_function():
    """
    Setup mixed precision policy and instantiate MyModel.
    Note: mixed_precision policy effects only MP layers - FP layers stay in float32.
    """
    from tensorflow.keras import mixed_precision

    # Set global mixed precision policy to 'mixed_float16'
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    model = MyModel()

    # The model has no external weights to load as this is a demo from scratch
    return model

def GetInput():
    """
    Return a random tensor input matching expected input shape with proper dtype.
    Use float32 input to be compatible with both mixed precision and fp32 paths.
    Batch dimension is arbitrary, here 32 for example.
    """
    batch_size = 32
    feature_size = 2048
    # Use float32 tensor since inputs are float32 in examples, mixed precision layers will handle casting internally
    return tf.random.normal((batch_size, feature_size), dtype=tf.float32)

