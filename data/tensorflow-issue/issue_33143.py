# tf.random.uniform((B, feats_len), dtype=tf.float32), tf.random.uniform((B,), maxval=segments_max, dtype=tf.int32)
import tensorflow as tf

# This model reproduces the "multiple input sizes" scenario where:
# - One input is features with shape (batch_size, feats_len)
# - The other input is segments/int32 tensor with shape (batch_size,)
# The output is after segment-wise aggregation, reducing the number of samples to num_segments
# so input_size != output_size, which causes problems in Keras fit_generator in older TF versions.

class SegmentedMean(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SegmentedMean, self).__init__(**kwargs)
    def call(self, inputs):
        features, segments = inputs
        # Aggregate features by segment indices - segment_mean reduces feature rows into segments.
        return tf.math.segment_mean(features, segments)

class MyModel(tf.keras.Model):
    def __init__(self, feats_len=3673, k=128, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Hidden dimension 'k' is arbitrary, from user's "settings['k']"
        self.dense1 = tf.keras.layers.Dense(k, activation='relu')
        self.dense2 = tf.keras.layers.Dense(k)
        self.segmented_mean = SegmentedMean()
        self.dense3 = tf.keras.layers.Dense(k, activation='relu')
        self.logits_layer = tf.keras.layers.Dense(2, name='output_logits')
        self.softmax = tf.keras.layers.Softmax()

        # Save other params if needed
        self.feats_len = feats_len
        self.k = k

    def call(self, inputs, training=None, mask=None):
        # inputs is expected to be a tuple (features, segments)
        features, segments = inputs
        x = self.dense1(features)
        x = self.dense2(x)
        x = self.segmented_mean((x, segments))  # segment_mean reduces the batch from features count to segments count
        x = self.dense3(x)
        logits = self.logits_layer(x)
        probs = self.softmax(logits)
        return logits, probs

def my_model_function():
    # Instantiate the model with example parameters.
    # feats_len could be inferred from your data, here chosen as 3673 per original post
    # k is a hyperparam; chosen as 128 as a reasonable default based on example
    model = MyModel(feats_len=3673, k=128)
    # Compile model with matching loss and optimizer as in the example
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss={'output_logits': loss})
    return model

def GetInput():
    # We need to generate a compatible input tuple matching the model inputs:
    # features: shape (batch_size, feats_len)
    # segments: shape (batch_size,), int32 segments indices for aggregating features
    batch_size = 486  # example number from the issue
    feats_len = 3673
    # segments must have integer segment indices in [0, num_segments-1]
    # The aggregated output size = number of unique segments
    segments_max = 87  # from the example (87 output samples)
    # Create random float features
    features = tf.random.uniform((batch_size, feats_len), dtype=tf.float32)
    # Create random segment indices corresponding to each row in features
    segments = tf.random.uniform((batch_size,), maxval=segments_max, dtype=tf.int32)
    return (features, segments)

