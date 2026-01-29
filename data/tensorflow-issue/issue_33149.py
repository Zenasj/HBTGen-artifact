# tf.random.uniform((B, feats_len), dtype=tf.float32), segments: tf.random.uniform((B,), maxval=B, dtype=tf.int32)
import tensorflow as tf

class SegmentedMean(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SegmentedMean, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features, segments = inputs
        return tf.math.segment_mean(features, segments)

class MyModel(tf.keras.Model):
    def __init__(self, k=40, feats_len=None):
        super(MyModel, self).__init__()
        # Assuming feats_len (feature vector size) is known at init
        # Defensive: if feats_len not provided, raise error
        if feats_len is None:
            raise ValueError("feats_len (input feature dimension) must be provided")
        self.k = k
        self.feats_len = feats_len
        
        self.dense1 = tf.keras.layers.Dense(k, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(k)
        self.segmented_mean = SegmentedMean()
        self.dense3 = tf.keras.layers.Dense(k, activation=tf.nn.relu)
        self.output_logits = tf.keras.layers.Dense(2, name='output_logits')
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=None):
        # inputs is tuple: (features, segments)
        features, segments = inputs
        # features shape: (batch_flat_size, feats_len)
        # segments shape: (batch_flat_size,) int32
        x = self.dense1(features)
        x = self.dense2(x)
        x = self.segmented_mean((x, segments))
        x = self.dense3(x)
        logits = self.output_logits(x)
        probs = self.softmax(logits)
        return logits, probs


def my_model_function():
    # Here we need to infer feats_len.
    # From the issue, feats_len = len(nums_used)
    # nums_used was computed from dummy data originally as a set of unique integers,
    # but we can assume a generic feats_len, e.g. 20 to match example usage.
    # To keep consistent with the example, let's choose feats_len=20.
    # This means input features will be one-hot or multi-hot vectors of length 20.
    feats_len = 20
    return MyModel(k=40, feats_len=feats_len)

def GetInput():
    # Generate inputs matching the expected model signature:
    # - features: (batch_flat_size, feats_len) float32 tensor
    # - segments: (batch_flat_size,) int32 tensor
    
    # In the example from the issue:
    # batch_size = 8 (target for final grouped batch)
    # The actual "flat" batch size is variable—it's the sum of segment lengths in batch
    
    # We'll simulate a small batch with 3 groups having segments of lengths 3, 4, 2 (sum 9)
    # features shape=(9, feats_len), segments shape=(9,)
    feats_len = 20
    segment_lengths = [3, 4, 2]
    batch_flat_size = sum(segment_lengths)
    
    # Create random float features
    features = tf.random.uniform((batch_flat_size, feats_len), dtype=tf.float32)
    
    # Construct segments tensor that assigns each row to a group
    # For segments [3,4,2], this is: [0,0,0,1,1,1,1,2,2]
    segments = []
    for i, length in enumerate(segment_lengths):
        segments.extend([i] * length)
    segments = tf.convert_to_tensor(segments, dtype=tf.int32)
    
    return (features, segments)

