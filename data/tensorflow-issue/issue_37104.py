# tf.random.uniform((B, feature_dim)) ← Input is 2D tensor batch of feature vectors with shape (batch_size, feature_dim)
import tensorflow as tf

# Since the issue is about metrics mismatch between softmax outputs (shape (n, 2)) with sparse labels (shape (n, 1))
# causing metrics like Precision to fail, this model integrates two submodels:
# 1) A softmax-output model commonly used for sparse_categorical_crossentropy loss with 2 outputs (multiclass)
# 2) A sigmoid-output model for binary classification with 1 output unit.
#
# The call method compares their predictions after thresholding / argmax to show the shape and logic differences,
# demonstrating the incompatibility issue.
#
# For demonstration, the comparison outputs a boolean tensor indicating where the two models produce the same predicted class.

class MyModel(tf.keras.Model):
    def __init__(self, feature_dim=10):
        super().__init__()
        # Feature layer placeholder, assuming dense input
        # This corresponds to the feature_layer from the original issue (unknown internals)
        self.feature_layer = tf.keras.layers.InputLayer(input_shape=(feature_dim,), name="feature_layer")

        # Model A: softmax output with 2 units - typical setup for sparse_categorical_crossentropy
        self.model_softmax = tf.keras.Sequential([
            self.feature_layer,
            tf.keras.layers.Dense(12, activation='relu', use_bias=True,
                                  kernel_initializer='glorot_normal', bias_initializer='zeros', name='d1'),
            tf.keras.layers.Dense(6, activation='relu', use_bias=True,
                                  kernel_initializer='glorot_normal', bias_initializer='zeros', name='d2'),
            tf.keras.layers.Dense(2, activation='softmax', name='out_softmax')
        ])

        # Model B: sigmoid output with 1 unit - typical setup for binary_crossentropy and binary classification
        self.model_sigmoid = tf.keras.Sequential([
            self.feature_layer,
            tf.keras.layers.Dense(12, activation='relu', use_bias=True,
                                  kernel_initializer='glorot_normal', bias_initializer='zeros', name='d1_sigmoid'),
            tf.keras.layers.Dense(6, activation='relu', use_bias=True,
                                  kernel_initializer='glorot_normal', bias_initializer='zeros', name='d2_sigmoid'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='out_sigmoid')
        ])

    def call(self, inputs, training=False):
        # Get softmax prediction (shape: (batch, 2))
        pred_softmax = self.model_softmax(inputs, training=training)
        # Get sigmoid prediction (shape: (batch, 1))
        pred_sigmoid = self.model_sigmoid(inputs, training=training)

        # Convert softmax predictions to predicted class indices (0 or 1)
        pred_class_softmax = tf.argmax(pred_softmax, axis=1, output_type=tf.int32)  # shape (batch,)
        # Convert sigmoid prediction > 0.5 to predicted class indices (0 or 1)
        pred_class_sigmoid = tf.cast(pred_sigmoid > 0.5, tf.int32)  # shape (batch, 1)
        pred_class_sigmoid = tf.squeeze(pred_class_sigmoid, axis=1)  # shape (batch,)

        # Comparison boolean tensor indicating if both models predict the same class
        comparison = tf.equal(pred_class_softmax, pred_class_sigmoid)  # shape (batch,)

        # We could output all: softmax probs, sigmoid probs, and comparison for demonstration
        return {
            "softmax_probs": pred_softmax,
            "sigmoid_probs": pred_sigmoid,
            "pred_class_softmax": pred_class_softmax,
            "pred_class_sigmoid": pred_class_sigmoid,
            "comparison": comparison,
        }

def my_model_function():
    # We default to feature dimension 10 (arbitrary choice since feature_layer unknown)
    model = MyModel(feature_dim=10)
    # Compile one or both submodels as needed - shown here with optimizer and losses for demonstration
    # Note: Compilation of MyModel itself is unclear since it returns dict output, so left uncompiled here.
    return model

def GetInput():
    # Returns a random float tensor simulating feature vectors with shape (batch_size=4, feature_dim=10)
    # matching the input expected by the model's feature_layer
    batch_size = 4
    feature_dim = 10
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

