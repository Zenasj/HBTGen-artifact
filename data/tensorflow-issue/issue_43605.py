# tf.random.uniform((B, 1), dtype=tf.int64) and tf.random.uniform((B, 1), dtype=tf.string) for categorical feature
import tensorflow as tf

# This model replicates the minimal subclassed keras.Model using tf.feature_column.DenseFeatures
# as described in the issue. The input is a dict with keys 'age' (int64) and 'education' (string),
# each with shape (batch_size, 1). The model outputs a sigmoid prediction.

# Because the key difficulty reported was inference failed due to input shape mismatch,
# the model expects inputs shaped as described in the issue: batch dimension (None),
# then 1 column per feature. This is enforced in the GetInput() to generate proper shape.

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Create feature columns same as in the provided code:
        age = tf.feature_column.numeric_column("age", dtype=tf.int64)
        education_cat = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
        education_emb = tf.feature_column.embedding_column(education_cat, dimension=100)

        feat_cols = [age, education_emb]

        # DenseFeatures layer processes the feature columns dict input
        self.dense_features = tf.keras.layers.DenseFeatures(feature_columns=feat_cols)
        self.dense1 = tf.keras.layers.Dense(10, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
        inputs: dict of tensors with keys "age" (int64 tensor shape (batch,1))
                and "education" (string tensor shape (batch,1))
        """
        x = self.dense_features(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    """
    Returns a dict of two tensors shaped (batch_size, 1):
    - 'age' as int64 tensor with shape (batch_size, 1)
    - 'education' as string tensor with shape (batch_size, 1)

    This matches the expected input shape for MyModel.
    Here we create batch size 2 example.
    """
    batch_size = 2
    age = tf.constant([[35],[40]], dtype=tf.int64)           # shape (2,1)
    education = tf.constant([["Bachelors"], ["Assoc-voc"]], dtype=tf.string)  # shape (2,1)

    return {"age": age, "education": education}

