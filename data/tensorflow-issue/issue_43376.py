# tf.random.uniform((32, 48, 48, 3), dtype=tf.float32) ← inferred input shape and dtype from ImageDataGenerator flow_from_directory with batch_size=32, target_size=(48,48), color_mode="rgb"

import tensorflow as tf

class CancerNet(tf.keras.Model):
    """
    A simplified placeholder for the CancerNet architecture.
    Since original CancerNet is from external source (pyimagesearch.cancernet),
    we implement a minimal ConvNet suitable for 48x48 RGB images and 2-class output.
    This is a reasonable inference based on input image size and classification task.
    """
    def __init__(self, classes=2):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.conv3 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(classes, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.output_layer(x)

class MyModel(tf.keras.Model):
    """
    Fused model combining CancerNet base and an Embedding layer appended at the end.
    This mimics the original approach where model was built from CancerNet,
    then an Embedding layer with batch_size=32 was added.

    We adapt the Embedding to operate on a placeholder input derived from
    CancerNet output shape. Since the original snippet uses inconsistent Embedding 
    parameters (input_shape=(classWeight,), which is a dict, and input_dim=1024*1000),
    we reason that the intention was to add an embedding to the output feature space,
    possibly to create a learned transformation of the features.

    Here, we implement this by:
    - Using CancerNet to produce logits/features of shape (batch_size, 2)
    - Projecting this to an integer sequence so we can embed it (dummy approach)
    - Using an Embedding layer with input_dim=1000, output_dim=256 (reasonable defaults)
    - Outputting the embedded feature averaged along sequence dimension to get vector output

    This is a best-effort reconstruction given incomplete and inconsistent code.
    """
    def __init__(self):
        super().__init__()
        self.cancernet = CancerNet(classes=2)        
        self.embedding_input_dim = 1000  # Arbitrary large enough vocabulary size
        self.embedding_output_dim = 256
        # Since CancerNet output is shape (batch, 2), map predictions to int indices (e.g., 0 or 1)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embedding_input_dim,
            output_dim=self.embedding_output_dim,
            input_length=1
        )
        # A dense layer to interpret embedded features for final prediction
        self.dense_final = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        # CancerNet forward pass
        features = self.cancernet(inputs, training=training)  # shape (batch, 2)
        # Convert features (probabilities) to "indices" to lookup in embedding
        # Since embedding indices must be integers >=0 and < input_dim,
        # We'll scale and clip features to integers between 0 and embedding_input_dim-1
        indices = tf.clip_by_value(tf.cast(features[:, 0] * self.embedding_input_dim, tf.int32), 0, self.embedding_input_dim-1)
        indices = tf.expand_dims(indices, axis=1)  # shape (batch, 1)
        embedded = self.embedding(indices)  # shape (batch, 1, 256)
        embedded = tf.squeeze(embedded, axis=1)  # shape (batch, 256)
        # Combine embedded features and original features (concatenate)
        combined = tf.concat([features, embedded], axis=1)  # shape (batch, 2 + 256)
        # Final classification output from combined features
        output = self.dense_final(combined)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected input shape of MyModel:
    # batch_size=32 (the original BS in code), height=48, width=48, channels=3 (RGB)
    # dtype=tf.float32, values scaled as typical for images in [0,1] from rescale=1/255
    return tf.random.uniform(shape=(32, 48, 48, 3), minval=0, maxval=1, dtype=tf.float32)

