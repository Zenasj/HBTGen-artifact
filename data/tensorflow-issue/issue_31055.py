# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        mobilenet_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
        self.mobilenet_layer = hub.KerasLayer(mobilenet_url, trainable=False)
        self.classifier = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, name='prediction')

    def call(self, inputs, training=False):
        features = self.mobilenet_layer(inputs)
        logits = self.classifier(features)
        return logits

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tensor simulating a batch of raw JPEG images bytes, here we create random float images as a proxy
    # but since the model expects float images of shape (B,224,224,3), dtype float32,
    # we generate random float tensors in that shape.
    # Assumption: Batch size 4 for example.
    batch_size = 4
    # Inputs are normalized float images of shape (224,224,3)
    return tf.random.uniform((batch_size, 224, 224, 3), minval=0, maxval=1, dtype=tf.float32)


# Additionally, we provide a serving function example that performs the preprocessing pipeline shown in the issue:
# This function is not part of the required output but reflects the documented approach to combine raw JPEG string input 
# preprocessing inside a tf.function for serving.

# The serving input signature expects a vector of strings (raw jpeg bytes).
# The function decodes, converts, resizes, then passes through the model.

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
def serving(input_image):
    # input_image: 1D tf.string tensor of jpeg-encoded images

    def _input_to_feature(img_bytes):
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # scale to [0,1]
        img = tf.image.resize_with_pad(img, 224, 224)
        return img

    img_batch = tf.map_fn(_input_to_feature, input_image, dtype=tf.float32)
    model = my_model_function()
    predictions = model(img_batch)  # (B, 1)

    # Squeeze the last dim to get (B,)
    predictions = tf.squeeze(predictions, axis=-1)

    # Example class_names hardcoded here for illustration (normally loaded from file 'idx2class.pkl')
    class_names = tf.constant(["cat", "dog"], dtype=tf.string)

    # Round sigmoid outputs: <0.5 -> 0, >=0.5 -> 1
    pred_ints = tf.cast(tf.math.round(predictions), tf.int32)

    # Lookup string class names according to predicted indices
    predicted_classes = tf.map_fn(lambda idx: class_names[idx], pred_ints, dtype=tf.string)

    # Probability adjustment: for class 0 (cat), prob = 1 - sigmoid output, for class 1 (dog), prob = sigmoid output
    def to_probability(logit):
        # Using tf.cond for graph compat
        return tf.cond(logit < 0.5, lambda: 1.0 - logit, lambda: logit)
    class_probabilities = tf.map_fn(to_probability, predictions, dtype=tf.float32)

    return {
        'classes': predicted_classes,
        'probabilities': class_probabilities
    }

