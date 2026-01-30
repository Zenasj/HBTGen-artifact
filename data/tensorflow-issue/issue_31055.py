import math
import tensorflow as tf
from tensorflow import keras

mobilenet_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

input = l.Input(shape=(224,224,3), name='input_image')

mobilenet = hub.KerasLayer(mobilenet_url)(input)
logits = l.Dense(units=1, activation=tf.nn.sigmoid, name='prediction')(mobilenet)

model = tf.keras.Model(inputs=input, outputs=logits)

tf.saved_model.save(model, 'export/mobilenet_finetuned')

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
def serving(input_image):
    def _input_to_feature(img):
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
    img = tf.map_fn(_input_to_feature, input_image)
    img = tf.image.resize_with_pad(img, 224, 224)
    return model.predict({ 'input_image': img })

tf.saved_model.save(model, export_dir='export/transformed_for_serving', signatures=serving)

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
def serving(input_image):

    # Convert bytes of jpeg input to float32 tensor for model
    def _input_to_feature(img):
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_pad(img, 224, 224)
        return img
    img = tf.map_fn(_input_to_feature, input_image, dtype=tf.float32)

    # Predict
    predictions = model(img)

    with open('export/idx2class.pkl', 'rb') as f:
        class_names = pickle.load(f)
        class_names = tf.constant(class_names, dtype=tf.string)

    # Single output for model so collapse final axis for vector output
    predictions = tf.squeeze(predictions, axis=-1)

    # Predictions are output from sigmoid so float32 in range 0 -> 1
    # Round to integers for predicted class and string lookup for class name
    prediction_integers = tf.cast(tf.math.round(predictions), tf.int32)
    predicted_classes = tf.map_fn(lambda idx: class_names[idx], prediction_integers, dtype=tf.string)

    # Convert sigmoid output for probability
    # 1 (dog) will remain at logit output
    # 0 (cat) will be 1.0 - logit to give probability
    def to_probability(logit):
        if logit < 0.5:
            return 1.0 - logit
        else:
            return logit
    class_probability = tf.map_fn(to_probability, predictions, dtype=tf.float32)

    return {
        'classes': predicted_classes,
        'probabilities': class_probability
    }

tf.saved_model.save(model, export_dir='export/transformed_for_serving', signatures=serving)