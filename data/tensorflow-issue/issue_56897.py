import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PredictedDestination(tf.Module):
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_shape=(11, ), name='input'),
             tf.keras.layers.Dense(16, activation=tf.nn.relu, name='dense_1'),
             tf.keras.layers.Dense(8, activation=tf.nn.relu, name='dense_2'),
             tf.keras.layers.Dense(4, activation=tf.nn.relu, name='dense_3'),
            tf.keras.layers.Dense(2),
        ])
    
        self.model.compile(
            optimizer='sgd',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    # The `train` function takes a batch of input images and labels.
    @tf.function(input_signature=[
      tf.TensorSpec([None, 11], tf.float32),
      tf.TensorSpec([None, 2], tf.float32),
    ])
    def train(self, x, y):
        epochs = 100
        for i in range(epochs):
            with tf.GradientTape() as tape:
              prediction = self.model(x)
              loss = self.model.loss(y, prediction)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            result = {"loss": loss}
        return result

    @tf.function(input_signature=[
      tf.TensorSpec([None, 11], tf.float32),
    ])
    def predict(self, x): 
        logits = self.model(x)
        probabilities = tf.nn.softmax(logits, axis=-1)
        print(probabilities)
        print(logits)
        return {
            "output": probabilities,
            "logits": logits
        }

model = PredictedDestination()

SAVED_MODEL_DIR = "predicted_destination_model"

tf.saved_model.save(
    model,
    SAVED_MODEL_DIR,
    signatures={
        'train':
            model.train.get_concrete_function(),
        'predict':
            model.predict.get_concrete_function(),
    })

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

open('predicted_destination.tflite', 'wb').write(tflite_model)