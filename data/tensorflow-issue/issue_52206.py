import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.function
def preprocessing_space_tf(text):

    def remove_special_characters(text, remove_digits=False):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = tf.strings.regex_replace(text, pattern, '')
        return text

    def replace(text):
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, '_', ' ')
        text = tf.strings.regex_replace(text, '-', ' ')
        text = tf.strings.regex_replace(text, ':', ' ')
        text = tf.strings.regex_replace(text, '/', ' ')
        return text

    text = replace(text)
    text = remove_special_characters(text)

    return text


class USE_CNN(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(USE_CNN, self).__init__(name='USE_CNN', **kwargs)
        self.num_classes = num_classes
        self.preprocess = preprocessing_space_tf
        self.embedding = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', trainable=True)
        self.dense_1024 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.35)
        self.dense_out = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, text):
        x = self.preprocess(text)
        x = self.embedding(tf.squeeze(x, axis=1))
        x = self.dense_1024(x)
        x = self.dropout(x)
        x = self.dense_out(x)
        return x


nlp_model = USE_CNN(num_classes=3266)

nlp_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=tf.keras.metrics.SparseCategoricalAccuracy())

nlp_model.fit(x_train, y_train, epochs=50, batch_size=100)

nlp_model.save('/use_cnn/', save_format='tf')

# convert to tflite
converter = tf.lite.TFLiteConverter.from_saved_model('/use_cnn/')  # path to the SavedModel directory
converter.experimental_enable_resource_variables = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Inference using tflite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, np.expand_dims(['Rear Air Conditioning'], 0))
interpreter.invoke()
output_details = interpreter.get_output_details()