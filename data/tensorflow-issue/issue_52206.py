# tf.Tensor with shape (batch_size, 1), dtype=tf.string

import tensorflow as tf
import tensorflow_hub as hub

@tf.function
def preprocessing_space_tf(text):
    # Text preprocessing function to lowercase and remove some special chars and digits (optionally)
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


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=3266, **kwargs):
        super(MyModel, self).__init__(name="USE_CNN", **kwargs)
        self.num_classes = num_classes
        
        # Preprocessing function to clean input text strings
        self.preprocess = preprocessing_space_tf
        
        # Universal Sentence Encoder layer from TF Hub (trainable)
        self.embedding = hub.KerasLayer(
            'https://tfhub.dev/google/universal-sentence-encoder/4',
            trainable=True,
            input_shape=[]
        )
        
        self.dense_1024 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.35)
        self.dense_out = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, text, training=False):
        # text expected shape: (batch_size, 1) tensor of strings
        x = self.preprocess(text)
        # tf.squeeze(x, axis=1) converts shape (batch_size,1) to (batch_size,)
        x = self.embedding(tf.squeeze(x, axis=1))
        x = self.dense_1024(x)
        x = self.dropout(x, training=training)
        x = self.dense_out(x)
        return x


def my_model_function():
    # Return an instance of MyModel with default 3266 classes as from original code
    return MyModel()


def GetInput():
    # Return a random valid input for MyModel for batch size=1
    # The model expects a tf.Tensor of shape (batch_size, 1), dtype string
    sample_text = tf.constant([["Rear Air Conditioning"]])  # shape (1,1) string tensor
    return sample_text

