import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.function
def evaluate(image):
    hidden = decoder.reset_states(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0]
        # print(tokenizer.index_word)
        print(predicted_id,predicted_id.dtype)

        # for key,value in tokenizer.index_word.items():
        #     key = tf.convert_to_tensor(key)
        #     tf.dtypes.cast(key,tf.int64)
        #     print(key)

        # print(tokenizer.index_word)

        result.append(predicted_id)

        # if tokenizer.index_word[predicted_id] == '<end>':
        #     return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result

export_dir = "./"
tflite_enc_input = ''
ckpt.f = evaluate
to_save = evaluate.get_concrete_function('')

converter = tf.lite.TFLiteConverter.from_concrete_functions([to_save])
tflite_model = converter.convert()

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 64, features_shape),dtype=tf.dtypes.float32)])
    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       unroll = True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)


    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1], dtype=tf.int64),
                                  tf.TensorSpec(shape=[1, 64, 256], dtype=tf.float32),
                                  tf.TensorSpec(shape=[1, 512], dtype=tf.float32)])
    def call(self, x , features, hidden):

        context_vector, attention_weights = self.attention(features, hidden)

        #x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        #x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)


        output, state = self.gru(x)

        #shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        #x shape == (batch_size, max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_states(self, batch_size):
        return tf.zeros((batch_size, self.units))