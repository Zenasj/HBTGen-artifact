import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tflite_input_tensor = tf.constant(1., shape=[64, 39])
tflite_target_tensor = tf.constant(1., shape=[64, 7])
tflite_enc_hidden_tensor = tf.constant(1., shape=[64, 1024])
export_dir = "saved_models"
checkpoint.f = train_step
to_save = checkpoint.f.get_concrete_function(tflite_input_tensor, tflite_target_tensor, tflite_enc_hidden_tensor)
tf.saved_model.save(checkpoint, export_dir, to_save)

converter = tf.lite.TFLiteConverter.from_concrete_functions([to_save])
tflite_model = converter.convert()

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)
            
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

EPOCHS = 3

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 1 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       unroll=True)
        
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.rnn(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

tflite_input_shape = tf.TensorSpec([64, 39], tf.int32)
tflite_target_shape = tf.TensorSpec([64, 7], tf.float32)
tflite_enc_hidden_shape = tf.TensorSpec([64, 1024], tf.float32)
export_dir = "saved_models"
checkpoint.f = train_step
to_save = checkpoint.f.get_concrete_function(tflite_input_shape, tflite_target_shape, tflite_enc_hidden_shape)
tf.saved_model.save(checkpoint, export_dir, to_save)

@tf.function
def eval_step(enc_input):
    results = []
    
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(enc_input, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)



    for t in tf.range(max_length_targ):
        predictions, dec_hidden, _ = decoder([dec_input,dec_hidden,enc_out])

        predicted_id = tf.argmax(predictions[0], output_type=tf.int32)

        results.append(predicted_id)

        if tf.equal(predicted_id,tf.constant(2)): # <end> index
            break

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return tf.convert_to_tensor(results.values(),dtype=np.int32)

@tf.function
def eval_step(enc_input):
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(enc_input, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    predictions, dec_hidden, _ = decoder([dec_input,dec_hidden,enc_out])
    predicted_id_1 = tf.argmax(predictions[0], output_type=tf.int32)
    dec_input = tf.expand_dims([predicted_id_1], 0)
    
    predictions, dec_hidden, _ = decoder([dec_input,dec_hidden,enc_out])
    predicted_id_2 = tf.argmax(predictions[0], output_type=tf.int32)
    dec_input = tf.expand_dims([predicted_id_2], 0)
    
    predictions, dec_hidden, _ = decoder([dec_input,dec_hidden,enc_out])
    predicted_id_3 = tf.argmax(predictions[0], output_type=tf.int32)
    dec_input = tf.expand_dims([predicted_id_3], 0)
    
    predictions, dec_hidden, _ = decoder([dec_input,dec_hidden,enc_out])
    predicted_id_4 = tf.argmax(predictions[0], output_type=tf.int32)
    dec_input = tf.expand_dims([predicted_id_4], 0)
    
    return tf.convert_to_tensor([predicted_id_1,predicted_id_2,predicted_id_3,predicted_id_4])

@tf.function
def eval_step(enc_input, dec_input):
    enc_output, enc_hidden = encoder(enc_input)
    dec_hidden = enc_hidden

    predictions, dec_hidden, _ = decoder([dec_input, dec_hidden, enc_output])

    return predictions

tflite_enc_input_shape = tf.TensorSpec([None,list(dataset.take(1))[0][0].shape[1]], tf.int32)
# tflite_dec_input_shape = tf.TensorSpec([None, 1], tf.int32)
checkpoint.f = eval_step
to_save = checkpoint.f.get_concrete_function(tflite_enc_input_shape)

tf.random.set_seed(1234)

converter = tf.lite.TFLiteConverter.from_concrete_functions([to_save])
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

enc_input_shape = input_details[0]['shape']
enc_input_data = np.array(np.random.randint(39,size=enc_input_shape), dtype=np.int32)

interpreter.set_tensor(input_details[0]['index'], enc_input_data)
# interpreter.set_tensor(input_details[1]['index'], dec_input_data)

interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

@tf.function
def eval_step_enc(enc_input):
    enc_out, enc_hidden = encoder(enc_input, [tf.zeros((1, units))])
    
    return enc_out, enc_hidden

@tf.function
def eval_step_dec(dec_input, enc_out, dec_hidden):
    predictions, dec_hidden, _ = decoder([dec_input,dec_hidden,enc_out])
    scores = tf.exp(predictions) / tf.reduce_sum(tf.exp(predictions), axis=1)
    dec_input = tf.expand_dims(tf.argmax(predictions, axis=1, output_type=tf.int32), 1)
    
    return dec_input, enc_out, dec_hidden, scores

# ...standard TFLite conversion code