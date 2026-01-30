import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# Model definition
sequence_input = Input(shape=(MAX_LEN,), name='sequence_input')
model = Embedding(input_dim=n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
                  input_length=MAX_LEN)(sequence_input)  # default: 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM

model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer


crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PAD)


sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)

outputs = crf(model)

model = Model(sequence_input, outputs)

model.compile(
    loss=crf.neg_log_likelihood,
    metrics=[crf.accuracy],
    optimizer=tf.keras.optimizers.Adam(5e-5)
    )

interpreter = tf.lite.Interpreter(model_path="model1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(output_details)
print(input_details)
# Test model on random input data.
input_shape = input_details[0]['shape']
test_sentence = nltk.word_tokenize("A series of explosions shook the Iraqi capital")
# Preprocessing
x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                        padding="post", value=word2idx["PAD"], maxlen=MAX_LEN)
    
input_data = np.array([x_test_sent[0]], dtype=np.float32)
# input_data = input_data.astype(np.float) 
print(input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
np.set_printoptions(threshold=sys.maxsize)

print(output_data)