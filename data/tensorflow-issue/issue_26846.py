import tensorflow as tf

url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url, trainable=True)

def make_elmo_embedding(x):
    embeddings = embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]
    return embeddings

# elmo embedding dimension
elmo_dim = 1024

# Input Layers
elmo_input = Input(shape=(None, ), dtype="string")

# Hidden Layers
elmo_embedding = Lambda(make_elmo_embedding, output_shape=(None, elmo_dim))(elmo_input)


x = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))(elmo_embedding)
x = Dense(32, activation='relu')(x)
predict = Dense(2, activation='sigmoid')(x)
model = Model(inputs=[elmo_input], outputs=predict)
model.compile(loss='mse', optimizer='sgd')

make_elmo_embedding

tf.squeeze

tf.squeeze(z, axis=1)

def make_elmo_embedding(x):
    embeddings = embed(tf.squeeze(tf.cast(x, tf.string), axis=1), signature="default", as_dict=True)["elmo"]
    return embeddings