import tensorflow as tf
from tensorflow import keras

class EnhancedModel(Model):
    def __init__(self,  embedding_dim, hidden_dim, vocab_size, label_size,seq_len, pretrained_weight):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.label_size = label_size
        self.activation = tf.keras.activations.tanh
        self.num_layers = 1

        self.embedding=EnhancedEmbedding(vocab_size,embedding_dim,embeddings_initializer=keras.initializers.Constant(pretrained_weight))
        self.encoder = Bidirectional(LSTM(hidden_dim, return_sequences=True,input_shape=(seq_len,embedding_dim)))
        self.pool=MaxPool1D(hidden_dim*2)
        self.decoder = Dense(self.label_size)

    def call(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings=tf.reshape(embeddings,[-1,400,32])
        lstm_out = self.encoder(embeddings)
        lstm_out = tf.transpose(lstm_out, perm=[0,2,1])
        pool_out=self.pool(lstm_out)
        out = tf.squeeze(pool_out,[1])
        out = self.decoder(out)
        return out



#both train_x and train_y are list
train_x_enc=tf.ragged.constant(train_x,dtype=tf.int32)
train_y=train_y=tf.convert_to_tensor(train_y,dtype=tf.int32)

with strategy.scope():
    tf.config.run_functions_eagerly(False)
    model = EnhancedModel(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS, LABELS,SEQ_LEN, pretrain_vectors)
    model.compile(optimizer=tf.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    model.fit(train_x_enc,train_y)

tf.config.run_functions_eagerly(True)

train_x_enc=tf.ragged.constant(train_x,dtype=tf.int32)
train_y=tf.convert_to_tensor(train_y,dtype=tf.int32)

train_x=train_x[:len(train_x)//2]
train_y=train_y[:len(train_y)//2]
train_x_enc=tf.ragged.constant(train_x,dtype=tf.int32)
train_y=tf.convert_to_tensor(train_y,dtype=tf.int32)