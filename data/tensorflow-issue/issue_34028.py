import tensorflow as tf

class SiameseRNN(tfk.Model):
    def __init__(self, embeddings):
        super(SiameseRNN, self).__init__()
        self.embedding_layer = tfk.layers.Embedding(embeddings.shape[0], embeddings.shape[1], trainable=False,
                                                    embeddings_initializer=tf.initializers.Constant(embeddings))
        self.gru_layer = tfk.layers.GRU(FLAGS.gru_units, dropout=FLAGS.rnn_dropout,
                                        recurrent_dropout=FLAGS.rnn_recurrent_dropout)
        self.batchnorm_layer1 = tfk.layers.BatchNormalization()
        self.dropout_layer = tfk.layers.Dropout(FLAGS.dropout_rate)
        self.dense_layer1 = tfk.layers.Dense(FLAGS.dense_units, activation=FLAGS.dense1_activation)
        self.batchnorm_layer2 = tfk.layers.BatchNormalization()
        self.dense_layer2 = tfk.layers.Dense(1, FLAGS.dense2_activation)

    def call(self, inputs, training=False):
        x1 = inputs[0]
        x2 = inputs[1]
        x1_mask = inputs[2]
        x2_mask = inputs[3]
        x1_embed = self.embedding_layer(x1)
        x2_embed = self.embedding_layer(x2)
        x1_encoding = self.gru_layer(x1_embed, mask=x1_mask, training=training)
        x2_encoding = self.gru_layer(x2_embed, mask=x2_mask, training=training)
        x1_x2_encoding_concat = tf.concat([x1_encoding, x2_encoding], axis=-1)
        x1_x2_encoding_concat = self.batchnorm_layer1(x1_x2_encoding_concat, training=training)
        x1_x2_encoding_concat = self.dropout_layer(x1_x2_encoding_concat, training=training)
        hidden = self.dense_layer1(x1_x2_encoding_concat)
        hidden = self.batchnorm_layer2(hidden, training=training)
        y_pred = self.dense_layer2(hidden)
        return y_pred