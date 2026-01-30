import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# download fines
from google.colab import files

files.download( "./training_checkpoints/checkpoint" ) 
files.download( "./training_checkpoints/ckpt-5.index" ) 
files.download( "./training_checkpoints/ckpt-5.data-00000-of-00001" )

checkpoint_dir = '<my local directory>'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
    super(Model, self).__init__()
    self.rnn_units = rnn_units
    self.batch_size = batch_size

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if tf.test.is_gpu_available():
      self.gru = tf.keras.layers.CuDNNGRU(self.rnn_units, 
                                          return_sequences=True, 
                                          return_state=True, 
                                          recurrent_initializer='glorot_uniform')
    else:
      self.gru = tf.keras.layers.GRU(self.rnn_units, 
                                     return_sequences=True, 
                                     return_state=True, 
                                     recurrent_activation='sigmoid', 
                                     recurrent_initializer='glorot_uniform')

    self.fc = tf.keras.layers.Dense(vocab_size)
        
  def call(self, x, hidden):
    x = self.embedding(x)

    # output shape == (batch_size, max_length, hidden_size) 
    # states shape == (batch_size, hidden_size)

    # states variable to preserve the state of the model
    # this will be used to pass at every step to the model while training
    output, states = self.gru(x, initial_state=hidden)

    # The dense layer will output predictions for every time_steps(max_length)
    # output shape after the dense layer == (max_length * batch_size, vocab_size)
    x = self.fc(output)

    return x, states
  
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.rnn_units))

# Training step
EPOCHS = 5

for epoch in range(EPOCHS):
    start = time.time()
    
    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    # hidden = model.reset_states()
    hidden_f = model.initialize_hidden_state()
    
    
    for (batch_n, (inp, target)) in enumerate(dataset):
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions, _ = model(inp, hidden_f)
              loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)
              
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))

          if batch_n % 100 == 0:
              template = 'Epoch {} Batch {} Loss {:.4f}'
              print(template.format(epoch+1, batch_n, loss))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
      model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))