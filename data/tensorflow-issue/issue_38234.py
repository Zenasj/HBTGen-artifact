import tensorflow as tf
from tensorflow import keras

class TestNN(tf.keras.Model):
  def __init__(self):
    super(TestNN, self).__init__()
    self.seq = Sequential()
    self.b_lstm1 = Bidirectional(LSTM(128, return_sequences=True, implementation=2), input_shape=(None, 13))
    self.b_lstm2 = Bidirectional(LSTM(128, return_sequences=True, implementation=2))
    self.tmd = TimeDistributed(Dense(len(inv_mapping) + 2))
  
  def call(self, x):
    x = self.seq(x)
    x = self.b_lstm1(x)
    x = self.b_lstm2(x)
    x = self.tmd(x)
    return x

strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
  model = TestNN()
  model.compile(optimizer=tf.optimizers.Adam(1e-2), loss=CTCLoss())

model.fit(input_tensor, label_tensor, batch_size=36*8, epochs=1)

ds = tf.data.Dataset.from_tensor_slices((input_tensor, label_tensor))
test_dataset = ds.batch(256*8, drop_remainder=True)

history = model.fit(test_dataset, batch_size=128 * strategy.num_replicas_in_sync, epochs=128)

with strategy.scope():
    train_dataset = loader.get_train_dataset()
    validation_dataset = loader.get_validation_dataset()

    model.fit(
        train_dataset,
        batch_size=32,
        validation_data=validation_dataset,
        epochs=100,
        verbose=1
    )