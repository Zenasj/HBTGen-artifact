import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

3
def get_compiled_model():
    
    the_input = keras.layers.Input(shape=(window_length, input_length))
    embeddings = keras.layers.Embedding(num_addresses, embedding_dim)(the_input)

    encoded = keras.layers.Bidirectional(keras.layers.LSTM(NUM_HIDDEN))(embeddings)

    reconstr = keras.layers.RepeatVector(window_length)(encoded)
    reconstr = keras.layers.Bidirectional(keras.layers.LSTM(NUM_HIDDEN, return_sequences=True))(reconstr)
    
    reconstr = keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='relu'), 
                                                    name="the_output")(reconstr)
    
    losses = {
        "the_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }
    lossWeights = LOSS_WEIGHTS
        
    model = keras.Model([
        the_input, 
    ], [
        reconstr
    ])

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE),
                  loss=losses)
    
    return model

gpu_devices = [device for device in tf.config.list_physical_devices() if device.device_type=='GPU']
if len(gpu_devices):
    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3'])
#     strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_compiled_model()
else:
    model = get_compiled_model()

# train_generator is a custom keras.utils.Sequence 
model.fit(x=train_generator, epochs=200)