import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

def to_sequence(data, timesteps=1):
    n_features=data.shape[2]
    seq = []
    for i in range(len(data)-timesteps):
        # takes a window of data of specified timesteps
        temp = data[i:(i+timesteps)]
        temp = temp.reshape(timesteps, n_features)
        seq.append(temp)
        
    return np.array(seq)


def LSTM_autoencoder(data):
    
    
    n_timesteps = data.shape[1]
    n_features = data.shape[2]
    
    keras.backend.clear_session()
    
    
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_timesteps, n_features)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=15, padding='same', 
                            data_format='channels_last',dilation_rate=1, activation="linear"))
    model.add(keras.layers.LSTM(units=50, activation='relu', name='LSTM_1', return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    # to connect encoder with decoder RepeatVector repeats the provided 2D input multiple times to create 3D output
    model.add(keras.layers.RepeatVector(n=n_timesteps))
    # decoder expects the 3D input
    model.add(keras.layers.LSTM(units=50, activation='relu', name='LSTM_2', return_sequences=True))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=15, padding='same', 
                            data_format='channels_last',dilation_rate=1, activation="linear"))
    model.add(keras.layers.Dropout(0.2))
    # allows the same output layer to be reused for each element in sequence
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=n_features)))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    
    return model

model = keras.models.load_model('Lstm3')

test=np.ones(255).reshape(51,5) # (51,5)
test_expanded = np.expand_dims(test,axis=1) # (51,1,5)
test_seq = to_sequence(test_expanded , 50) # (1,50,5)

t=0
while t<100:
  t_start = time.time()
  model.predict(test_seq)
  print(time.time()-t_start)
  t+=1