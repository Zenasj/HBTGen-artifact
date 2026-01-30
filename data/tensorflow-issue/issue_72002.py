import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

for tr_opt in train_options_list:
    with tf.device('/GPU:0'):
        files = os.listdir(dir_data_path + param_dir + '/' + tr_opt)
        num_files = len(files)

        df_agg = pd.DataFrame()
        for i in range (0, num_files):
          current_file = pd.read_excel(dir_data_path + param_dir + '/' + tr_opt + '/' + files[i])
          current_file.columns = ['Время', param_dir]
          df_agg = pd.concat([df_agg, current_file])
    
        data = df_agg[param_dir].values.reshape(-1, 1)

        train_set_scale = scaler.transform(data)

        # Формируем матрицу
        n_steps = 180
        n_features = 1
        train_matrix = create_matrix(train_set_scale, n_steps)

        tf.keras.backend.clear_session()
        np.random.seed(41)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(n_steps, n_features)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True)))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))

        model.compile(optimizer='adam', loss="mse")

        net_history = model.fit(train_matrix, train_matrix, epochs=10, batch_size=150)
        model.save(save_path + 'model.h5')

with tf.device('/GPU:0'):
    model.fit(...)

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)