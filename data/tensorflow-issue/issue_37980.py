import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from kerastuner.tuners import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

tuner = BayesianOptimization(build_model,
                             objective='val_loss',
                             max_trials=1000,
                             executions_per_trial=3,
                             directory=LOG_DIR)

tuner.search(train_data_single,
             verbose=0,
             epochs=EPOCHS,
             steps_per_epoch=EVALUATION_INTERVAL,
             validation_data=val_data_single,
             validation_steps=EVALUATION_INTERVAL // 4)

def build_model(hp):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(units=hp.Int(f'LSTM_0_Units', min_value=8, max_value=128, step=8),
                                   dropout=hp.Float(f'LSTM_0_Dropout_Rate', min_value=0, max_value=0.5, step=0.1),
                                   batch_input_shape=(BATCH_SIZE, x_train_single.shape[1], x_train_single.shape[2]),
                                   return_sequences = True))

    for i in range(hp.Int('n_Extra_Layers', 0, 3)):
        model.add(tf.keras.layers.LSTM(units=hp.Int(f'LSTM_{i + 1}_Units', min_value=8, max_value=128, step=8),
                                       dropout=hp.Float(f'LSTM_{i + 1}_Dropout_Rate', min_value=0, max_value=0.5, step=0.1),
                                       return_sequences = True))
    
    model.add(tf.keras.layers.LSTM(units=hp.Int(f'LSTM_Closing_Units', min_value=8, max_value=128, step=8),
                                   dropout=hp.Float(f'LSTM_Closing_Dropout_Rate', min_value=0, max_value=0.5, step=0.1),
                                   return_sequences = False))
    
    if hp.Boolean("Extra_Dense"):
        model.add(tf.keras.layers.Dense(units=hp.Int(f'Extra_Dense_Units', min_value=8, max_value=128, step=8)))
    
    if hp.Boolean("Extra_Dropout"):
        model.add(tf.keras.layers.Dense(units=hp.Float(f'Extra_Dropout_Rate', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='mae')
    
    return model