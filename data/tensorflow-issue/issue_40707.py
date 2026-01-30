import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def dummy_model_2(params):
    METRICS = [
            # keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.RootMeanSquaredError(name='RMSE')
    ]

    A = tf.keras.layers.DenseFeatures(feature_columns=f)({'drugsName' : tf.keras.Input(name='drugsName', shape=(1,), dtype=tf.string)})
    B = tf.keras.Input((), dtype = tf.string, name = 'condition')
    C = tf.keras.Input((), dtype = tf.string, name = 'reviews')

    model = tf.keras.Model([{'drugsName' : tf.keras.Input(name='drugsName', shape=(1,), dtype=tf.string)}, B, C], [B, C])

    #Set optimizer
    opt = tf.keras.optimizers.Adam(lr= params['lr'], beta_1=params['beta_1'], 
                                        beta_2=params['beta_2'], epsilon=params['epsilon'])

    #Compile model
    model.compile(loss='mean_squared_error',  optimizer=opt, metrics = METRICS)

    #Print Summary
    print(model.summary())
    return model