import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# Create training/test datasets
train_data = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
test_data = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
forecast_data = tf.data.Dataset.from_tensor_slices((forecast_ahead_df.values))

# batching
train_batch = train_data.batch(len(X_train)) 
test_batch = test_data.batch(len(X_test))
forecast_batch = forecast_data.batch(len(forecast_ahead_df))


def predict_with_confidence_ints(self, 
                                   X: np.array,
                                   dropout_rate: float, 
                                   n_classes: int,
                                   n_iter: int):
    """Load saved .h5 keras model and generate predictions using dropout.
    Args: TODO
    Returns: TODO
    """

    # Load saved Keras model (.h5)
    model = tf.keras.models.load_model(self.saved_keras_model,
                                       custom_objects = self.custom_objects) 

    # Load the config of the original model
    conf = model.get_config()
    
    # Add the specified dropout to all layers
    for layer in conf['layers']:
      # Dropout layers
      if layer["class_name"]=="Dropout":
        layer["config"]["rate"] = dropout_rate
      # Recurrent layers with Dropout
      elif "dropout" in layer["config"].keys():
        layer["config"]["dropout"] = dropout_rate

    # # Create a new model with specified dropout
    if type(model) == tf.keras.Sequential:
      # Sequential
      model_with_dropout = tf.keras.Sequential.from_config(conf)
    else:
      # Functional
      model_with_dropout = tf.keras.models.Model.from_config(conf)
 
    # Get config weights and init new model with same weights
    model_with_dropout.set_weights(model.get_weights())  
    
    # Create function that applies dropout to learning phase
    f = K.function([model_with_dropout.layers[0].input, K.learning_phase()],
               [model_with_dropout.layers[-1].output])

    result = np.zeros((n_iter,) + (x.shape[0], 1))

    for i in range(n_iter):
        result[i, :] = f((x, 1))[0]
    
    predictions_ = result.mean(axis=0)
    uncertainty_ = result.std(axis=0)
    # predictions = pd.Series(predictions_.reshape(-1), name = 'Predictions')
    # std_devs = pd.Series(uncertainty_.reshape(-1), name = 'Uncertainty')
    return predictions_, uncertainty_ 

# Create predictions with dropout

# Get custom objects
custom_objects={'root_mean_squared_error': root_mean_squared_error,
                'glorot_uniform': keras.initializers.glorot_uniform(seed=None)}

# Get saved model
model_path = 'my_saved_keras_model.h5'

# Set dropout rate
dropout = 0.3

# Convert input data to np array
X = np.array(forecast_ahead_df)

# Get predictions 
predictions_new, uncertainties_new = predict_with_confidence_ints(X, dropout, 1, 100)

# Define train/test
X_train = final_df[(final_df.index > '2017-05-01') & (final_df.index < '2019-03-01')]
X_test = final_df[(final_df.index >= '2019-03-01')]

# Separate label
y_train = X_train.pop('platts_alumina_1-2.5%')
y_test = X_test.pop('platts_alumina_1-2.5%')

# Loss Function, RMSE
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Optimizer
opt = Adam(learning_rate = 0.003)

# Stopping callback
early_stop = EarlyStopping(monitor = 'val_loss', 
                           mode = 'min', 
                           verbose = 1, 
                           patience = 20)
# Model
def mlp_model():
  model = tf.keras.Sequential([   
    tf.keras.layers.Dense(3, input_dim = 3,
                          kernel_initializer = 'glorot_uniform',
                          activation = 'elu'),
    tf.keras.layers.Dropout(0.2),                      
    tf.keras.layers.Dense(160, 
                          activation = 'elu', 
                          kernel_regularizer = regularizers.l2(0.001)),                     
    tf.keras.layers.GaussianNoise(0.3),  
    tf.keras.layers.Dense(160, 
                          activation = 'elu',
                          kernel_regularizer = regularizers.l2(0.003)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(160, 
                          activation = 'elu',
                          kernel_regularizer = regularizers.l2(0.003)),                                               
    tf.keras.layers.Dense(160, 
                          activation = 'relu',
                          kernel_regularizer = regularizers.l2(0.004)), #128 for unscaled
    tf.keras.layers.Dense(1) 
  ])

  model.compile(optimizer = opt, 
                loss = root_mean_squared_error,
                metrics=['mean_squared_error', 
                         'mean_absolute_error',
                         root_mean_squared_error])
  return model

model = mlp_model()
model_history = model.fit(train_batch, 
                          validation_data = test_batch,
                          epochs = 100,
                          shuffle = False,
                          callbacks = [early_stop])