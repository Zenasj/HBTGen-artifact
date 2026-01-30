from tensorflow.keras import layers
from tensorflow.keras import models

# keras == 2.3.1, tensorflow == 2.3.0
from tensorflow.python import keras  
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
tf_operations = model.outputs[0].graph.get_operations()
print(len(tf_operations))
# result: 8

# keras == 2.3.1, tensorflow == 1.15.0
from tensorflow.python import keras  
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
tf_operations = keras.backend.get_session().graph.get_operations()
print(len(tf_operations))
# result: 25

tf_operations = model.outputs[0].graph.get_operations()