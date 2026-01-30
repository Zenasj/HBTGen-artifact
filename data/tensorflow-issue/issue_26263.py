import tensorflow as tf
from tensorflow.keras import layers

# Import relevant packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define the number of inputs and outputs
n_input_nodes = 2
n_output_nodes = 3

# First define the model 
model = Sequential()

'''TODO: Define a dense (fully connected) layer to compute z'''
dense_layer = Dense(n_output_nodes, input_shape=(n_input_nodes,),activation='sigmoid') # TODO 

# Add the dense layer to the model
model.add(dense_layer)

# Test model with example input
x_input = tf.constant([[1.0,2.]], shape=(1,2))
'''TODO: feed input into the model and predict the output!'''
print(model.predict(x_input)) # TODO