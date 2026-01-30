import tensorflow as tf
from tensorflow.keras import layers

# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):
    
    c_str = ''
    
    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'
    
    # Add array length at top of file
    c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'
    
    # Declare C variable
    c_str += 'unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data):

        # Construct string from hex
        hex_str = format(val, '#04x')
        
        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)
        
    # Add closing brace
    c_str += '\n  ' + format(' '.join(hex_array)) + '\n};\n\n'
    
    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'
    
    return c_str

# Write TFLite model to a C source file
c_model_name = "mnist_model_quant_io"
with open(c_model_name + '.h', 'w') as file:
    file.write(hex_to_c_array(tflite_model_quant, c_model_name))

model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=4, kernel_size=(2, 2), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(4, 4)),
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])