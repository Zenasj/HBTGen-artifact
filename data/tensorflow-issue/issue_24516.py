# tf.random.uniform((2, 256, 256, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Subtract, Lambda
from tensorflow.keras.models import Model, Sequential

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        input_shape = (256,256,1)
        # Build the shared convolutional subnetwork (convnet) per original code
        self.convnet = Sequential([
            Conv2D(64, (9,9), activation='relu', input_shape=input_shape,
                   kernel_initializer='random_normal', kernel_regularizer=l2(2e-4)),
            MaxPooling2D(),
            Conv2D(128, (7,7), activation='relu',
                   kernel_regularizer=l2(2e-4),
                   kernel_initializer='random_normal',
                   bias_initializer='random_normal'),
            MaxPooling2D(),
            Conv2D(256, (5,5), activation='relu',
                   kernel_initializer='random_normal',
                   kernel_regularizer=l2(2e-4),
                   bias_initializer='random_normal'),
            MaxPooling2D(),
            Conv2D(64, (3,3), activation='relu',
                   kernel_initializer='random_normal',
                   kernel_regularizer=l2(2e-4),
                   bias_initializer='random_normal'),
            Flatten(),
            Dense(2048, activation='sigmoid',
                  kernel_regularizer=l2(1e-3),
                  kernel_initializer='random_normal',
                  bias_initializer='random_normal'),
        ])

        # Layers for final comparison and prediction
        self.subtract = Subtract()
        self.abs_lambda = Lambda(lambda x: K.abs(x))
        self.prediction_dense = Dense(1, activation='sigmoid',
                                      bias_initializer='random_normal')
        # Define inputs as layers for functional usage inside call; 
        # Inputs will be provided externally so not defined as Input layers here.

        # Optimizer and loss will be configured externally at compile
    
    def call(self, inputs, training=False):
        # inputs will be tuple/list of two tensors: (left, right)
        left, right = inputs  # Expect shape (batch, 256, 256, 1)

        encodedL = self.convnet(left, training=training)
        encodedR = self.convnet(right, training=training)

        diff = self.abs_lambda(self.subtract([encodedL, encodedR]))
        prediction = self.prediction_dense(diff)
        return prediction

def my_model_function():
    model = MyModel()

    # Build model by calling once to initialize weights
    # (Dummy inputs for shape inference)
    dummy_input = (
        tf.random.uniform((1, 256, 256, 1), dtype=tf.float32),
        tf.random.uniform((1, 256, 256, 1), dtype=tf.float32)
    )
    model(dummy_input)

    # Compile model with same optimizer and loss as original code
    # Note: run_opts with report_tensor_allocations_upon_oom is TF1.x style,
    # in TF2.x we just compile normally here.
    optimizer = Adam(learning_rate=6e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def GetInput():
    # Must return a pair of tensors matching the model inputs shape:
    # (batch_size, 256, 256, 1)
    batch_size = 32  # typical batch size as in original code
    input_shape = (batch_size, 256, 256, 1)

    # Generate random inputs simulating grayscale images
    left_input = tf.random.uniform(input_shape, dtype=tf.float32)
    right_input = tf.random.uniform(input_shape, dtype=tf.float32)
    return (left_input, right_input)

