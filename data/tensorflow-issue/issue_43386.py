import tensorflow as tf

model = Sequential([
    Conv2D(4, 3, 
           padding='same',
           activation='relu',
           input_shape=(IMG_WIDTH, IMG_HEIGHT, 1),
           name='conv_layer1'),
    MaxPooling2D(name='max_pooling1'),
    Conv2D(4, 3, 
           padding='same',
           activation='relu',
           name='conv_layer2'),
    MaxPooling2D(name='max_pooling2', pool_size=(2,2)),
    Flatten(),
    Dense(
        20,
        activation='relu',
        name='hidden_layer'
    ),
    Dense(1, activation='sigmoid', name='output')
])

converter = tf.lite.TFLiteConverter.from_saved_model("checkpoint.model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
model = converter.convert()
open("converted_model.tflite", "wb").write(model)