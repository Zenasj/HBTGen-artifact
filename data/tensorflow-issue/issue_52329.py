import tensorflow as tf

def main():
    # Create model

    # Start with smaller model that processes the two images in the same way.
    single_image_input = keras.Input(shape=(256,256,3))

    image = layers.Conv2D(64, (3,3))(single_image_input)
    image = layers.LeakyReLU()(image)
    image = layers.BatchNormalization()(image)
    # Run through MaxPool2D to help the algorithm identify features in different areas of the image.
    # Has the effect of downsampling and cutting the dimensions in half.
    image = layers.MaxPool2D()(image)

    image = layers.Conv2D(128, (3, 3))(image)
    image = layers.LeakyReLU()(image)
    image = layers.BatchNormalization()(image)
    image = layers.Dropout(.3)(image)

    image_model = keras.Model(single_image_input, image)
    
    # Create larger model
    image_inputs = keras.Input(shape=(2,256,256,3))

    first_image, second_image = tf.split(image_inputs, num_or_size_splits=2, axis=0)
    first_image, second_image = tf.squeeze(first_image), tf.squeeze(second_image)

    image_outputs = [image_model(first_image), image_model(second_image)]
    model = layers.Concatenate()(image_outputs)

    model = layers.Flatten()(model)

    model = layers.Dense(128)(model)
    model = layers.LeakyReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.Dropout(.3)(model)

    # Output is change in y-position of drone
    out_layer = layers.Dense(1, activation='linear')(model)

    final_model = keras.Model(image_inputs, out_layer)
    final_model.compile(loss="mse", optimizer=optimizers.Adam(lr=0.0003, beta_1=0.7))

    image_model.summary()

    final_model.summary()


    #Preprocess data
    print("Loading and processing data...")
    train_data = tf_load_data()

    #Train model
    final_model.fit(train_data, epochs=5)