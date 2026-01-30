from tensorflow.keras import layers

keras.Sequential([
            keras.layers.Flatten(input_shape=(self.dataset.features_num(),)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(nr_classes, activation='softmax')
        ])