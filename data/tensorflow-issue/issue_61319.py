dnn_model = keras.Sequential([
        normalizer,
        layers.Dense(512, activation='relu', input_dim=x_train.shape[1]),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

interpreter.resize_tensor_input(input_details['index'], (1,8))