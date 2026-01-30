def square_it(x):
    return (x ** 2)

def output_of_lambda(input_shape):
    return (None, None, 10, 20)

model_2 = keras.Sequential(
    [
        layers.Input(shape=(784)),
        layers.Dense(2, activation="relu", name="layer1"),
        lambda_layer,
        layers.Dense(4, name="layer5"),
    ]
)