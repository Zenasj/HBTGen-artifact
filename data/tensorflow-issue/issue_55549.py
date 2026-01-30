from tensorflow.keras import layers

py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.get_logger().setLevel("INFO")
# tf.debugging.set_log_device_placement(True)
tf.config.experimental.set_memory_growth(
    device=tf.config.list_physical_devices("GPU")[0],
    enable=True
)

def test_gpu():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Image Classes (Input Image)
# Input Image = 28x28
# Input Layer: (28 * 28) = 784 Neurons
# Hidden Layer [1]
# Output Classes: 5
# Output Layer: 5 Neurons (Val: [0, 1]), simga(output_layer) = 1
# Activation FX: f(sigma(layer) + b)
# Cost/Loss FX: distance(expected, current) [MSE, MAE, HL]
# Gradient Descent: Function to tell us how to make cost/loss
# as minimum as possible. (Optimizing cost/loss fx)
# Optimizers just implement the above functions in different ways.
def nn_task():
    mnist = keras.datasets.fashion_mnist
    # (60K, 28, 28), [0, 0, 0] = 0  ---- 0: Black, 255: White
    (train_features, train_labels), (test_features, test_labels) = mnist.load_data()
    print(train_features.shape)
    label_names = [
        "Top", "Bottom", "Pullover",
        "Dress", "Coat", "Sandal",
        "Dress Shirt", "Sneaker",
        "Bag", "Ankle Boot"
    ]

    # plt.figure()
    # plt.imshow(train_features[0])
    # plt.colorbar()
    # plt.gray()
    # plt.plot()
    # plt.pause(5)

    # Normalize to scale of [0, 1]
    # Ensures that values only range from 0 to 1.
    train_features = train_features / 255
    test_features = test_features / 255

    model = keras.Sequential([
        # Flattens the input to 784 neurons.
        keras.layers.Flatten(input_shape=(28, 28), name="Input"),

        # Dense just means all neurons from previous
        # layer are connected to this layer too.
        # The number 128 is arbitrary - usually, you'd
        # want this number to be less than input neurons
        # but more than output neurons.
        # ReLU: x < 0 => y = 0 || x >= 0 => y = x
        keras.layers.Dense(128, activation="relu", name="DataParser"),

        # Dense just means all neurons from previous
        # layer are connected to this layer too.
        # Layer size is based on how many classes we have.
        # Softmax: sigma(layer) = 1
        keras.layers.Dense(len(label_names), activation="softmax", name="Output")
    ], "Fashion-Cloth-Classifier")

    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy",  # We use SCCE for loss.
        metrics=["accuracy"]  # Check only the accuracy metric.
    )
    model.summary()

    with tf.device("/GPU:0"):  # Change this line to train on CPU or GPU
        model.fit(train_features, train_labels, epochs=10, batch_size=1)

    t_loss, t_acc = model.evaluate(test_features, test_labels, verbose=1)

    print("Test Loss: ", t_loss)
    print("Test Accuracy: ", t_acc)


def work():
    nn_task()


if __name__ == '__main__':
    test_gpu()
    work()