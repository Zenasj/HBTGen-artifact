import random
from tensorflow.keras import layers
from tensorflow.keras import models

def get_model(num_inputs, num_outputs):
    from keras.layers import Dense, Input
    from keras.models import Model

    inputs = Input((num_inputs,))
    outputs = Dense(num_outputs)(inputs)
    return Model(inputs, outputs)


def get_custom_auc(output):
    import tensorflow as tf

    # I may also want to use other metrics other than AUC (e.g. BinaryAccuracy).
    auc = tf.metrics.AUC()

    @tf.function
    def custom_auc(y_true, y_pred):
        y_true = y_true[:, output]
        y_pred = y_pred[:, output]
        auc.update_state(y_true, y_pred)
        return auc.result()

    custom_auc.__name__ = "custom_auc_" + str(output)
    return custom_auc


def train():
    import numpy as np

    num_inputs = 5
   
    # I want to implement an AUC metric for each of these outputs SEPARATELY.
    num_outputs = 3 

    num_examples = 10

    model = get_model(num_inputs, num_outputs)

    # Create a separate AUC metric for each of the outputs.
    metrics = [get_custom_auc(m) for m in range(num_outputs)]

    # I want to visualize the metrics for each of the outputs (separately) during training.
    model.compile(loss='mse', metrics=metrics, optimizer='adam')

    print(model.metrics)

    # Error occurs when calling fit.
    model.fit(np.random.rand(num_examples, num_inputs), np.zeros((num_examples, num_outputs)))


if __name__ == '__main__':
    train()

priority

department