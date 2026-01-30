import tensorflow as tf
from tensorflow import keras

class ANNForSentimentAnalysis(tf.keras.Model):
    def __init__(self, embedding = "https://tfhub.dev/google/nnlm-en-dim128/1", name="ANNForSentimentAnalysis", **kwargs):
        super(ANNForSentimentAnalysis, self).__init__(name=name, **kwargs)
        self._layers = [
            hub.KerasLayer(embedding, trainable=True, dtype=tf.string),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ]
        # self._model = Sequential(self._layers)

    @tf.function
    def call(self, inputs):
        # return self._model(inputs)
        for layer in self._layers:
            inputs = layer(inputs)
        return inputs

def train_step(train_data, model, optimizer, loss_func, metric):
    for i, batch in enumerate(train_data):
        x_train, y_train = batch
        with tf.GradientTape() as tape:
            preds = model(x_train)
            loss = loss_func(y_train, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        metric_eval = metric(y_train, preds)
        sys.stdout.write(f"\rStep {i}: [" + i*"*" + f"]\tloss: {loss:.4f}\taccuracy: {metric_eval:.4f}")
    print("")

def accuracy(y_train, y_pred):
    return tf.reduce_mean(tf.cast(tf.cast(y_train, tf.float32) == tf.cast(y_pred > 0.5, tf.float32), tf.float32))