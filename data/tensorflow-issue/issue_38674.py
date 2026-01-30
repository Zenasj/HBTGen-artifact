import math
import tensorflow as tf

def call(self, inputs, training=None, mask=None):
        mean_predictions = []
        variance_predictions = []
        for idx in range(self.num_models):
            mean_predictions.append(self.mean[idx](inputs, training=training))
            variance_predictions.append(self.variance[idx](inputs, training=training))
        mean_stack = tf.stack(mean_predictions)
        variance_stack = tf.stack(variance_predictions)
        return mean_stack, variance_stack

class GaussianNLL(Loss):

    def __init__(self):
        super(GaussianNLL, self).__init__()

    def call(self, y_true, y_pred):

        mean, variance = y_pred
        variance = y_pred + 0.0001
        nll = (tf.math.log(variance) / 2 + ((y_true - mean) ** 2) / (2 * variance))
        nll = tf.math.reduce_mean(nll)
        return nll

model.compile(optimizer='adam',
                  loss=loss_fn)
history = model.fit(x_train, y_train, y_train,
                        batch_size=2048,
                        epochs=10000,
                        verbose=1,
                        validation_data=(x_val, y_val))

mean = y_pred[0]
variance = y_pred[1] + 0.0001