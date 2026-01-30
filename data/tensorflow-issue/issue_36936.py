import tensorflow as tf
from tensorflow import keras

for epoch in range(epochs):
        for step, (x_batch, y_batch) in enumerate(train_dataset):

            if epoch > mlp.warm_up:
                for layer in mlp.mean.layers:
                    layer.trainable = False
                for layer in mlp.variance.layers:
                    layer.trainable = True
            with tf.GradientTape() as tape:
                output = mlp(x_batch)
                loss = loss_fn(y_batch, output)
            grads = tape.gradient(loss, mlp.trainable_weights)
            optimizer.apply_gradients(zip(grads, mlp.trainable_weights))
            mean_preds, _ = output

def call(self, inputs, training=None, mask=None):

        mean_predictions = []
        variance_predictions = []
        for idx in range(self.num_models):
            mean_predictions.append(self.mean[idx](inputs))
            variance_predictions.append(self.variance[idx](inputs))
        mean_stack = tf.stack(mean_predictions)
        variance_stack = tf.stack(variance_predictions)

        return mean_stack, variance_stack

self.mean = [
            tf.keras.Sequential([
                layers.Dense(512, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.output_dim)
            ])
            for _ in range(self.num_models)
        ]