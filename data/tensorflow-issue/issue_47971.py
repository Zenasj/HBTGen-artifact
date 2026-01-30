from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
import time

n_samples = 300000
n_features = 100
n_targets = 5
batch_size = 100
n_inducing_points = 100
x = np.array(range(n_samples * n_features), dtype=np.float64).reshape((n_samples, n_features))
y = np.array(range(n_samples * n_targets), dtype=np.float64).reshape((n_samples, n_targets))
for t_idx in range(10):
    tf.keras.backend.clear_session()
    dataset = [x, y]
    dataset = tf.data.Dataset.from_tensor_slices(tuple(dataset)).shuffle(n_samples).repeat().batch(batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    data_iterator = iter(dataset)

    inputs = tf.keras.Input(shape=(n_features,), name='input')
    outputs = tf.keras.layers.Dense(n_features, name='dense_1', activation=tf.keras.activations.relu)(inputs)
    outputs = tf.keras.layers.Dense(n_features, name='dense_2', activation=tf.keras.activations.relu)(outputs)
    outputs = tf.keras.layers.Dense(n_features, name='dense_3', activation=tf.keras.activations.relu)(outputs)
    outputs = tf.keras.layers.Dense(n_features, name='dense_4', activation=tf.keras.activations.relu)(outputs)
    outputs = tf.keras.layers.Dense(n_targets, name='output', activation=tf.keras.activations.linear)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    trainable_variables = list(model.trainable_variables)

    adam_opt = tf.optimizers.Adam(learning_rate=0.001)


    @tf.function
    def loss(batch):
        x_, y_ = batch
        y_pred_ = model(x_)
        return tf.keras.losses.MSE(y_pred_, y_)


    @tf.function
    def optimization_step():
        batch = next(data_iterator)
        def f(): return loss(batch)
        adam_opt.minimize(f, var_list=trainable_variables)

    iterations = 50000
    loop_start = time.time()
    optimization_times = []
    for idx in range(iterations):
        optimization_step()

    loop_end = time.time()
    print(f'Elapsed: {loop_end - loop_start}')

Elapsed: 49.798316955566406
Elapsed: 55.18571472167969
Elapsed: 58.57510209083557
Elapsed: 64.41855955123901
Elapsed: 66.76858448982239
Elapsed: 68.3305652141571
Elapsed: 67.73438382148743
Elapsed: 69.73751258850098
Elapsed: 73.59102845191956
Elapsed: 73.14124798774719

Elapsed: 49.9630708694458
Elapsed: 53.02399563789368
Elapsed: 53.265995502471924
Elapsed: 53.51599383354187
Elapsed: 54.79951238632202
Elapsed: 55.34299564361572
Elapsed: 56.96799683570862
Elapsed: 57.81099009513855
Elapsed: 59.2099826335907
Elapsed: 59.92298483848572

Elapsed: 54.97995686531067
Elapsed: 52.393901348114014
Elapsed: 53.56126165390015
Elapsed: 53.747889280319214
Elapsed: 54.081284284591675
Elapsed: 55.353312253952026
Elapsed: 56.64463949203491
Elapsed: 55.35281229019165
Elapsed: 55.846272230148315
Elapsed: 56.97290635108948