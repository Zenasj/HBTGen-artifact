from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def df_to_dataset(df, shuffle=True, batch_size=32):
    df = df.copy()
    labels = df.pop('target')
    ds = tf.data.Dataset.from_tensor_slices(
        (dict(df), labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


def generate_features():
    feature_columns = []
    feature_layer_inputs = {}


    thal = tf.feature_column.categorical_column_with_vocabulary_list(
          'thal', ['fixed', 'normal', 'reversible'])
    thal_one_hot = tf.feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)
    feature_layer_inputs['thal'] = tf.keras.Input(shape=(1,), name='thal', dtype=tf.string)

    return feature_columns, feature_layer_inputs


def create_model(feature_columns, feature_layer_inputs):
    input_layer = tf.keras.layers.DenseFeatures(feature_columns)
    inputs = input_layer(feature_layer_inputs)

    l1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
    l2 = tf.keras.layers.Dense(128, activation='relu')(l1)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(l2)

    model = tf.keras.Model(
        inputs=[v for v in feature_layer_inputs.values()],
        outputs=[output]
    )
    return model


def make_loss(loss_object):
    def loss(model, x, y):
        y_pred = model(x)
        return loss_object(y_true=y, y_pred=y_pred)
    return loss


def grad(model, inputs, targets, loss):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def fit(epochs, train_ds, model, optimizer, loss_obj):
    loss = make_loss(loss_obj)
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_ds):
            loss_values, grad_values = grad(model, x, y, loss)
            optimizer.apply_gradients(zip(grad_values, model.trainable_variables))


if __name__ == '__main__':
    URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
    df = pd.read_csv(URL)
    CUSTOM_TRAINING = True

    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    # hardcoded stuff
    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)

    # Create model and features
    feature_columns, feature_layer_inputs = generate_features()
    model = create_model(feature_columns, feature_layer_inputs)

    if CUSTOM_TRAINING:
        print('Trying custom training')
        bce = tf.keras.losses.BinaryCrossentropy()
        adam = tf.keras.optimizers.Adam()
        fit(epochs=5, train_ds=train_ds,
            model=model, optimizer=adam, loss_obj=bce)
    else:
        print('Using pre-defined fit')
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_ds, epochs=5)