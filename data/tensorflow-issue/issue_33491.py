from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import feature_column

print('tf version:', tf.__version__)

feature_description = {
    'h_k_u_watchanch_his': tf.io.VarLenFeature(tf.string),
    'a_gender': tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64),
    'l_label': tf.io.FixedLenFeature([], tf.int64)
}
feature_columns = []
thal = feature_column.categorical_column_with_hash_bucket(
    'h_k_u_watchanch_his', hash_bucket_size=100
)
thal_one_hot = feature_column.embedding_column(thal, dimension=10, combiner='mean')
feature_columns.append(thal_one_hot)

dataSet = tf.data.TFRecordDataset(
    "/Users/lyx/projects/recommend/embedding/tmp/PUSH.TFRecords/dt=20191012/hour=10/part-r-00000")


def _parse_function(serilized_example):
    feature = tf.io.parse_single_example(
        serilized_example,
        feature_description
    )
    label = feature.get('l_label')
    return feature, label


parsed_dataset = dataSet.map(_parse_function)

input1 = tf.keras.Input(shape=(), name='h_k_u_watchanch_his', sparse=True, dtype=tf.string)
input2 = tf.keras.Input(shape=(), name='a_gender', dtype=tf.int64)

input_layers = {'h_k_u_watchanch_his': input1, 'a_gender': input2}
feature_layer = tf.keras.layers.DenseFeatures(feature_columns, name='DenseFeatures')(input_layers)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(feature_layer)
model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x=parsed_dataset,
    validation_data=parsed_dataset,
    epochs=5,
)

loss, accuracy = model.evaluate(parsed_dataset)
print("Accuracy", accuracy)