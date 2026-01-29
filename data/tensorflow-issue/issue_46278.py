# tf.random.uniform((1024, None), dtype=tf.int64) and (batch_size, None) ragged int inputs for sparse features

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class DenseToRaggedLayer(tf.keras.layers.Layer):
    """Converts dense tensor padded by ignore_value (-1) to RaggedTensor."""
    def __init__(self, ignore_value=-1, **kwargs):
        super(DenseToRaggedLayer, self).__init__(**kwargs)
        self.ignore_value = ignore_value

    def call(self, inputs):
        return tf.RaggedTensor.from_tensor(inputs, padding=self.ignore_value)


def embedding_dim_fn(bucket_size):
    # Embedding dim heuristic from the original code: 2 ^ (ceil(log(bucket_size^0.25)) + 3)
    return int(np.power(2, np.ceil(np.log(bucket_size ** 0.25)) + 3))


class WideModel(tf.keras.Model):
    def __init__(self, simple_sparse_config, share_embedding_sparse_config):
        super().__init__()
        self.wide_encoders = []
        self.wide_hash_layers = []
        self.wide_dense_to_ragged = []
        self.simple_sparse_config = simple_sparse_config
        self.share_embedding_sparse_config = share_embedding_sparse_config

        # For simple sparse features
        for feature_name, conf in simple_sparse_config.items():
            # Layers for wide part
            self.wide_dense_to_ragged.append(DenseToRaggedLayer(name=feature_name + '_rag'))
            self.wide_hash_layers.append(preprocessing.Hashing(num_bins=conf['bucket_size'], name=feature_name + '_hash'))
            self.wide_encoders.append(
                preprocessing.CategoryEncoding(max_tokens=conf['bucket_size'], output_mode='binary', sparse=True, name=feature_name + '_cat')
            )

        # For shared embedding sparse features (only wide part)
        self.share_wide_dense_to_ragged = []
        self.share_wide_hash_layers = []
        self.share_wide_encoders = []
        for conf in share_embedding_sparse_config:
            for feature_name in conf['columns'].keys():
                self.share_wide_dense_to_ragged.append(DenseToRaggedLayer(name=feature_name + '_rag'))
                self.share_wide_hash_layers.append(preprocessing.Hashing(num_bins=conf['bucket_size'], name=feature_name + '_hash'))
                self.share_wide_encoders.append(
                    preprocessing.CategoryEncoding(max_tokens=conf['bucket_size'], output_mode='binary', sparse=True, name=feature_name + '_cat')
                )
        # Final linear model for wide inputs
        self.linear_model = keras.experimental.LinearModel()

    def call(self, inputs):
        # inputs is dict of features (tensors)
        wide_vectors = []

        # simple sparse features wide encoding
        for i, (feature_name, conf) in enumerate(self.simple_sparse_config.items()):
            x = inputs[feature_name]
            x = self.wide_dense_to_ragged[i](x)
            x = self.wide_hash_layers[i](x)
            x = self.wide_encoders[i](x)
            wide_vectors.append(x)

        # shared embedding sparse features wide encoding
        offset = len(self.simple_sparse_config)
        for j, conf in enumerate(self.share_embedding_sparse_config):
            bucket_size = conf['bucket_size']
            for k, feature_name in enumerate(conf['columns'].keys()):
                idx = offset + sum(len(c['columns']) for c in self.share_embedding_sparse_config[:j]) + k
                x = inputs[feature_name]
                x = self.share_wide_dense_to_ragged[idx-offset](x)
                x = self.share_wide_hash_layers[idx-offset](x)
                x = self.share_wide_encoders[idx-offset](x)
                wide_vectors.append(x)

        wide_output = self.linear_model(wide_vectors)
        return wide_output


class DeepModel(tf.keras.Model):
    def __init__(self, simple_sparse_config, share_embedding_sparse_config):
        super().__init__()
        self.simple_sparse_config = simple_sparse_config
        self.share_embedding_sparse_config = share_embedding_sparse_config

        self.deep_embeddings = []
        self.deep_dense_to_ragged = []
        self.deep_hash_layers = []

        # For simple sparse features deep part: embedding + pooling layers
        for feature_name, conf in simple_sparse_config.items():
            self.deep_dense_to_ragged.append(DenseToRaggedLayer(name=feature_name + '_rag'))
            self.deep_hash_layers.append(preprocessing.Hashing(num_bins=conf['bucket_size'], name=feature_name + '_hash'))
            emb_dim = embedding_dim_fn(conf['bucket_size'])
            emb_layer = layers.Embedding(conf['bucket_size'], emb_dim, name=feature_name + '_emb')
            self.deep_embeddings.append(emb_layer)

        # For shared embeddings
        self.shared_emb_layers = []
        for conf in share_embedding_sparse_config:
            emb_layer = layers.Embedding(conf['bucket_size'], conf['embedding_size'], name=conf['name'] + '_share_emb')
            self.shared_emb_layers.append(emb_layer)

        # Dense layers for DNN part (512 units x4)
        self.dnn_layers = [
            layers.Dense(512, activation=None),
            layers.Dense(512, activation=None),
            layers.Dense(512, activation=None),
            layers.Dense(1, activation=None),
        ]

    def call(self, inputs):
        deep_vectors = []

        # simple sparse features deep vector encoding
        for i, (feature_name, conf) in enumerate(self.simple_sparse_config.items()):
            x = inputs[feature_name]
            x = self.deep_dense_to_ragged[i](x)
            x = self.deep_hash_layers[i](x)
            x = self.deep_embeddings[i](x)
            x = layers.GlobalAveragePooling1D(name=feature_name + '_avg_pool')(x)
            deep_vectors.append(x)

        # shared embedding sparse features deep vector encoding
        offset = len(self.simple_sparse_config)
        idx_emb = 0
        for conf in self.share_embedding_sparse_config:
            emb_layer = self.shared_emb_layers[idx_emb]
            idx_emb += 1
            for feature_name in conf['columns'].keys():
                x = inputs[feature_name]
                x = DenseToRaggedLayer(name=feature_name + '_rag')(x)
                x = preprocessing.Hashing(num_bins=conf['bucket_size'], name=feature_name + '_hash')(x)
                x = emb_layer(x)
                x = layers.GlobalAveragePooling1D(name=feature_name + '_avg_pool')(x)
                deep_vectors.append(x)

        deep_concat = layers.Concatenate()(deep_vectors)
        x = deep_concat
        for layer in self.dnn_layers:
            x = layer(x)
        return x


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define configs same as the issue
        self.simple_sparse_config = {
            'sparse_feature1': {'bucket_size': 2100},
            'sparse_feature2': {'bucket_size': 5000000},
            'sparse_feature5': {'bucket_size': 500000},
            'sparse_feature6': {'bucket_size': 800000},
            'sparse_feature7': {'bucket_size': 800000},
            'sparse_feature8': {'bucket_size': 30000},
            'sparse_feature9': {'bucket_size': 30000},
            'sparse_feature10': {'bucket_size': 23000},
            'sparse_feature11': {'bucket_size': 23000},
            'sparse_feature12': {'bucket_size': 800000},
            'sparse_feature13': {'bucket_size': 800000},
            'sparse_feature14': {'bucket_size': 80000},
            'sparse_feature15': {'bucket_size': 80000},
            'sparse_feature16': {'bucket_size': 30000},
            'sparse_feature17': {'bucket_size': 30000},
            'sparse_feature19': {'bucket_size': 100000},
        }
        self.share_embedding_sparse_config = [
            {'name': 'ss1',
             'columns': {'sparse_feature_20': {}, 'sparse_feature_21': {}, 'sparse_feature_22': {}, 'sparse_feature_23': {}},
             'bucket_size': 220000,
             'embedding_size': 128},
            {'name': 'ss2',
             'columns': {'sparse_feature_24': {}, 'sparse_feature_25': {}, 'sparse_feature_26': {}},
             'bucket_size': 260000,
             'embedding_size': 128},
            {'name': 'ss3',
             'columns': {'sparse_feature_27': {}, 'sparse_feature_28': {}, 'sparse_feature_29': {}},
             'bucket_size': 7500000,
             'embedding_size': 64}  # limited by protobuf size issue in multi worker
        ]

        self.wide_model = WideModel(self.simple_sparse_config, self.share_embedding_sparse_config)
        self.deep_model = DeepModel(self.simple_sparse_config, self.share_embedding_sparse_config)

    def call(self, inputs):
        wide_out = self.wide_model(inputs)
        deep_out = self.deep_model(inputs)
        # Combine wide and deep outputs (simple addition, can be changed if needed)
        output = wide_out + deep_out
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Construct a dictionary of inputs with batch_size=1024, ragged int64 tensors (padded with -1)
    batch_size = 1024
    # For simplicity, simulate each sparse feature input as a dense tensor of shape (batch_size, variable_length)
    # Here we fix length=5 for demo; replace None with actual variable length if needed
    length = 5

    simple_sparse_config = {
        'sparse_feature1': 2100,
        'sparse_feature2': 5000000,
        'sparse_feature5': 500000,
        'sparse_feature6': 800000,
        'sparse_feature7': 800000,
        'sparse_feature8': 30000,
        'sparse_feature9': 30000,
        'sparse_feature10': 23000,
        'sparse_feature11': 23000,
        'sparse_feature12': 800000,
        'sparse_feature13': 800000,
        'sparse_feature14': 80000,
        'sparse_feature15': 80000,
        'sparse_feature16': 30000,
        'sparse_feature17': 30000,
        'sparse_feature19': 100000,
    }
    share_embedding_sparse_config = [
        {'name': 'ss1',
         'columns': ['sparse_feature_20', 'sparse_feature_21', 'sparse_feature_22', 'sparse_feature_23'],
         'bucket_size': 220000,
         'embedding_size': 128},
        {'name': 'ss2',
         'columns': ['sparse_feature_24', 'sparse_feature_25', 'sparse_feature_26'],
         'bucket_size': 260000,
         'embedding_size': 128},
        {'name': 'ss3',
         'columns': ['sparse_feature_27', 'sparse_feature_28', 'sparse_feature_29'],
         'bucket_size': 7500000,
         'embedding_size': 64}
    ]

    inputs = dict()
    # Create random integer inputs with some positions padded with -1
    def random_feature_input(bucket_size):
        # Generate random indices in [0, bucket_size), shape (batch_size, length)
        data = tf.random.uniform((batch_size, length), minval=0, maxval=bucket_size, dtype=tf.int64)
        # Randomly set some values to -1 to simulate padding
        mask = tf.random.uniform((batch_size, length), minval=0, maxval=1) < 0.2
        data = tf.where(mask, -1 * tf.ones_like(data), data)
        return data

    for fname, bsize in simple_sparse_config.items():
        inputs[fname] = random_feature_input(bsize)

    for conf in share_embedding_sparse_config:
        for fname in conf['columns']:
            inputs[fname] = random_feature_input(conf['bucket_size'])

    return inputs

