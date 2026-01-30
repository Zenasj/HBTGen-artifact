from tensorflow.keras import layers

import tensorflow as tf
import keras

def create_dataset():
    float_data = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    string_data = tf.constant([["foo", "bar"], ["baz", "qux"]], dtype=tf.string)
    labels = tf.constant([[1], [0]], dtype=tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(((float_data, string_data), labels))
    return dataset

def create_model():
    input_float = keras.Input(shape=(2,), dtype=tf.float32, name='float_input')
    input_string = keras.Input(shape=(2,), dtype=tf.string, name='string_input')
    
    string_lookup = keras.layers.StringLookup(vocabulary=["foo", "bar", "baz", "qux"], name='string_lookup')
    string_embedding = string_lookup(input_string)
    
    concatenated = keras.layers.Concatenate(name='concatenate')([input_float, string_embedding])
    
    dense = keras.layers.Dense(10, activation='relu', name='dense_1')(concatenated)
    output = keras.layers.Dense(1, activation='sigmoid', name='output')(dense)
    
    model = keras.Model(inputs=[input_float, input_string], outputs=output, name='simple_model')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Single GPU strategy")
    strategy = tf.distribute.get_strategy()
    
    dataset = create_dataset()
    
    with strategy.scope():
        model = create_model()
        
    model.fit(dataset.batch(2), epochs=5)

    print("Multiple GPUs strategy")

    strategy = tf.distribute.MirroredStrategy()
    
    dataset = create_dataset()
    
    with strategy.scope():
        model = create_model()
        
    model.fit(dataset.batch(2), epochs=5)

if __name__ == "__main__":
    main()

import tensorflow as tf
import keras

def create_dataset():
    float_data = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    string_data = tf.constant([["foo", "bar"], ["baz", "qux"]], dtype=tf.string)
    labels = tf.constant([[1], [0]], dtype=tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(((float_data, string_data), labels))
    return dataset

def create_model():
    input_float = keras.Input(shape=(2,), dtype=tf.float32, name='float_input')
    input_string = keras.Input(shape=(2,), dtype=tf.string, name='string_input')
    
    string_lookup = keras.layers.StringLookup(vocabulary=["foo", "bar", "baz", "qux"], name='string_lookup')
    string_embedding = string_lookup(input_string)
    
    concatenated = keras.layers.Concatenate(name='concatenate')([input_float, string_embedding])
    
    dense = keras.layers.Dense(10, activation='relu', name='dense_1')(concatenated)
    output = keras.layers.Dense(1, activation='sigmoid', name='output')(dense)
    
    model = keras.Model(inputs=[input_float, input_string], outputs=output, name='simple_model')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():

    print("Single GPU strategy")

    strategy = tf.distribute.get_strategy()
    n_gpu: int = len(tf.config.list_physical_devices('GPU'))
    n_replicas: int = strategy.num_replicas_in_sync
 
    print(f'GPU: {n_gpu}')
    print(f'Replicas: {n_replicas}')
    
    dataset = create_dataset()
    
    with strategy.scope():
        model = create_model()
        
    model.fit(dataset.batch(2), epochs=5)

    print("Multiple GPUs strategy")

    strategy = tf.distribute.MirroredStrategy()

    n_gpu: int = len(tf.config.list_physical_devices('GPU'))
    n_replicas: int = strategy.num_replicas_in_sync
 
    print(f'GPU: {n_gpu}')
    print(f'Replicas: {n_replicas}')
    
    dataset = create_dataset()
    
    with strategy.scope():
        model = create_model()
        
    model.fit(dataset.batch(2), epochs=5)

if __name__ == "__main__":
    main()