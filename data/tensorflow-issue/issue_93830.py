import random
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow Version: {tf.__version__}")

# Optional: Test with eager execution
# tf.config.run_functions_eagerly(True)

# 1. Define Model Inputs (all initially with symbolic batch_size=None)
vocab_list = ["1", "2", "3", "cat", "dog", "mouse"] # Example vocabulary
input_a = keras.Input(shape=(1,), dtype=tf.string, name='feature_a')
input_b = keras.Input(shape=(1,), dtype=tf.string, name='feature_b')
input_c = keras.Input(shape=(1,), dtype=tf.string, name='feature_c')

# 2. StringLookup for each feature
lookup_a = keras.layers.StringLookup(vocabulary=vocab_list, mask_token=None, num_oov_indices=0, name='lookup_a')(input_a)
lookup_b = keras.layers.StringLookup(vocabulary=vocab_list, mask_token=None, num_oov_indices=0, name='lookup_b')(input_b)
lookup_c = keras.layers.StringLookup(vocabulary=vocab_list, mask_token=None, num_oov_indices=0, name='lookup_c')(input_c)

print(f"Initial shape of lookup_a: {lookup_a.shape}")
print(f"Initial shape of lookup_b: {lookup_b.shape}")
print(f"Initial shape of lookup_c: {lookup_c.shape}")

# 3. Chained HashedCrossing
num_bins_crossing = 1000
# First cross (A x B)
cross_ab = keras.layers.HashedCrossing(num_bins=num_bins_crossing, name='cross_ab')([lookup_a, lookup_b])
print(f"Shape of cross_ab: {cross_ab.shape}")

# Second cross ((A x B) x C) - This is where the error is expected during model.fit()
# due to shape mismatch between cross_ab and lookup_c
cross_abc = keras.layers.HashedCrossing(num_bins=num_bins_crossing, name='cross_abc')([cross_ab, lookup_c])
print(f"Shape of cross_abc: {cross_abc.shape}")


# 4. Embedding and Output layers
embedding_dim = 8
embedding = keras.layers.Embedding(input_dim=num_bins_crossing, output_dim=embedding_dim, name='embedding')(cross_abc)
flatten = keras.layers.Flatten(name='flatten')(embedding)
output = keras.layers.Dense(1, activation='sigmoid', name='output')(flatten)

# 5. Create and Compile Model
model = keras.Model(inputs=[input_a, input_b, input_c], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 6. Create Dummy Data
num_samples = 200
batch_s = 10 # Using a batch size that will appear in the error

dummy_a_np = np.random.choice(vocab_list, size=(num_samples, 1))
dummy_b_np = np.random.choice(vocab_list, size=(num_samples, 1))
dummy_c_np = np.random.choice(vocab_list, size=(num_samples, 1))

# Ensure input arrays are of dtype=object for robust tf.string conversion
dummy_a = dummy_a_np.astype(object)
dummy_b = dummy_b_np.astype(object)
dummy_c = dummy_c_np.astype(object)

dummy_labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

print(f"\nFeeding data with batch_size = {batch_s}")
# 7. Fit the model - Error is expected here
try:
    model.fit(
        [dummy_a, dummy_b, dummy_c],
        dummy_labels,
        batch_size=batch_s,
        epochs=1,
        verbose=2
    )
except ValueError as e:
    print(f"\nCaught expected ValueError: {e}")
    import traceback
    traceback.print_exc()

print("\nScript finished.")