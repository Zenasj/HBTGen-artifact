import random
from tensorflow.keras import layers
from tensorflow.keras import models

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import math
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from scipy.sparse import hstack, csr_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations

# ---- Initialize Encoders & Scaler ----
onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()

# ---- Stream Train/Test Split in Chunks ----
def stream_split_data(parquet_files, categorical_columns, numeric_columns):
    """Splits merged data into training and testing sets dynamically in chunks."""
    for chunk in stream_merged_data(parquet_files, categorical_columns, numeric_columns, merge_key):
        #np.random.seed(42)  # Ensures the split is always the same upon restarting
        mask = np.random.rand(len(chunk)) < train_ratio
        yield chunk[mask], chunk[~mask] 

# ---- Fit OneHotEncoder & StandardScaler in Chunks ----
numeric_columns_for_model = [col for col in numeric_columns if col != "Result"]
for train_chunk, _ in stream_split_data(parquet_files, categorical_columns, numeric_columns):
    X_train = train_chunk.drop(columns=["Result"]) # Do not fit target variable
    scaler.partial_fit(X_train[numeric_columns_for_model])  # Fit StandardScaler

def get_all_categories(parquet_files, categorical_columns, merge_key):
    # Initialize sets to collect unique values for each categorical column.
    all_categories = {col: set() for col in categorical_columns}
    for chunk in stream_merged_data(parquet_files, categorical_columns, numeric_columns, merge_key):
        for col in categorical_columns:
            all_categories[col].update(chunk[col].unique())
    # Convert sets to sorted lists (or lists in any consistent order)
    for col in all_categories:
        all_categories[col] = sorted(list(all_categories[col]))
    return all_categories

complete_categories = get_all_categories(parquet_files, categorical_columns, merge_key)

one_hot_encoder = OneHotEncoder(
    categories=[complete_categories[col] for col in categorical_columns],
    sparse_output=False,
    handle_unknown="ignore"
)
# Fit it on a dummy DataFrame with the correct column names.
dummy_df = pd.DataFrame([["0"] * len(categorical_columns)], columns=categorical_columns)
one_hot_encoder.fit(dummy_df)


print(f"Preprocessing the data in batches...")


# ---- Generator for Streaming Batches into `tf.data.Dataset` ----
def preprocess_dense_batches(parquet_files, categorical_columns, numeric_columns, batch_size, end_idx=None):
    """Generator function for streaming chunks into tf.data.Dataset dynamically."""

    label_mapping = {0.5: 0, 0: 1, 1: 2}  

    current_idx = 0
    for train_chunk, _ in stream_split_data(parquet_files, categorical_columns, numeric_columns):
        
        y_chunk = train_chunk.pop("Result").apply(lambda x: label_mapping[x]).values.astype(int)  # Extracts y (target variable) and drops column from the rest of the data in one step

        X_chunk = np.hstack((
            one_hot_encoder.transform(train_chunk[categorical_columns]),  
            scaler.transform(train_chunk[numeric_columns_for_model])  
        )).astype(np.float32)

        # Yield batches from each chunk
        for i in range(0, len(X_chunk), batch_size):
            if end_idx is not None and current_idx >= end_idx:
                return  # Stop when we reach end index
            X_batch = X_chunk[i:i + batch_size]
            y_batch = y_chunk[i:i + batch_size]

            yield X_batch, y_batch
            current_idx += batch_size

# ---- Create `tf.data.Dataset` Using the Chunked Generator ----
train_dataset = tf.data.Dataset.from_generator(
    lambda: preprocess_dense_batches(parquet_files, categorical_columns, numeric_columns, batch_size, end_idx=int(train_ratio * total_rows)),
    output_signature=(
        tf.TensorSpec(shape=(None, one_hot_encoder.transform(pd.DataFrame([["0"] * len(categorical_columns)], columns=categorical_columns)).shape[1] + len(numeric_columns_for_model)), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).repeat().prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: preprocess_dense_batches(parquet_files, categorical_columns, numeric_columns, batch_size, end_idx=total_rows),
    output_signature=(
        tf.TensorSpec(shape=(None, one_hot_encoder.transform(pd.DataFrame([["0"] * len(categorical_columns)], columns=categorical_columns)).shape[1] + len(numeric_columns_for_model)), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).prefetch(buffer_size=tf.data.AUTOTUNE)

# ---- Class Weights Handling ----
class_weight_dict = {
    0: 1.0,  # Pushes (0.5)
    1: 2.0,  # Misses (0)
    2: 3.0   # Hits (1)
}

# ---- Calculate Steps per Epoch ----
steps_per_epoch = math.ceil((train_ratio * total_rows) / batch_size)
test_steps = math.ceil(((1 - train_ratio) * total_rows) / batch_size)

print("Datasets created dynamically.")

# ---- Define Your Neural Network (Unchanged) ----
input_all_data = Input(shape=(one_hot_encoder.transform(pd.DataFrame([["0"] * len(categorical_columns)], columns=categorical_columns)).shape[1] + len(numeric_columns_for_model),), name='all_data_input')

print("Creating the dense layers...")

# Create the dense layers
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_all_data)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

output = Dense(3, activation='softmax')(x)  # 3 neurons, softmax for multi-class

print("Compiling the model...")

# Compile the model
model = Model(inputs=input_all_data, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ---- Train Model Using Existing Training Loop ----
print("\nBegin training the model.\n")

history_all_nfl_data = model.fit(
    train_dataset,                    # Training dataset
    validation_data=test_dataset,     # Testing dataset
    epochs=1,                    
    steps_per_epoch=steps_per_epoch,
    validation_steps=test_steps,
    class_weight=class_weight_dict,  # Pass class weights
    verbose=1
)