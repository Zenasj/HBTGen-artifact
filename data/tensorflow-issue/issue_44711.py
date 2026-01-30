import numpy as np
import tensorflow as tf
from tensorflow import keras

def generate_batches(
    x: np.ndarray | tf.Tensor, batch_size: int = 32
) -> np.ndarray | tf.Tensor:
    """Generate batches of test data for inference.

    Args:
        x (np.ndarray | tf.Tensor):
            Full test data set.
        batch_size (int, default=32):
            Batch size.

    Yields:
        np.ndarray | tf.Tensor:
            Yielded batches of test data.
    """
    for index in range(0, x.shape[0], batch_size):
        yield x[index : index + batch_size]


def predict(
    model: tf.keras.Model,
    x: np.ndarray | tf.Tensor,
    batch_size: int = 32,
) -> np.ndarray:
    """Predict using generated batched of test data.

    - Used instead of model.predict() due to memory leaks.
    - https://github.com/tensorflow/tensorflow/issues/44711

    Args:
        model (tf.keras.Model):
            The model to use for prediction.
        x (np.ndarray | tf.Tensor):
            Full test data set.
        batch_size (int, default=32):
            Batch size.

    Returns:
        np.ndarray:
            Predictions on the test data.
    """
    y_batches = []
    for x_batch in generate_batches(x=x, batch_size=batch_size):
        y_batch = model(x_batch, training=False).numpy()
        y_batches.append(y_batch)

    return np.concatenate(y_batches)


# instead of
# y_pred = model.predict(x_test)

# use
y_pred = predict(model=model, x=x_test, batch_size=32)

def create_tf_dataset(
    data_split: str,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    use_mixed_precision: bool,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset.

    - Cache train data before shuffling for performance (consider full dataset size).
    - Shuffle train data to increase accuracy (not needed for validation or test data).
    - Batch train data after shuffling for unique batches at each epoch.
    - Cache test data after batching as batches can be the same between epochs.
    - End pipeline with prefetching for performance.
    
    Args:
        data_split (str):
            The data split to create the dataset for.
            Supported are "train", "validation", and "test".
        x (np.ndarray):
            The feature data.
        y (np.ndarray):
            The target data.
        batch_size (int):
            The batch size.
        use_mixed_precision (bool):
            Whether to use mixed precision.

    Raises:
        ValueError: If the data split is not supported.

    Returns:
        tf.data.Dataset:
            The TensorFlow dataset.
    """
    if data_split not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid data split: {data_split}")

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        x = x.astype(np.float16)
        y = y.astype(np.float16)

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if data_split == "train":
        ds = ds.cache()
        set_random_seed(seed=RANDOM_SEED)
        ds = ds.shuffle(number_of_samples, seed=RANDOM_SEED)
        ds = ds.batch(batch_size)
    else:
        ds = ds.batch(batch_size)
        ds = ds.cache()

    ds = ds.prefetch(AUTOTUNE)

    return ds


# need to do this call separately on a machine with enough memory
ds_test = create_tf_dataset(
    data_split="test",
    x=x_test,
    y=y_test,
    batch_size=32,
    use_mixed_precision=True,
)

# then use it
y_pred = model.predict(ds_test)