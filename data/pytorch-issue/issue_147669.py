import torch.nn as nn

import tensorflow as tf 
print("TensorFlow version:", tf.__version__) 
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), 
  tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)
]) 
predictions = model(x_train[:1]).numpy() 
tf.nn.softmax(predictions).numpy() 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
loss_fn(y_train[:1], predictions).numpy() 
model.compile(optimizer='adam',loss=loss_fn, metrics=['accuracy']) 
model.fit(x_train, y_train, epochs=5) 
model.evaluate(x_test, y_test, verbose=2)

print("sys.executable", sys.executable)

print("out", out)

def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages."""
    if patterns is None:
        patterns = PIP_PATTERNS + COMMON_PATTERNS + NVIDIA_PATTERNS

    pip_version = 'pip3' if sys.version[0] == '3' else 'pip'

    os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    # People generally have pip as `pip` or `pip3`
    # But here it is invoked as `python -mpip`
    out = run_and_read_all(run_lambda, [sys.executable, '-mpip', 'list', '--format=freeze'])
    print("sys.executable",sys.executable)
    print("out",out)
    filtered_out = '\n'.join(
        line
        for line in out.splitlines()
        if any(name in line for name in patterns)
    )

    return pip_version, filtered_out