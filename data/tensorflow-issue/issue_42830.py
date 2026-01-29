# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← input shape unknown from issue, so we assume a generic 2D input with batch dimension (B, features)
import tensorflow as tf

class WB_Init(tf.keras.initializers.Initializer):
    def __init__(self, dat, name=None):
        # To enable correct loading from saved model, dat should be saved as a tensor or numpy array
        # Also, to be Keras-serializable, implementing get_config/from_config is necessary
        # This is a fixed version that requires dat at construction and supports serialization.
        self.dat = dat
        self.name = name

    def __call__(self, shape, dtype=None):
        # Return stored tensor data as initializer, fallback to zeros if dat is None (shouldn't be None at load if properly saved)
        if self.dat is None:
            # Defensive fallback: create zeros if no data is provided (not recommended)
            return tf.zeros(shape, dtype=dtype)
        else:
            # Convert dat to tensor of requested dtype (likely float32)
            return tf.convert_to_tensor(self.dat, dtype=dtype, name=self.name)

    def get_config(self):
        # Serialize dat as a list or numpy array to allow saving/loading
        # Note: This assumes self.dat is a numpy array or convertible to list
        dat_serializable = None
        try:
            dat_serializable = self.dat.tolist()
        except Exception:
            try:
                dat_serializable = list(self.dat)
            except Exception:
                dat_serializable = None  # fallback
        return {"dat": dat_serializable, "name": self.name}

    @classmethod
    def from_config(cls, config):
        dat = config.get("dat", None)
        if dat is not None:
            dat = tf.convert_to_tensor(dat, dtype=tf.float32)
        return cls(dat=dat, name=config.get("name"))


class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assume a simple dense model to illustrate usage of custom initializer WB_Init
        # Input shape unknown: we assume input vectors of size 10 for example purposes
        
        # Create some example predefined weights for the initializer
        # In a real scenario, these would be passed in or loaded from saved data
        predefined_weights = tf.random.uniform((10, 20), dtype=tf.float32)
        predefined_bias = tf.random.uniform((20,), dtype=tf.float32)
        
        self.dense1 = tf.keras.layers.Dense(
            units=20,
            kernel_initializer=WB_Init(predefined_weights, name="kernel_init"),
            bias_initializer=WB_Init(predefined_bias, name="bias_init"),
            activation='relu'
        )
        # Output layer with default initialization
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Normally, you would compile and/or load weights here if available.
    # For demonstration, we just return the model instance.
    return model

def GetInput():
    # Return a random tensor input matching the input expected by MyModel
    # Based on MyModel, input shape should be (?, 10)
    batch_size = 4  # arbitrary batch size for example
    feature_dim = 10
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

