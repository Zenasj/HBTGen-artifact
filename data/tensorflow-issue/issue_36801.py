# tf.random.uniform((B=100, H=1), dtype=tf.float32) ← inferred from example input shape (100,1)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple sequential model structure from the issue
        self.dense1 = tf.keras.layers.Dense(8, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
        # Instantiate optimizer (Adam) as in original example
        self.optimizer = tf.keras.optimizers.Adam()
        # Build model with input shape (None, 1)
        self._built = False

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

    def build_model(self, input_shape):
        # Build weights for Dense layers to enable optimizer state creation on first fit
        x = tf.keras.Input(shape=input_shape[1:])
        self.call(x)
        self._built = True

    def get_optimizer_config_json_serializable(self):
        """
        Return the optimizer configuration dictionary with all float32/numpy.float32 values 
        converted to standard Python floats to ensure JSON serializability.
        This addresses the reported problem where numpy.float32 types inside the config 
        cause json.dumps() to throw TypeError.
        """
        config = self.optimizer.get_config()

        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_floats(v) for v in obj]
            # Convert numpy.float32, tf.float32, or np.float64 to native Python float
            elif isinstance(obj, (float, int)):
                return obj
            elif hasattr(obj, 'dtype'):
                # Might be a numpy or tf dtype float scalar
                try:
                    return float(obj)
                except Exception:
                    return obj
            elif isinstance(obj, (tf.dtypes.DType,)):
                return str(obj)
            else:
                return obj
        
        cleaned_config = convert_floats(config)
        return cleaned_config

def my_model_function():
    model = MyModel()
    # Build model to initialize weights (optional, but recommended for cleaner usage)
    model.build_model((None, 1))
    return model

def GetInput():
    # Return random input tensor consistent with input shape used (batch size 100, 1 feature)
    # We use float32 dtype as typical for keras models and training.
    return tf.random.uniform((100, 1), dtype=tf.float32)

