# tf.random.uniform((BATCH_SIZE, T, D), dtype=tf.float32) 
# Assuming typical input shape: batch_size x timesteps x features for LSTM
# Here BATCH_SIZE, T, D are placeholders for batch size, time steps, and feature dims respectively.
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, 
                 lstm_units_list=None, 
                 lstm_dropout_list=None, 
                 extra_dense_units=None,
                 extra_dropout_rate=None,
                 use_extra_dense=False,
                 use_extra_dropout=False,
                 input_shape=None):
        super().__init__()
        # Validate or set defaults
        if lstm_units_list is None:
            # Example default config: 3 LSTM layers + final closing layer
            lstm_units_list = [64, 64, 64, 64]
        if lstm_dropout_list is None:
            lstm_dropout_list = [0.0] * len(lstm_units_list)
        assert len(lstm_units_list) == len(lstm_dropout_list), \
            "Length of units and dropout lists must match"

        # Store layers
        self.lstm_layers = []
        for i, (units, dropout_rate) in enumerate(zip(lstm_units_list[:-1], lstm_dropout_list[:-1])):
            # All except last LSTM return sequences
            self.lstm_layers.append(
                tf.keras.layers.LSTM(units=units, 
                                     dropout=dropout_rate,
                                     return_sequences=True,
                                     input_shape=input_shape if i == 0 else None)
            )
        # Last LSTM layer returns last output (return_sequences=False)
        last_units = lstm_units_list[-1]
        last_dropout = lstm_dropout_list[-1]
        self.lstm_layers.append(
            tf.keras.layers.LSTM(units=last_units,
                                 dropout=last_dropout,
                                 return_sequences=False)
        )

        self.use_extra_dense = use_extra_dense
        if self.use_extra_dense:
            # Default units if not provided
            dense_units = extra_dense_units if extra_dense_units is not None else 64
            self.extra_dense = tf.keras.layers.Dense(units=dense_units)
        else:
            self.extra_dense = None

        self.use_extra_dropout = use_extra_dropout
        if self.use_extra_dropout:
            # The original snippet suggested a Dense layer with units=float dropout prob which seems like an error
            # Instead, we'll interpret this as an actual Dropout layer with given dropout rate
            dropout_rate = extra_dropout_rate if extra_dropout_rate is not None else 0.3
            self.extra_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        else:
            self.extra_dropout = None

        # Final output layer: 1 unit (regression or scalar output)
        self.output_dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x, training=training)
        if self.use_extra_dense:
            x = self.extra_dense(x)
        if self.use_extra_dropout:
            x = self.extra_dropout(x, training=training)
        return self.output_dense(x)


def my_model_function():
    # For demonstration purposes, use example hyperparameters similar to the reported tuning space
    # Because no 'hp' (hyperparameters) object is available here, use fixed sensible values.

    # Assumptions based on the issue:
    # - Input shape: (BATCH_SIZE, time_steps, features)
    # - Use 3 LSTM layers + closing LSTM layer
    # - Units and dropout rates chosen arbitrarily within ranges from issue
    lstm_units = [64, 64, 64, 64]
    lstm_dropouts = [0.2, 0.1, 0.1, 0.0]

    # Enable extra dense and dropout to mimic full model
    extra_dense_units = 32
    extra_dropout_rate = 0.3

    # Input shape without batch size, inference from typical time series of 100 timesteps and 10 features
    input_shape = (100, 10)

    model = MyModel(
        lstm_units_list=lstm_units,
        lstm_dropout_list=lstm_dropouts,
        extra_dense_units=extra_dense_units,
        extra_dropout_rate=extra_dropout_rate,
        use_extra_dense=True,
        use_extra_dropout=True,
        input_shape=input_shape
    )

    # Compile the model to mimic original compilation in user code
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

    return model


def GetInput():
    # Return a random tensor shaped for the model input: (batch_size, time_steps, features)
    # Batch size inferred from typical use case in the original issue (commonly 32 or configurable)
    batch_size = 32
    time_steps = 100
    features = 10
    return tf.random.uniform((batch_size, time_steps, features), dtype=tf.float32)

