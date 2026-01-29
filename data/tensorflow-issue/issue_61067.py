# tf.random.uniform((B, 126), dtype=tf.float32) ← Assumed input shape based on the TFInferModel input_signature

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # The original code loads two separate saved_models:
        # feat_gen and model, each expecting and producing tensors.
        # Here we encapsulate them as submodules, assuming they are TF SavedModels in the form of tf.functions.
        
        # For the purpose of this code, since we don't have the actual "feat_gen.pb" and "model.pb",
        # we create placeholders where these submodules would be loaded.
        # In the real scenario, you would replace these with:
        # self.feat_gen = tf.saved_model.load("feat_gen.pb")
        # self.model = tf.saved_model.load("model.pb")
        
        # Placeholder layers simulating feat_gen and model. Their behavior should mirror the original.
        # Using simple dense layers with fixed weights is just for shape and flow demonstration.
        self.feat_gen = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, 126)),  # input adds batch dim
            tf.keras.layers.Dense(64, activation='relu', trainable=False, name="feat_gen_dense"),
        ])
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(64,)),  # input after feat_gen
            tf.keras.layers.Dense(10, activation=None, trainable=False, name="model_dense"),
        ])

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 126), dtype=tf.float32, name="inputs")])
    def call(self, inputs):
        # inputs shape: (batch_size, 126)
        # The original code adds a batch dimension: inputs = inputs[None]
        # This adds a leading batch dim making shape (1, batch_size, 126).
        # But since inputs already have batch dimension, we interpret the original as:
        # The TFInferModel expects a single vector of shape (126,), so inputs shape (None, 126) means batch.
        # To mimic the original, let's assume the batch dimension is batch_size and simulate the logic:
        
        # So "inputs = inputs[None]" turns (batch_size,126) -> (1,batch_size,126)
        # Since our placeholder layers expect last dimension 126 or 64, we will reshape accordingly.

        # Add batch dimension as in original: new shape (1, batch_size, 126)
        shaped_inputs = tf.expand_dims(inputs, axis=0)
        # Pass through feat_gen; outputs with shape (1, batch_size, 64)
        features = self.feat_gen(shaped_inputs)
        # Remove first batch dimension of feat_gen output for model input: shape (batch_size, 64)
        features = tf.squeeze(features, axis=0)
        # Pass through model; outputs with shape (batch_size, 10)
        outputs = self.model(features)
        # The original code removes batch dimension from outputs as outputs = outputs[0]
        # But since original batch dimension was added at front (1, batch_size, ...)
        # outputs shape (batch_size, 10)
        # So outputs[0] selects outputs of first sample, shape (10,)
        final_output = outputs[0]
        
        return {"outputs": final_output}


def my_model_function():
    # Return initialized MyModel instance
    return MyModel()


def GetInput():
    # Return a random tensor input with shape (126,), matching the input expected by MyModel.call after accounting for batch dim.
    # The original input_signature is (None, 126), but the call uses inputs[None], so to run single input, use (126,)
    # Since call signature expects (batch_size, 126), we provide batch_size=1 compatible input as (1,126), 
    # but in the call function we expand dims again, so let's provide (126,) to replicate typical use.
    return tf.random.uniform((126,), dtype=tf.float32)

