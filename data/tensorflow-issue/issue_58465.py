# tf.random.uniform((B,), dtype=object or tf.float32) ← Input is a dictionary with keys like 'convertedManaCost', 'is_creature', 'type', 'text', 'name', each input shape (None, 1)

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Helper function as defined in the discussion
def get_output_sequence_length(column):
    """Return the length of the longest sequence of split strings in the column."""
    lengths = [len(x.split()) for x in column]
    return max(lengths)

class PreprocessingModel(tf.keras.Model):
    def __init__(self, features):
        super().__init__()
        # Create input layers dictionary
        self.inputs_dict = {}
        for name, column in features.items():
            dtype = column.dtype
            if dtype == object:
                input_dtype = tf.string
            else:
                input_dtype = tf.float32
            # Each input is shape (None, 1)
            self.inputs_dict[name] = tf.keras.Input(shape=(1,), name=name, dtype=input_dtype)
        
        # 1. Numeric inputs concatenation + normalization
        numeric_inputs = {name: inp for name, inp in self.inputs_dict.items() if inp.dtype == tf.float32}
        self.concat_numeric = layers.Concatenate()
        self.norm = layers.Normalization()
        # As norm.adapt takes numpy array, we have to assemble numeric features from passed features dict (Pandas Dataframe)
        numeric_features_array = np.array(features[list(numeric_inputs.keys())])
        self.norm.adapt(numeric_features_array)
        
        # For text feature inputs, build TextVectorization layers on initialization
        self.text_vectorizers = {}
        self.text_output_lengths = {}
        self.text_names = []
        for name, inp in self.inputs_dict.items():
            if inp.dtype == tf.float32:
                continue
            self.text_names.append(name)
            seq_len = get_output_sequence_length(features[name])
            self.text_output_lengths[name] = seq_len
            tv_layer = layers.TextVectorization(output_sequence_length=seq_len)
            tv_layer.adapt(features[name])
            self.text_vectorizers[name] = tv_layer

    def call(self, inputs):
        # Inputs is a dict of tensors keyed by feature names
        # Step 1: Numeric inputs concatenation and normalization
        numeric_inputs = [inputs[name] for name in self.inputs_dict if self.inputs_dict[name].dtype == tf.float32]
        x_numeric = self.concat_numeric(numeric_inputs)
        x_numeric_norm = self.norm(x_numeric)
        
        preprocessed = [x_numeric_norm]
        
        # Step 2: Text inputs vectorization + cast to float32
        for name in self.text_names:
            x_text = self.text_vectorizers[name](inputs[name])  # output is int32 indices tensor
            x_text = tf.cast(x_text, tf.float32)
            preprocessed.append(x_text)
        
        return preprocessed

class MyModel(tf.keras.Model):
    def __init__(self, features):
        super().__init__()
        # Instantiate the preprocessing submodel
        self.preprocessing = PreprocessingModel(features)
        
        # Dense layers for final classification
        self.concat_all = layers.Concatenate()
        self.dense1 = layers.Dense(units=128, activation="relu")
        self.out_layer = layers.Dense(units=5, activation="sigmoid")
        
        # Prepare the functional inputs corresponding to preprocessing inputs
        # This is just for use with the functional style; not strictly required for subclassing
        self._inputs_dict = self.preprocessing.inputs_dict

    def call(self, inputs):
        # Preprocess inputs
        preprocessed_list = self.preprocessing(inputs)
        # Concatenate all processed inputs
        x = self.concat_all(preprocessed_list)
        # Dense + output
        x = self.dense1(x)
        return self.out_layer(x)

def my_model_function():
    """
    Builds model MyModel with given data.
    Since features are needed to initialize text vectorization layers and Normalization layer,
    we assume 'mtg_features' to be loaded before calling this function.
    For demonstration we mock mtg_features with minimal workable structure;
    in real usage replace mtg_features/data accordingly.
    """
    import pandas as pd
    
    # Placeholder feature data simulated to show the expected structure with sample data 
    # (as in the issue, features is a pandas DataFrame loaded from 'all_cards.csv')
    mtg_features = pd.DataFrame({
        "convertedManaCost": np.array([1.0, 2.0, 3.0], dtype=float),
        "is_creature": np.array([0.0, 1.0, 0.0], dtype=float),
        "type": ["creature", "sorcery", "instant"],
        "text": ["Some text spell", "Creature text", "Instant effect"],
        "name": ["Name1", "Name2", "Name3"]
    })
    
    model = MyModel(mtg_features)
    
    # Compile for usage similar to the issue
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["acc"]
    )
    return model

def GetInput():
    """
    Return a dictionary input matching the expected inputs of MyModel.
    We'll create a batch of 2 samples for demonstration.
    Numeric inputs are float32 tensors shape (batch,1).
    Text inputs are string tensors shape (batch,1).
    """
    batch_size = 2
    
    # Create sample values matching the mock feature data in `my_model_function`
    inputs = {
        "convertedManaCost": tf.random.uniform((batch_size, 1), minval=0, maxval=10, dtype=tf.float32),
        "is_creature": tf.random.uniform((batch_size, 1), minval=0, maxval=2, dtype=tf.float32),
        "type": tf.constant([["creature"], ["sorcery"]], dtype=tf.string),
        "text": tf.constant([["Fireball deals damage"], ["Summon creature"]], dtype=tf.string),
        "name": tf.constant([["Goblin"], ["Elf"]], dtype=tf.string),
    }
    return inputs

