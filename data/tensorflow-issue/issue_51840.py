# tf.random.uniform((B, 4), dtype=tf.float32) ← Input shape is (batch_size, 4) as from the example
import tensorflow as tf

class WeightedSum(tf.keras.layers.Layer):
    def __init__(self, n_models=2, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.n_models = n_models
        self.ensemble_weights = []
        self.output_init = tf.Variable(0., validate_shape=False, trainable=False)

    def build(self, input_shape):
        for i in range(self.n_models):
            # Adding explicit name parameter to add_weight to avoid saving issues
            self.ensemble_weights.append(
                self.add_weight(
                    name=f'ensemble_weight_{i}',
                    shape=(1,),
                    initializer='ones',
                    trainable=True
                )
            )

    def call(self, inputs):
        # Normalize weights so they sum to 1
        new_normalizer = tf.convert_to_tensor(0., dtype=inputs[0].dtype)
        for i in range(self.n_models):
            new_normalizer = new_normalizer + self.ensemble_weights[i]
        new_normalizer = tf.constant(1., dtype=new_normalizer.dtype) / new_normalizer
        
        output = tf.cast(self.output_init, dtype=inputs[0].dtype)
        for i in range(self.n_models):
            # Cast ensemble weight to input dtype for compatibility
            output = tf.add(
                output,
                tf.multiply(tf.cast(self.ensemble_weights[i], dtype=inputs[i].dtype), inputs[i])
            )
        output = tf.multiply(output, new_normalizer)
        return output


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Low fidelity submodel
        self.lf_dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.lf_dense2 = tf.keras.layers.Dense(10, activation='relu')
        self.lf_dense3 = tf.keras.layers.Dense(10, activation='relu')

        # Two high fidelity submodels (linear and nonlinear)
        # Note: Based on original code hf_lin_mod and hf_nonlin_mod took 14-dim input (concatenated low fidelity output + input)
        self.hf_lin_dense1 = tf.keras.layers.Dense(10)
        self.hf_lin_dense2 = tf.keras.layers.Dense(10)
        self.hf_lin_dense3 = tf.keras.layers.Dense(10, activation='relu')

        self.hf_nonlin_dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.hf_nonlin_dense2 = tf.keras.layers.Dense(10, activation='relu')
        self.hf_nonlin_dense3 = tf.keras.layers.Dense(10, activation='relu')

        # WeightedSum layer with 2 inputs
        self.weighted_sum = WeightedSum(n_models=2)

        # Concatenate layer
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        """
        inputs: tf.Tensor of shape (batch_size, 4)
        """

        # Low fidelity output
        x = self.lf_dense1(inputs)
        x = self.lf_dense2(x)
        lf_out = self.lf_dense3(x)

        # Concatenate low fidelity output and original input along last axis
        concat_input = self.concat([lf_out, inputs])

        # High fidelity linear branch
        lin = self.hf_lin_dense1(concat_input)
        lin = self.hf_lin_dense2(lin)
        lin = self.hf_lin_dense3(lin)

        # High fidelity nonlinear branch
        nonlin = self.hf_nonlin_dense1(concat_input)
        nonlin = self.hf_nonlin_dense2(nonlin)
        nonlin = self.hf_nonlin_dense3(nonlin)

        # Weighted sum of linear and nonlinear outputs
        hf_out = self.weighted_sum([lin, nonlin])

        # Return dictionary matching original outputs: low fidelity and high fidelity
        return {'low_fidelity': lf_out, 'high_fidelity': hf_out}


def my_model_function():
    # Return an instance of MyModel (weights initialized randomly)
    model = MyModel()
    # It's useful to call the model once with dummy data to build weights and shapes
    dummy_input = tf.zeros((1,4), dtype=tf.float32)
    _ = model(dummy_input)
    return model

def GetInput():
    # Generate a random input tensor matching the expected input shape: (batch_size, 4)
    # Batch size is arbitrary, here we choose 8
    return tf.random.uniform((8,4), dtype=tf.float32)

