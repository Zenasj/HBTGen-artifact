import tensorflow as tf
from tensorflow.keras import layers

class Model3(keras.Model):
    def __init__(self):
        super().__init__()

        self.fc1 = keras.layers.Dense(8)
        self.fc2 = keras.layers.Dense(1)

    def build(self, input_shape):
        self.fc1.build((None, sum([shape[1] for shape in input_shape])))
        self.fc2.build((None, 8))

    @tf.function
    def call(self, inputs, **kwargs):
        concat_inputs = keras.ops.concatenate(inputs, axis=1)
        out1 = self.fc1(concat_inputs)
        out2 = self.fc2(out1)
        return out2


model3 = Model3()
model3.build([(None, 8), (None, 8)])
tf.saved_model.save(
    model3,
    f"{path}/model3",
    signatures=model3.call.get_concrete_function(
        (
            tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
        )
    ),
)

loaded3 = tf.saved_model.load(f"{path}/model3")
print(f"loaded3: {loaded3.signatures}")

# got:
# loaded3: _SignatureMap({'serving_default': <ConcreteFunction (*, inputs: TensorSpec(shape=(None, 8), dtype=tf.float32, name='inputs'), inputs_1: TensorSpec(shape=(None, 8), dtype=tf.float32, name='inputs_1')) -> Dict[['output_0', TensorSpec(shape=(None, 1), dtype=tf.float32, name='output_0')]] at 0x7F2CB1FF4190>})

# for all loaded1, loaded2, loaded3
print(loaded1.trainable_variables)
# got: AttributeError: '_UserObject' object has no attribute 'trainable_variables'.

print(loaded1.__call__)
# got: AttributeError: '_UserObject' object has no attribute '__call__'.

print(loaded3.call((tf.zeros((1, 8)), tf.zeros((1, 8)))))

outputs = restored_model.signatures["serving_default"](inputs=x, inputs_1=y)["output_0"]