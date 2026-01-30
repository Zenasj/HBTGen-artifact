from tensorflow import keras

import tensorflow as tf
from tensorflow.python.pywrap_mlir import import_graphdef

SIZE = 3

class MyModel(tf.keras.Model):
    def build(self, input_shape):
        self.w = self.add_weight(shape=(SIZE,), trainable=True)

    def call(self, input):
        self.w.assign_add(input)
        return input

if __name__ == "__main__":
    model = MyModel()

    func = tf.function(model)
    concrete_func = func.get_concrete_function(
        tf.TensorSpec(shape=(SIZE,), dtype=tf.float32)
    )
    graph = concrete_func.graph

    mlir_tf = import_graphdef(
        graph.as_graph_def(add_shapes=True),
        "tf-standard-pipeline",
        False,
        input_names=[t.name for t in graph.inputs],
        input_data_types=["DT_FLOAT", "DT_RESOURCE"],
        input_data_shapes=[",".join(str(d) for d in t.shape) for t in graph.inputs],
        output_names=[t.name for t in graph.outputs],
    )
    print(mlir_tf)

    with open("model.mlir", "w") as f:
        f.write(mlir_tf)

input_data_types=["DT_FLOAT", "DT_RESOURCE<tensor<3xf32>>"],

input_data_types=["DT_FLOAT", "DT_RESOURCE(3:DT_FLOAT)"],