import numpy as np

for node in graph.node:
    if "ReduceMax" in node.op_type:
        for index in range(len(node.attribute)):
            if node.attribute[index].name == "axes":
                del node.attribute[index]
                axes_input = onnx.helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1])
                axes_value = numpy_helper.from_array(np.array([1]), "axes")
                onnx_model.graph.input.extend([axes_input])
                onnx_model.graph.initializer.extend([axes_value])
                node.input.append("axes")
                break