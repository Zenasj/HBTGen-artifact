import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.framework import config

hvd.init()
class TestModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[2], dtype=tf.float32)],autograph=False)
    def gradient(self, x):
        all_reduced = hvd.allreduce(x, op=hvd.Average)
        dummy = tf.add(all_reduced, tf.constant(0.0, dtype=all_reduced.dtype))
        with tf.control_dependencies([dummy]):
            return tf.identity(dummy, name="final_output")

# 保存模型
model = TestModel()
gradient_concrete_function = model.gradient.get_concrete_function(tf.TensorSpec(shape=[2], dtype=tf.float32))
graph_def = gradient_concrete_function.graph.as_graph_def()
print("Graph nodes:", [n.name for n in graph_def.node])
#tf.saved_model.save(model, "test_model",signatures={"gradient" : gradient_concrete_function})
tf.saved_model.save(
    model,
    "test_model",
    options=tf.saved_model.SaveOptions(
        save_debug_info=True,
        experimental_io_device='/job:localhost'
        #disable_graph_optimization=True  # 关键修正
    ),
    signatures = {"gradient" : gradient_concrete_function}
)

## 1. 加载模型

loaded = tf.saved_model.load("test_model")
#
## 2. 获取计算图
concrete_func = loaded.signatures["gradient"]
graph_def = concrete_func.graph.as_graph_def()
print("Graph nodes:", [n.name for n in graph_def.node])

