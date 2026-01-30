import tensorflow as tf
from tensorflow.tools import graph_transforms

tf.enable_eager_execution()

# Define a simple model with an extra op
class MyTrackable(tf.train.Checkpoint):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32),
                                  tf.TensorSpec(shape=None, dtype=tf.int32)])
    def test_func(self, first_arg, second_arg):
        result1 = first_arg + second_arg
        result2 = first_arg - second_arg
        tf.constant(42., name="dead_code")
        return {"sum": result1, "difference": result2}
    
before_dir = "./model_before"
after_dir = "./model_after"

tf.saved_model.save(MyTrackable(), before_dir)

one = tf.constant(1, dtype=tf.int32)
two = tf.constant(2, dtype=tf.int32)
model_before = tf.saved_model.load_v2(before_dir)
print("Result before: {}".format(
    model_before.signatures["serving_default"](first_arg=two, second_arg=one)))
print("Contains dead code before: {}".format( 
    str("dead_code" in str(model_before.signatures["serving_default"].graph.as_graph_def()))))

graph_transforms.TransformSavedModel(before_dir, after_dir, ["strip_unused_nodes(type=float32)"])
model_after = tf.saved_model.load_v2(after_dir)
print("Result after: {}".format(
    model_after.signatures["serving_default"](first_arg=two, second_arg=one)))
print("Contains dead code after: {}".format( 
    str("dead_code" in str(model_after.signatures["serving_default"].graph.as_graph_def()))))