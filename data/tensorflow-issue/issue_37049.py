import tensorflow as tf


class MyModule(tf.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        tf.saved_model.save(
            self,
            "/tmp/foomodel/001",
            signatures={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.some_method,
            },
        )

    @tf.function(input_signature=[
        tf.TensorSpec((None, ), dtype=tf.int64),
        tf.RaggedTensorSpec((None, None), dtype=tf.string),
    ])
    def some_method(self, dense, ragged):
        return tf.constant(["foobar"], dtype=tf.string)

m = MyModule()
dense = tf.constant([[1,2]], dtype=tf.int64)
ragged = tf.ragged.constant([["foo"], ["foo", "bar"]], dtype=tf.string)
some_result = m.some_method(dense, ragged)
print(f"some_result => {some_result}")

#ValueError: Python inputs incompatible with input_signature:
# inputs: (
#    tf.Tensor([[1 2]], shape=(1, 2), dtype=int64),
#    <tf.RaggedTensor [[b'foo'], [b'foo', b'bar']]>)
#  input_signature: (
#    TensorSpec(shape=(None,), dtype=tf.int64, name=None),
#    RaggedTensorSpec(TensorShape([None, None]), tf.string, 1, tf.int64))

loaded = tf.saved_model.load("/tmp/foomodel/001")
print(loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]._arg_keywords)
# ['ragged', 'ragged_1']

# Or inspect with saved_model_cli:
# !saved_model_cli show --dir /tmp/foomodel/001 --all
#
# MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
# 
# signature_def['__saved_model_init_op']:
#   The given SavedModel SignatureDef contains the following input(s):
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['__saved_model_init_op'] tensor_info:
#         dtype: DT_INVALID
#         shape: unknown_rank
#         name: NoOp
#   Method name is: 
# 
# signature_def['serving_default']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['ragged'] tensor_info:
#         dtype: DT_STRING
#         shape: (-1)
#         name: serving_default_ragged:0
#     inputs['ragged_1'] tensor_info:
#         dtype: DT_INT64
#         shape: (-1)
#         name: serving_default_ragged_1:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['output_0'] tensor_info:
#         dtype: DT_STRING
#         shape: (1)
#         name: PartitionedCall:0
#   Method name is: tensorflow/serving/predict
# 
# Defined Functions:
#   Function Name: 'some_method'
#     Option #1
#       Callable with:
#         Argument #1
#           DType: RaggedTensorSpec
#           Value: RaggedTensorSpec(TensorShape([None, None]), tf.string, 1, tf.int64)