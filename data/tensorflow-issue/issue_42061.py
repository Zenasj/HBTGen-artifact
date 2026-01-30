import tensorflow as tf

tf.compat.v1.disable_eager_execution()
version = 1
name = 'tmp_model'
export_path = f'/opt/tf_serving/{name}/{version}'
builder = saved_model_builder.SavedModelBuilder(export_path)

model_signature = tf.compat.v1.saved_model.predict_signature_def(
    inputs={
        'input': model.input
    }, 
    outputs={
        'output': model.output
    }
)

with tf.compat.v1.keras.backend.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict': model_signature
        },
        # For initializing Hashtables
        main_op=tf.compat.v1.tables_initializer()
    )
    builder.save()

class MyModule(tf.Module):

    def __init__(self, model):
        super(MyModule, self).__init__()
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 16), dtype=tf.int32, name='input')])
    def predict(self, input):
        result = self.model(input)
        return {"output": result}

version = 1
name = 'tmp_model'
export_path = f'/opt/tf_serving/{name}/{version}'

module = MyModule(model)
tf.saved_model.save(module, export_path, signatures={"predict": module.predict.get_concrete_function()})