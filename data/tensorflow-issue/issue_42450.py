import tensorflow as tf

class CustomModule(tf.Module):

  @tf.function
  def f(self, x, y):
    z = x * y   # potentially complex graph where I want to run the different grappler optimizers
    return z

module = CustomModule()

# 
#  here I apply graph optimization to `module`.  How can I implement run_grappler_optimizations_on_module?
#
module = run_grappler_optimizations_on_module(module)


# here I save it
tf.saved_model.save(module, '/my/model/directory')