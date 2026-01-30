# import necessary modules
import tensorflow as tf
import tensorflow.keras as tk
print(tf.__version__)

# define a custom model
class MyModel(tk.Model):
    ...

# Define a simple sequential model
def create_model():
    a = tk.Input(shape=(32,))
    b = tk.layers.Dense(32)(a)
    model = MyModel(inputs=a, outputs=b)

    model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model


# create model
my_model = create_model()

# Display the model's architecture
my_model.summary()

# save model
my_model.save(filepath="./saved_model", save_format="tf")

# load back model
my_model.load_weights(filepath="./saved_model")

my_model.load_weights(filepath="./saved_model/variables/variables")
print(my_model.__class__)

_loaded_my_model = tk.models.load_model("./saved_model")
print(_loaded_my_model.__class__)

# load back model
# .... this does not work
my_model.load_weights(filepath="saved_model")