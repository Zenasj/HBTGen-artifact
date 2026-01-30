import tensorflow as tf

#Make GradCAM heatmap following the Keras tutorial.
last_conv_layer = model.layers[-4].layers[-1]
last_conv_layer_model = keras.Model(model.layers[-4].inputs, last_conv_layer.output)

# Second, we create a model that maps the activations of the last conv
# layer to the final class predictions
classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer in model.layers[-3:]:
    x = layer(x)
classifier_model = keras.Model(classifier_input, x)

#Preparing the image with the preprocessing layers
preprocess_layers = keras.Model(model.inputs, model.layers[-5].output)
img_array = preprocess_layers(prepared_image)

# Then, we compute the gradient of the top predicted class for our input image
# with respect to the activations of the last conv layer
with tf.GradientTape() as tape:
    # Compute activations of the last conv layer and make the tape watch it
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    # Compute class predictions
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

# This is the gradient of the top predicted class with regard to
# the output feature map of the last conv layer
grads = tape.gradient(top_class_channel, last_conv_layer_output)