import tensorflow as tf

class MyMetaArch(ModelDesc):

    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, None, 1), 'image'),
            tf.placeholder(tf.float32, (8, None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (8, None,), 'gt_labels')
        ]
        return ret

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        images = inputs["image"]

        x = Conv2D('conv1', images, filters=16, kernel_size=3, strides=(2, 2))
        x = Conv2D('conv2', x, filters=32, kernel_size=3, strides=(2, 2))
        x = Conv2D('conv3', x, filters=64, kernel_size=3, strides=(2, 2))
        x = Conv2D('conv4', x, filters=64, kernel_size=3, strides=(2, 2))
        ...
        class_prediction_with_background = tf.concat(...) # concat multi branches 
        class_prediction_with_background = class_prediction_with_background [:,:,1] # face detector, only care about the foreground

        gradients = tf.gradients(class_prediction_with_background , [images])[0]
        regular_vals = tf.reduce_mean(gradients)
        ...
        class_losses = class_losses + regular_vals