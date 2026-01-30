import numpy as np
import tensorflow as tf
from tensorflow import keras

def vgg16_model_fn(features, mode, params):
    
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    
    with tf.variable_scope('vgg_base'):
        # Use a pre-trained VGG16 model and drop off the top layers as we will retrain 
        # with our own dense output for our custom classes
        vgg16_base = tf.keras.applications.VGG16(
            include_top=False,
            input_shape=(224, 224, 3),
            input_tensor=features['image'],
            pooling='avg')

        # Disable training for all layers to increase speed for transfer learning
        # If new classes significantely different from ImageNet, this may be worth leaving as trainable = True
        for layer in vgg16_base.layers:
            layer.trainable = False

        x = vgg16_base.output
    
    with tf.variable_scope("fc"):
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=4096, activation=tf.nn.relu, trainable=is_training, name='fc1')
        x = tf.layers.dense(x, units=4096, activation=tf.nn.relu, trainable=is_training, name='fc2')
        x = tf.layers.dropout(x, rate=0.5, training=is_training)
        
    # Finally add a 2 dense layer for class predictions
    with tf.variable_scope("Prediction"):
        x = tf.layers.dense(x, units=NUM_CLASSES, trainable=is_training)
        return x

dog_cat_estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params=params
)
train_spec = tf.estimator.TrainSpec(
    input_fn=data_input_fn(train_record_filenames, num_epochs=None, batch_size=10, shuffle=True), 
    max_steps=10)
eval_spec = tf.estimator.EvalSpec(
    input_fn=data_input_fn(validation_record_filenames)
)
tf.estimator.train_and_evaluate(dog_cat_estimator, train_spec, eval_spec)

# load_keras_model.py
class LoadKerasModel:
    model = None
    graph = None

    def __init__(self):
        self.keras_resource()
        self.init_model()

    def init_model(self):
        self.graph = tf.get_default_graph()
        self.model = load_model(file_path)
        self.model.predict(np.ones((1, 1, 1, 1)))

    def keras_resource(self):
        num_cores = 4

        if os.getenv('TENSORFLOW_VERSION') == 'GPU':
            num_gpu = 1
            num_cpu = 1
        elif os.getenv('TENSORFLOW_VERSION') == 'CPU':
            num_gpu = 0
            num_cpu = 1
        else:
            raise NonResourceException()

        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                                device_count={'CPU': num_cpu, 'GPU': num_gpu})
        config.gpu_options.allow_growth = True

        session = tf.Session(config=config)
        K.set_session(session)

    def predict_target(selfl, img_generator):
        with self.graph.as_default():
            predict = self.model.predict_generator(
                img_generator,
                steps=len(img_generator),
                verbose=1
            )
        return predict

load_keras_model = LoadKerasModel()