import tensorflow as tf
from tensorflow.keras import models

class WebServer(threading.Thread):
    def run(self):
        application.listen(8888)
        #asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.set_event_loop(AnyThreadEventLoopPolicy())
        tornado.ioloop.IOLoop.instance().start()
WebServer.start()

application.listen(8888)
application.listen(8888)
tornado.ioloop.IOLoop.instance().start()

def load_tensorflow_shared_session(self):
        """ Load a Tensorflow/Keras shared session """
        # LP: create a config by gpu cpu backend
        if os.getenv('HAS_GPU', '0') == '1':
            config = tf.ConfigProto(
                device_count={'GPU': 1},
                intra_op_parallelism_threads=1,
                allow_soft_placement=True
            )
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
        else:
            config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                allow_soft_placement=True
            )
        # LP: create session by config
        self.tf_session = tf.Session(config=config)

        return self.tf_session

def load__model(MODEL_PATH, session):
    '''
        Load the model
    '''

    # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
    # Otherwise, their weights will be unavailable in the threads after the session there has been set
    set_session(session)
    model = keras.models.load_model(MODEL_PATH)
    model._make_predict_function()