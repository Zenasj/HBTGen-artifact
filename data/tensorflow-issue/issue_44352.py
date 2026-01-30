import tensorflow as tf  
from tensorflow.keras import layers  

class Evaluation:

    def __init__(self, strategy=None):  
        
        # prepare for encoded img data
        self.strategy = strategy
        H, W, C = 10, 10, 3
        imgs = tf.cast(tf.zeros([8, H, W, C]), tf.uint8)
        encodes = []
        for img in imgs:
            encode = tf.io.encode_jpeg(img)
            encodes.append(encode)
        encodes = tf.stack(encodes, axis = 0) 
        
        # convert encoded img data to tf.data
        self.dataset = tf.data.Dataset.from_tensor_slices(encodes)
        self.dataset = self.dataset.batch(2)
        self.dataset = self.strategy.experimental_distribute_dataset(self.dataset)
        with self.strategy.scope():
            self.conv = layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same')
        self.parallel_iterations = 10
    
    def preprocess(self, encoded):
        # preprocess for tf.data
        image = tf.io.decode_jpeg(encoded, channels=3)
        image = tf.image.resize(image, [20,20])
        return image

    @tf.function
    def serving(self, inputs):
        # data preprocess
        image = tf.map_fn(self.preprocess,
                          inputs,
                          fn_output_signature=tf.float32,
                          parallel_iterations=self.parallel_iterations)
        
        # inference for each batch
        prediction = self.conv(image)
        return prediction

    @tf.function
    def infer(self, serve_summary_writer):
        # inference for all batches
        with serve_summary_writer.as_default():
            batch = tf.cast(0, tf.int64)
            for data in self.dataset:
                prediction_perReplica = strategy.run(self.serving, args=(data,))
                prediction_tensor = prediction_perReplica.values
                prediction_concat = tf.concat(prediction_tensor, axis = 0)
                tf.summary.write(tag="prediction", tensor=prediction_concat, step=batch)
                batch += 1
                
    def eval(self):
        serve_summary_writer = tf.summary.create_file_writer('save_file', max_queue=100000, flush_millis=100000)
        self.infer(serve_summary_writer)
        serve_summary_writer.close()
        tf.io.gfile.rmtree('save_file')  

if __name__ == "__main__":

    strategy = tf.distribute.MirroredStrategy()
    e = Evaluation(strategy)   
    e.eval()