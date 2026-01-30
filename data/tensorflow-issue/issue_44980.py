import tensorflow as tf

class Network:
    def call(self, x):
        self.weights = tf.zeros((4,5), dtype=tf.double)
        self.encoder_graph(x)
        self.weights = tf.zeros((2,3), dtype=tf.double)

        return (self.encoder_graph(x).shape == self.encoder_eager(x).shape)

    @tf.function
    def encoder_graph(self, x):
        return self.weights 
        
    def encoder_eager(self, x):
        return self.weights

N = Network()

tf.config.run_functions_eagerly(False)
print("Graph mode: ", N.call(tf.zeros((1,1))))
tf.config.run_functions_eagerly(True)
print("Eager mode: ", N.call(tf.zeros((1,1))))