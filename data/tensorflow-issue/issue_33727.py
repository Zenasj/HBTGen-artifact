# tf.random.uniform((1, 3, 3, 2), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.resnet_depth = 96
        self.board_x = 3
        self.board_y = 3
        self.depth_dim = 2
        self.action_size = 10
        self.num_chan = 4
        
        # Initial convolution + BN + ReLU
        self.conv0 = tf.keras.layers.Conv2D(self.num_chan, 1, padding='same', use_bias=False)
        self.bn0 = tf.keras.layers.BatchNormalization(axis=3)
        self.act0 = tf.keras.layers.Activation('relu')
        
        # ResNet blocks with Conv-BN-ReLU repeated resnet_depth times
        self.res_blocks_conv = [
            tf.keras.layers.Conv2D(self.num_chan, 1, padding='same', use_bias=False) 
            for _ in range(self.resnet_depth)
        ]
        self.res_blocks_bn = [
            tf.keras.layers.BatchNormalization(axis=3) 
            for _ in range(self.resnet_depth)
        ]
        self.res_blocks_act = [tf.keras.layers.Activation('relu') for _ in range(self.resnet_depth)]
        
        # Flatten
        self.flatten = tf.keras.layers.Flatten()
        
        # Dense + BN + ReLU + Dropout
        self.dense1 = tf.keras.layers.Dense(16, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
        self.act1 = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        
        # Output layer with softmax
        self.pi = tf.keras.layers.Dense(self.action_size, activation='softmax', name='pi')

    def call(self, inputs, training=False):
        x = self.conv0(inputs)
        x = self.bn0(x, training=training)
        x = self.act0(x)
        
        for conv_layer, bn_layer, act_layer in zip(self.res_blocks_conv, self.res_blocks_bn, self.res_blocks_act):
            x = conv_layer(x)
            x = bn_layer(x, training=training)
            x = act_layer(x)
            
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout(x, training=training)
        output = self.pi(x)
        return output

def my_model_function():
    model = MyModel()
    # Compile model as per original snippet
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(0.001)
    )
    return model

def GetInput():
    # Input shape matches Input(shape=(3,3,2)) from original code
    return tf.random.uniform((1, 3, 3, 2), dtype=tf.float32)

