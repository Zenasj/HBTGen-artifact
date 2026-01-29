# tf.random.uniform((BATCH_SIZE, 224, 224, 3), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Fuse two models shown in the issue:
        # 1) Keras style MobileNetV2 model
        # 2) Custom training loop style ResNet50-based model
        
        # Keras API style model - MobileNetV2 (224x224 RGB input)
        self.mobilenet = tf.keras.applications.MobileNetV2(weights=None, input_shape=(224,224,3), classes=1000)
        
        # Custom training loop style model - ResNet50 backbone + GAP + dense output
        base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3),
                                                    weights=None,
                                                    include_top=False)
        self.resnet_model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1000, activation='softmax')
        ])
        
        # Weights and training details are not specified - models are left untrained
        # The outputs of both models will be compared with a tolerance to reflect
        # the kind of comparison discussed (e.g., differences in training time or behavior)

        # For this fused model, output the boolean tensor marking if outputs are numerically close
        self.tolerance = 1e-3  # assumed tolerance for output comparison
    
    def call(self, inputs, training=False):
        # Forward pass through both submodels
        out_mobilenet = self.mobilenet(inputs, training=training)
        out_resnet = self.resnet_model(inputs, training=training)
        
        # Compute elementwise absolute difference
        diff = tf.abs(out_mobilenet - out_resnet)

        # Compare outputs within tolerance
        comparison = tf.less_equal(diff, self.tolerance)

        # Return boolean tensor showing where predictions match closely
        # This simulates the comparison idea mentioned in the issue description:
        # evaluating differences due to training style or dataset/model choice
        return comparison

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a valid random input tensor for this model
    # Input shape: (batch_size, 224, 224, 3), float32 dtype, values scaled [0,1]
    # Use batch size 8 as a realistic default batch size used in examples
    batch_size = 8
    return tf.random.uniform((batch_size, 224, 224, 3), dtype=tf.float32, minval=0, maxval=1)

