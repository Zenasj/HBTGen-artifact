import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# DATA
BUFFER_SIZE = 512 # Isn't taken into account
BATCH_SIZE = 256 # Number of training examples in 1 forward/backward pass. 
                 # Higher batch size, more memory space needed

# AUGMENTATION
IMAGE_SIZE = 256 
PATCH_SIZE = 32 # CHANGE HERE TO CHANGE THE NUMBER OF PATCHES 256/16=16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2 # 64

# OPTIMIZER
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# TRAINING
EPOCHS = 50 # 1 epoch = 1 forward and 1 backward pass of all the training examples

# ARCHITECTURE
LAYER_NORM_EPS = 1e-6
TRANSFORMER_LAYERS = 8
PROJECTION_DIM = 32 # https://www.tensorflow.org/recommenders/examples/dcn , messo a 64 prima
NUM_HEADS = 4
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]
MLP_HEAD_UNITS = [2048, 1024]


data_augmentation = keras.Sequential(
    [
        layers.Normalization(), #if using a new version of tf
        #tf.keras.layers.experimental.preprocessing.Normalization, #if using an old version of tf
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_tr)

class ShiftedPatchTokenization(layers.Layer):
    def __init__(
      self,
      image_size=IMAGE_SIZE,
      patch_size=PATCH_SIZE,
      half_patch=PATCH_SIZE//2,
      num_patches=NUM_PATCHES,
      projection_dim=PROJECTION_DIM,
      flatten_patches=None,
      projection=None,
      layer_norm=None,
      vanilla=False,
      **kwargs,
    ):
      super(ShiftedPatchTokenization,self).__init__(**kwargs)
      self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor
      self.image_size = image_size
      self.patch_size = patch_size
      self.half_patch = patch_size // 2 # la divisione con // dà il numero in int()
      self.flatten_patches = layers.Reshape((num_patches, -1))
      self.projection = layers.Dense(units=projection_dim)
      self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)

    # Override function to avoid error while saving model
    def get_config(self):
      config = super().get_config().copy()
      config.update(
          {
          "image_size": self.image_size,
          "patch_size": self.patch_size,
          "half_patch": self.half_patch,
          "flatten_patches": self.flatten_patches,
          "vanilla": self.vanilla,
          "projection": self.projection,
          "layer_norm": self.layer_norm,
          }
      )
      return config
     


    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return (tokens, patches)

# PATCH ENCODING LAYER, accepts projected patches and then adds positional information to them

class PatchEncoder(layers.Layer):
    def __init__(
        self,
        num_patches=NUM_PATCHES,
        projection_dim=PROJECTION_DIM,
        position_embedding=None,
        positions=None,
        **kwargs
    ):
        super(PatchEncoder,self).__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def get_config(self):
      config = super().get_config().copy()
      config.update({
          "num_patches": self.num_patches,
          "position_embedding": self.position_embedding,
          "positions": self.positions.numpy(),
      })
      return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches

# LOCALITY SELF ATTENTION

# ho aggiunto elementi dove si trovano i commenti 'modificato'

class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(
        self,
        tau=None, #modificato, prima non c'era
        **kwargs
    ):
        super(MultiHeadAttentionLSA,self).__init__(**kwargs)
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True) # The trainable temperature term. The initial value is the square root of the key dimension.

    def get_config(self):
      config = super().get_config().copy()
      config.update({
          "tau": self.tau.numpy(), #modificato, prima era solo self.tau
      })
      return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask) 
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

## PECRHè SI USA LA GELU piuttosto che altre?
# RISOLTO IN PARTE: si utilizza perche da risultati migliori in Computer Vision ed NLP

# Multi layer perceptron
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x) # A Tensor with the same type as features where features are a Tensor representing preactivation values https://www.tensorflow.org/api_docs/python/tf/nn/gelu 
        # The GELU is the standard Gaussian cumulative distribution function
        x = layers.Dropout(dropout_rate)(x)
    return x


# Build the diagonal attention mask
diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

def create_vit_classifier(vanilla=False):
    inputs = layers.Input(shape=(256,256,3))
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(vanilla=vanilla)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder()(tokens)

    # Create multiple layers of the Transformer block.
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        if not vanilla: # VANILLA TRUE
            attention_output = MultiHeadAttentionLSA(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else: # VANILLA FALSE
            attention_output = layers.MultiHeadAttention(
                num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
            )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(len(classes))(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
  def __init__(
    self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps, pi=None
  ):
    super(WarmUpCosine, self).__init__()

    self.learning_rate_base = learning_rate_base
    self.total_steps = total_steps
    self.warmup_learning_rate = warmup_learning_rate
    self.warmup_steps = warmup_steps
    self.pi = tf.constant(np.pi)

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          "learning_rate_base": self.learning_rate_base,
          "total_steps": self.total_steps,
          "warmup_learning_rate": self.warmup_learning_rate,
          "warmup_steps": self.warmup_steps,
          "pi": self.pi,
      })
      return config

  def __call__(self, step):
      if self.total_steps < self.warmup_steps:
          raise ValueError("Total_steps must be larger or equal to warmup_steps.")

      cos_annealed_lr = tf.cos(
          self.pi
          * (tf.cast(step, tf.float32) - self.warmup_steps)
          / float(self.total_steps - self.warmup_steps)
      )
      learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

      if self.warmup_steps > 0:
          if self.learning_rate_base < self.warmup_learning_rate:
              raise ValueError(
                  "Learning_rate_base must be larger or equal to "
                  "warmup_learning_rate."
              )
          slope = (
              self.learning_rate_base - self.warmup_learning_rate
          ) / self.warmup_steps
          warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
          learning_rate = tf.where(
              step < self.warmup_steps, warmup_rate, learning_rate
          )
      return tf.where(
          step > self.total_steps, 0.0, learning_rate, name="learning_rate"
      )


y_pred=[]
checkpoint_filepath_h5="drive/MyDrive/Tirocinio/ViT/Model/vit_"+ num_dataset +".h5"
checkpoint_filepath="drive/MyDrive/Tirocinio/ViT/Model/vit_"+ num_dataset

# https://keras.io/api/models/model_training_apis/#evaluate-method
def run_experiment(model):
  total_steps = int((len(x_tr) / BATCH_SIZE) * EPOCHS)
  warmup_epoch_percentage = 0.10
  warmup_steps = int(total_steps * warmup_epoch_percentage)
  scheduled_lrs = WarmUpCosine(
      learning_rate_base=LEARNING_RATE,
      total_steps=total_steps,
      warmup_learning_rate=0.0,
      warmup_steps=warmup_steps,
  )

  optimizer = tfa.optimizers.AdamW(
      learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
  )

  model.compile(
      optimizer=optimizer,
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[
          keras.metrics.SparseCategoricalAccuracy(name="accuracy"), # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseCategoricalAccuracy
          keras.metrics.SparseTopKCategoricalAccuracy(2, name="top-2-accuracy"), #https://www.tensorflow.org/api_docs/python/tf/keras/metrics/SparseTopKCategoricalAccuracy
      ],
  )
  
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath_h5,
      save_weights_only=False, # False=tutto il modello, True=solo i pesi
      mode='max',
      monitor='accuracy',
      verbose=0,
      save_best_only=True,
  )

  model.fit(
      x=x_tr,
      y=y_tr,
      batch_size=BATCH_SIZE,
      verbose=1,
      epochs=3,
      #callbacks = [model_checkpoint_callback],
      #save_freq=4,
      #validation_split=0.2,
  )

  # The model weights (that are considered the best) are loaded into the model.
  #model.load_weights(checkpoint_filepath_h5)
  
  y_pred_ = model.predict(x_te, batch_size=BATCH_SIZE)
  for i in range(len(y_pred_)):
    y_pred.append(y_pred_[i])
  loss_value, accuracy, top_k_accuracy = model.evaluate(x_te, y_te, batch_size=BATCH_SIZE)
  
  print("\n\n\n")
  print("Y_pred is like: " + str(y_pred[0]))
  print("Len of Y_pred is: " + str(len(y_pred)))
  print(f"Test accuracy --- : {round(accuracy * 100, 2)}%") # projection_dim=64 --> acc=0.9412; projection_dim=32 --> acc=0.9533
  print(f"Test top 2 accuracy --- : {round(top_k_accuracy * 100, 2)}%")

 
#  for i, w in enumerate(model.weights): 
#    print(i, w.name)
#    model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)
#    print(i, w.name)
#    print()
  
#  for i in range(len(model.weights)):
#    model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)


  # https://www.tensorflow.org/guide/keras/save_and_serialize
  #model.save(checkpoint_filepath+".h5", save_format="h5") # add .h5 or .hdf5, if not then will be PB -- ValueError: Unable to create dataset (name already exists)
  tf.keras.models.save_model(model,checkpoint_filepath) # (modello, filepath) --> Save in PB
  return model

# Run experiments with the vanilla ViT
#vit = create_vit_classifier(vanilla=True)
#history = run_experiment(vit)

# Run experiments with the Shifted Patch Tokenization and Locality Self Attention modified ViT
vit_net = create_vit_classifier(vanilla=False)
vit_model = run_experiment(vit_net)

# From PB to TFLITE 

checkpoint_filepath="drive/MyDrive/Tirocinio/ViT/Model/vit_"+ num_dataset

saved_model_dir = checkpoint_filepath
converter = tf.lite.TFLiteConverter.from_saved_model(
    saved_model_dir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

fo = open(
    saved_model_dir +"/vit_1_model.tflite", "wb"
    )
fo.write(tflite_model)
fo.close