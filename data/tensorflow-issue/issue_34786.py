import random
import tensorflow as tf
from tensorflow import keras

####### running this part doubles memory every two times ##########
for x_ in batch(np.random.uniform(size=(100,6,108,192,3)).astype(np.float32), 10):
     with tf.GradientTape() as tape:
             count_ = tf.reduce_sum(model(x_))

from tensorflow.keras import layers
import numpy as np
 
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
                        
 
with tf.device('/gpu:0'):
    inputs = tf.keras.Input(shape=(6, 108, 192, 3), name='img') ## (108, 192, 3)
    x = layers.TimeDistributed(layers.Conv2D(16, 3, activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.Conv2D(16, 3, activation='relu'))(x)
    block_1_output = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
 
    x = layers.TimeDistributed(layers.Conv2D(16, 3, activation='relu', padding='same'))(block_1_output)
    # x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_1_output])
    block_3_output = layers.TimeDistributed(layers.MaxPooling2D(2))(block_3_output)
 
    x = layers.TimeDistributed(layers.Conv2D(16, 3, activation='relu'))(block_3_output)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
 
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1)(x)
    counts = tf.keras.activations.softplus(x)
    # x = layers.Dropout(0.5)(x)
    # outputs = layers.Dense(10, activation='softmax')(x)
 
    model = tf.keras.Model(inputs, counts, name='toy_resnet')
    model.summary()
 
            ### everytime this is run, gpu memory grows
for x_ in batch(np.random.uniform(size=(100,6,108,192,3)).astype(np.float32), 10):
    temp = model(x_)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

####### here's an example model #############
with tf.device('/gpu:0'):
    inputs = tf.keras.Input(shape=(108, 192, 3), name='img') ## (108, 192, 3)
    x = layers.Conv2D(16, 3, activation='relu')(inputs)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_1_output)
    # x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.add([x, block_1_output])
    block_2_output = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_2_output)
    x = layers.add([x, block_2_output])
    x = layers.MaxPooling2D(2)(x)
    block_3_output = layers.GlobalAveragePooling2D()(x)

    # x = layers.Flatten()(x)
    # x = layers.Dense(16, activation='relu')(x)
    # x = layers.Dense(1)(x)
    # counts = tf.keras.activations.softplus(x)

    cnn = tf.keras.Model(inputs, block_3_output, name='toy_resnet')
    # model = tf.keras.Sequential()
    # model.add(layers.TimeDistributed(cnn, input_shape=(6, 108, 192, 3)))
    # model.add(layers.Dense(16, activation='relu'))
    # model.add(layers.Dense(1))

    input_sequences = tf.keras.Input(shape=(6, 108, 192, 3)) ## (108, 192, 3)
    x = layers.TimeDistributed(cnn)(input_sequences)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1)(x)
    counts = tf.keras.activations.softplus(x)
    model = tf.keras.Model(input_sequences, counts, name='toy_resnet')
    model.summary()

###### run code below using tf.signal.frame and watch the VRAM memory on the GPU.  You will see it grow before throwing an error.  Then do the same with "more_itertools" and you'll see it's fine.

for n in range(50):
    ##### exchange more_itertools with tf.signal.frame to get memory leak
    x_mb = tf.signal.frame(np.random.uniform(size=(200,108,192,3)).astype(np.float32), args.num_frames, 1, axis=0)
    for x_ in batch(x_mb, 10):
    ################# no memory leak with more_itertools for sliding window framing ###############
    # for x_ in batch(np.array(list(more_itertools.windowed(np.random.uniform(size=(100, 108, 192, 3)).astype(np.float32), n=6, step=1))),10):
    #########################################
        temp = model(x_)

data_loader_train = np.random.uniform(size=(100, 120, 108, 192, 3)).astype(np.float32)

indcount = 0
for epoch in range(args.epochs):
    train_loss = 0
    for ind in range(len(data_loader_train)):
    # for ind in np.random.permutation(len(data_loader_train)):
    #     print(r'epoch:  %i,   index:  %i' % (epoch, ind), end="\r")
        print(r'indcount:  %i' % indcount, end="\r")
        # x_mb = np.array(list(more_itertools.windowed(data_loader_train[ind][0], n=args.num_frames, step=1)))
        # y_mb = data_loader_train[ind][1]
        x_mb = np.array(list(more_itertools.windowed(data_loader_train[ind], n=args.num_frames, step=1)))
        y_mb = np.random.randint(20,40)
        count = 0
        grads = [np.zeros_like(x) for x in model.trainable_variables]
        # print("index:  " + str(ind))
        for x_ in batch(x_mb, args.batch_size):
            indcount += 1
            with tf.GradientTape() as tape:
                count_ = tf.reduce_sum(model(x_))
            count += count_
            grads_ = tape.gradient(count_, model.trainable_variables)
            grads = [x1 + x2 for x1, x2 in zip(grads, grads_)]

        # grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
        loss = count-y_mb
        globalstep = optimizer.apply_gradients(zip([2*loss*x for x in grads], model.trainable_variables))

        tf.summary.scalar('loss/train', loss**2, globalstep)
## after 5 epochs