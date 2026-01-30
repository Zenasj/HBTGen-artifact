import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import multiprocessing
import os
import portpicker
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub

print (tf.__version__)
#1. Define Workers
def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec, job_name="worker", task_index=i, config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec, job_name="ps", task_index=i, protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, task_id=0, task_type="worker",rpc_layer="grpc")
  return cluster_resolver

NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

variable_partitioner = (
    tf.distribute.experimental.partitioners.FixedShardsPartitioner(
        num_shards=NUM_PS))

strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
messages = [word, sentence, paragraph, paragraph]
#labels=["1","2","3"]
reviews = [[1,0,0],[0,1,0],[0,0,1],[0,0,1]]


encoder=hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

X_train=encoder(messages)

# from sentence_transformers import SentenceTransformer
#
# bertmodel = SentenceTransformer('stsb-roberta-large')
# X_train=bertmodel.encode(messages)

BUFFER_SIZE = len(X_train)
BATCH_SIZE_PER_REPLICA = 3
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 4


with strategy.scope():

    model = keras.Sequential()

    model.add(
        keras.layers.Dense(
            units=256,
            input_shape=(X_train.shape[1],),
            activation='relu'
        )
    )
    model.add(
        keras.layers.Dropout(rate=0.5)
    )

    model.add(
        keras.layers.Dense(
            units=128,
            activation='relu'
        )
    )
    model.add(
        keras.layers.Dropout(rate=0.5)
    )

    model.add(keras.layers.Dense(3, activation='softmax'))
    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=keras.optimizers.Adam(0.001),
    #     metrics=['accuracy']
    # )

    # history = model.fit(
    #     np.array(X_train), np.array(reviews),
    #     epochs=10,
    #     batch_size=16,
    #     verbose=1,
    #     shuffle=True
    # )
    optimizer=keras.optimizers.Adam(0.001)
    accuracy = keras.metrics.Accuracy()


def step_fn(x_train_slice):
    x_train, y_train = next(x_train_slice)
    with tf.GradientTape() as tape:

        pred=model(x_train,training=True)
        # tf.print(x_train)
        # tf.print(pred)
        # tf.print(y_train)

        per_example_loss = keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)(y_train, pred)
        loss = tf.nn.compute_average_loss(per_example_loss)
        gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    argmax_pred = tf.one_hot(tf.math.argmax(pred, axis=1), depth=pred.shape[1])
    tf.print("train values are",x_train)
    tf.print(" pred Values are : ", pred)
    tf.print(" ArgMAx Values are ",type(tf.math.argmax(pred,axis=0)))
    # tf.print(" actual_pred Values are : ", actual_pred)
    tf.print(" Labels  are : ", y_train)
    tf.print(" Labels Max Values are : ", tf.argmax(y_train))
    tf.print(" tf.math.argmax(pred, axis=1) ", tf.math.argmax(pred, axis=1))
    tf.print("argmax_pred ",argmax_pred)
    accuracy.update_state(y_train, argmax_pred)
    tf.print("Accuracy is : ",accuracy.result())

    return (loss,tf.math.argmax(pred,axis=0))

@tf.function
def distributed_train_step(per_worker_iterator):
    (losses,argmaxes) = strategy.run(step_fn,args=(per_worker_iterator,))
    strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
    return argmaxes


@tf.function
def per_worker_dataset_fn():
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, reviews)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    print(train_dataset)
    tf.print("train_dataset ",train_dataset)
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
    return train_dist_dataset


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)
num_epoches = 5
steps_per_epoch = 1
for i in range(num_epoches):
  accuracy.reset_states()
  for _ in range(steps_per_epoch):
    argmaxes=coordinator.schedule(distributed_train_step, args=(per_worker_iterator,))

    # Wait at epoch boundaries.
  coordinator.join()
  print("argmaxes", argmaxes.fetch())
  print ("Finished epoch %d, accuracy is %f.",(i,accuracy.result().numpy()))

import tensorflow.keras as keras
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf

word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
messages = [word, sentence, paragraph]
reviews = [[1,0,0],[0,1,0],[0,0,1]]

encoder=hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

X_train=encoder(messages)

# def UniversalEmbeddingsHook(sentences):
#      print(X_train.shape[1])
#      return X_train
#
# # with strategy.scope():
# inputSentences=keras.layers.Input(shape=(1,),dtype=tf.string)
# embeddings=keras.layers.Lambda(UniversalEmbeddingsHook,output_shape=(512,))(inputSentences)
# dense=keras.layers.Dense(256,activation='relu')(embeddings)
# dropouts=keras.layers.Dropout(rate=0.5)(dense)
# ll=keras.layers.Dense(128,activation='relu')(dropouts)
# pred=keras.layers.Dense(3,activation='softmax')(ll)
# models=keras.Model(inputs=[inputSentences],outputs=pred)
# models.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# models.summary()
# models.fit(messages,reviews,epochs=60)



model = keras.Sequential()

model.add(
  keras.layers.Dense(
    units=256,
    input_shape=(X_train.shape[1], ),
    activation='relu'
  )
)
model.add(
  keras.layers.Dropout(rate=0.5)
)

model.add(
  keras.layers.Dense(
    units=128,
    activation='relu'
  )
)
model.add(
  keras.layers.Dropout(rate=0.5)
)

model.add(keras.layers.Dense(3, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)


history = model.fit(
    np.array(X_train), np.array(reviews),
    epochs=10,
    batch_size=16,
    verbose=1,
    shuffle=True
)
print(history)

tf.print("Accuracy ",model.evaluate(np.array(X_train), np.array(reviews)))

validate=X_train[1:2]
y_pred = model.predict(np.array(X_train))
print(y_pred)
model.save("/tmp/transfer/classifier/",history)