import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

def model_builder():
    resnet = ResNet50(include_top=False, weights='imagenet')
    model = Sequential()
    model.add(resnet)
    
    model.add(Dense(100, activation="softmax"))

    return model

def experiment_dist(model_builder, x_train, y_train, x_test, y_test):  
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)
    test_ds = mirrored_strategy.experimental_distribute_dataset(test_ds)
   
    

    with mirrored_strategy.scope():

        test_loss = tf.keras.metrics.Mean(name='test_loss')

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

        model = model_builder()
        # model = ResNet50(include_top=False, weights='imagenet')

        optimizer = tf.keras.optimizers.SGD()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
        def train_step(inputs):
            x, y = inputs
            x = tf.dtypes.cast(x, tf.float32)
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = loss_object(y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(y, predictions)

        def test_step(inputs) :
            x, y = inputs
            x = tf.dtypes.cast(x, tf.float32)

            predictions = model(x)
            t_loss = loss_object(y, predictions)

            test_loss(t_loss)
            test_accuracy(y, predictions)

        @tf.function
        def distributed_train_step(inputs):
            per_replica_losses = strategy.experimental_run_v2(train_step,
                                                            args=(inputs, ))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
        
        @tf.function
        def distributed_test_step(inputs):
            return strategy.experimental_run_v2(test_step, args=(inputs, ))            
    iter_count = 0
    MAX_ITER = 95000

    while True:
        print("start")
        for inputs in train_ds:
            print("train_loop")
            train_step(inputs)
            iter_count += 1
            if iter_count > MAX_ITER:
                break


        for test_images, test_labels in test_ds:
            images = tf.dtypes.cast(images, tf.float32)
            test_step(test_images, test_labels)

        template = 'Iteration {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(iter_count,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        if iter_count > MAX_ITER:
            break
    
experiment_dist(model_builder, x_train, y_train, x_test, y_test)