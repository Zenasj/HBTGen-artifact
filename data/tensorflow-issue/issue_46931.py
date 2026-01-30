import random

def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(T, D)))
    model.add(TimeDistributed(Dense(1, activation='relu')))
    return model


if __name__ == "__main__":

    random.set_seed(1)

    parser = argparse.ArgumentParser()

    # inputs for setting environment variable
    parser.add_argument('-node1', default=None, required=True, help='Node 1 IP and port')
    parser.add_argument('-node2', default=None, required=True, help='Node 2 IP and port')
    parser.add_argument('-type', default=None, required=True, help='Node type')
    parser.add_argument('-index', default=None, required=True, help='Node number', type=int)

    args = parser.parse_args()
    node_1 = args.node1
    node_2 = args.node2
    worker_type = args.type
    worker_index = args.index

    # set environment variable
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            "worker": [node_1, node_2]
        },
        'task': {'type': worker_type, 'index': worker_index}
    })

    # load data
    with open('data/X.json', 'rb') as f:
        X = pickle.load(f)

    with open('data/y.json', 'rb') as f:
        y = pickle.load(f)

    # split data, set random state (for replicability)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    _, T, D = X_train.shape

    # dynamic memory allocation
    # gpu = config.experimental.list_physical_devices('GPU')
    # config.experimental.set_memory_growth(gpu[0], True)

    # set training strategy
    strategy = distribute.MultiWorkerMirroredStrategy( )
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    num_epochs = 10000
    batch_size_per_replica = 64
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # create data structure
    train_data = data.Dataset.from_tensor_slices((X_train, y_train))
    val_data = data.Dataset.from_tensor_slices((X_test, y_test))

    train_data = train_data.batch(batch_size, drop_remainder=True)
    val_data = val_data.batch(batch_size, drop_remainder=True)

    # Disable AutoShard.
    # options = data.Options()
    # options.experimental_distribute.auto_shard_policy = data.experimental.AutoShardPolicy.OFF
    # train_data = train_data.with_options(options)
    # val_data = val_data.with_options(options)

    with strategy.scope():
        model = create_model()
        model.compile(
            loss='mse',
            optimizer='adam',
            metrics=[metrics.RootMeanSquaredError()],
        )

    # fit
    model.fit(
        train_data,
        epochs=num_epochs,
        validation_data=val_data,
        # callbacks=[cp_callback, es_callback, tbd_callback]
    )

    # save
    # model.save('model/model.h5')