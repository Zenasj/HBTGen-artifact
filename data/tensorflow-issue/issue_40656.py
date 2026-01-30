import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

def load_file(path):

    arr = np.load(path)['arr_0'][0:None:resolution, 0:None:resolution, :fields]
    for field in range(arr.shape[-1]):

        scaler = StandardScaler()
        scaler.fit(arr[:, :, field])
        arr[:, :, field] = scaler.transform(arr[:, :, field])
    
    name = path.replace(".npz", "").split("/")[-1]

    if "no" in name:
        cat, start_lat, end_lat, start_lon, end_lon, date = name.split("_")
        label = 0
    else:
        cat, press, wind, start_lat, end_lat, start_lon, end_lon, date = name.split("_")
        cat = int(cat)
        if cat < 1:
            label = 0
        else:
            label = 1
            
    return arr, label

pool = mp.Pool(int(mp.cpu_count()))
res = list(tqdm.tqdm(pool.imap(load_file, cat1_files), total=len(cat1_files)))
res = list(tqdm.tqdm(pool.imap(load_file, cat2_files), total=len(cat2_files)))
res = list(tqdm.tqdm(pool.imap(load_file, cat3_files), total=len(cat3_files)))
res = list(tqdm.tqdm(pool.imap(load_file, cat4_files), total=len(cat4_files)))
res = list(tqdm.tqdm(pool.imap(load_file, cat5_files), total=len(cat5_files)))
res = list(tqdm.tqdm(pool.imap(load_file, no_files), total=len(no_files)))

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, paths, batch_size=32, dim=(86, 128, 5), shuffle=True, data_aug=True):
        self.dim = dim
        self.batch_size = batch_size
        self.paths = paths
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        paths_temp = [self.paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(paths_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, paths_temp):
        
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]), dtype="float32")
        y = np.zeros(self.batch_size)
        pool_res = list(pool.imap(load_file, paths_temp))
        
        for i in range(len(pool_res)):
            X[i], y[i] = pool_res[i]
            
        return X, y

def create_model(shape):
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        model = models.Sequential() 

        model.add(layers.Conv2D(8, (2, 2), padding="same", strides=(1, 1), input_shape=(shape))) 
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2), strides=1)) 
        model.add(layers.Conv2D(16, (1, 1), strides=(1, 1))) 
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2), strides=1))
        model.add(layers.Conv2D(32, (2, 2), strides=(1, 1))) 
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2), strides=1))
        model.add(layers.Conv2D(64, (2, 2), strides=(1, 1))) 
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2), strides=1))
        model.add(layers.Conv2D(128, (2, 2), strides=(1, 1))) 
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2), strides=1))
        model.add(layers.Conv2D(256, (2, 2), strides=(1, 1))) 
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2), strides=1))

        model.add(layers.Flatten()) 
        model.add(layers.Dense(128))
        model.add(layers.ReLU())
        model.add(layers.Dense(64))
        model.add(layers.ReLU())
        model.add(layers.Dense(32))
        model.add(layers.ReLU())
        model.add(layers.Dense(1, activation='sigmoid')) 

        METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'), 
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc'),
        ]

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=METRICS) 
    
    return model

train_fold_paths = []
val_fold_paths = []

for fold in range(k):
    
    cat1_fold = cat1_train_files[int(fold/k*len(cat1_train_files)) : int((fold+1)/k*len(cat1_train_files))]
    cat2_fold = cat2_train_files[int(fold/k*len(cat2_train_files)) : int((fold+1)/k*len(cat2_train_files))]
    cat3_fold = cat3_train_files[int(fold/k*len(cat3_train_files)) : int((fold+1)/k*len(cat3_train_files))]
    cat4_fold = cat4_train_files[int(fold/k*len(cat4_train_files)) : int((fold+1)/k*len(cat4_train_files))]
    cat5_fold = cat5_train_files[int(fold/k*len(cat5_train_files)) : int((fold+1)/k*len(cat5_train_files))]  
    yes_fold = cat1_fold + cat2_fold + cat3_fold + cat4_fold + cat5_fold
    no_fold = no_train_files[int(fold/k*len(no_train_files)) : int((fold+1)/k*len(no_train_files))]
    
    fold_data = yes_fold + no_fold
    
    val_fold_paths.append(fold_data)
    
    fold_data = no_fold + yes_fold
    train_fold_paths.append(fold_data)

history_folds = []
model_folds = []

for i_fold in range(k):
    
    train_folds = list(np.arange(i_fold, k-1+i_fold)%k)

    train_paths = []
    for fold_index in train_folds:
        if train_paths != None:
            train_paths += train_fold_paths[fold_index]
        else:
            train_paths = train_fold_paths[fold_index]
            
    val_paths = val_fold_paths[(k+i_fold-1)%k]
    
    dummy_file, _ = load_file(cat1_files[0])
    
    params_train = {'dim': dummy_file.shape, 'batch_size': batch_size, 'shuffle': True, 'data_aug': False}
    params_val = {'dim': dummy_file.shape, 'batch_size': batch_size, 'shuffle': False, 'data_aug': False}

    training_generator = DataGenerator(train_paths, **params_train)
    validation_generator = DataGenerator(val_paths, **params_val)
    
    model = create_model(dummy_file.shape)

    print("Training Fold " + str(i_fold+1))
    print("Training Fold " + str(i_fold+1), file=console_out)
        
    history = model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=True, workers=int(0.5*mp.cpu_count()), 
                                  verbose=1, epochs=epochs, max_queue_size = int(len(train_paths)/batch_size))
    
    history_folds.append(history)
        
    model.save(fold_model_name + str(i_fold+1))
    tf.keras.backend.clear_session()
    
    tf.keras.experimental.terminate_keras_multiprocessing_pools(grace_period=0.1, use_sigkill=True)