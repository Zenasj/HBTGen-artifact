import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

class Args():
    def __init__(self,dataset_path="../datasets/train" ,mymodel="outputs/my_model.h5", label="outputs/le.pickle", embeddings="outputs/embeddings.pickle", image_out="../datasets/test/img_test.jpg", image_in="../datasets/test/001.jpg", video_out="../datasets/videos_output/test.mp4", video_in="../datasets/videos_input/Ok_Arya_Stark.mp4", image_size='112,112', model='../models/arcface_r100_v1/model,0', ga_model='', detector='', gpu=0, det=0, flip=0, threshold=1.24):
        
        self.dataset=dataset_path
        self.mymodel=mymodel
        self.le=label
        self.embeddings=embeddings
        self.image_out=image_out
        self.image_in=image_in
        self.video_out=video_out
        self.video_in=video_in
        self.image_size=image_size
        self.model=model
        self.ga_model=ga_model
        self.detector=detector
        self.gpu=gpu
        self.det=det
        self.flip=flip
        self.threshold=threshold
        
    def init_parsearges(self):
        ap = argparse.ArgumentParser()
        
        # Argument of insightface
        ap.add_argument("--dataset", default=self.dataset,
                help="Path to training dataset")
        
        ap.add_argument("--mymodel", default=self.mymodel,
            help="Path to recognizer model")
        ap.add_argument("--le", default=self.le,
            help="Path to label encoder")
        ap.add_argument("--embeddings", default=self.embeddings,
            help='Path to embeddings')
        ap.add_argument("--image-out", default=self.image_out,
            help='Path to output image')
        ap.add_argument("--image-in", default=self.image_in,
            help='Path to output image')
        ap.add_argument("--video-out", default=self.video_out,
            help='Path to output video')
        ap.add_argument("--video-in", default=self.video_in)


        ap.add_argument('--image-size', default=self.image_size, help='')
        ap.add_argument('--model', default=self.model, help='path to load model.')
        ap.add_argument('--ga-model', default=self.ga_model, help='path to load model.')
        ap.add_argument('--detector', default=self.detector, type=str, help='face detector name')
        ap.add_argument('--gpu', default=self.gpu, type=int, help='gpu id')
        ap.add_argument('--det', default=self.det, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        ap.add_argument('--flip', default=self.flip, type=int, help='whether do lr flip aug')
        ap.add_argument('--threshold', default=self.threshold, type=float, help='ver dist threshold')

        args = ap.parse_args()
        
        return args



class SoftMax():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        from keras.losses import categorical_crossentropy
        from keras.models import Sequential
        from keras.optimizers import Adam
        from keras.layers import Dense, Dropout


        # create model
        model = Sequential()

        # add model layers
        model.add(Dense(1024, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        # loss and optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss=categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model


def make_model(args, classifier=SoftMax):

    # Load the face embeddings
    data = pickle.loads(open(args.embeddings, "rb").read())

    num_classes = len(np.unique(data["names"])) 
    ct = ColumnTransformer([('myٔName', OneHotEncoder(), [0])])
    labels = np.array(data["names"]).reshape(-1, 1)
    labels = ct.fit_transform(labels)

    embeddings = np.array(data["embeddings"])

    # Initialize Softmax training model arguments
    BATCH_SIZE = 32
    EPOCHS = 32
    input_shape = embeddings.shape[1]

    # Build classifier
    init_classifier = classifier(input_shape=(input_shape,), num_classes=num_classes)
    model = init_classifier.build()

    # Create KFold
    cv = KFold(n_splits = 5, random_state = None, shuffle=True)
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
    # Train
    for train_idx, valid_idx in cv.split(embeddings):
        X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
        his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))


    # write the face recognition model to output
    model.save(args.mymodel)
    f = open(args.le, "wb")
    f.write(pickle.dumps(LabelEncoder()))
    f.close()

x_train = vectorizer.fit_transform(train_texts).todense()
x_val = vectorizer.transform(val_texts).todense()