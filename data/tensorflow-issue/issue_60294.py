from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import cv2
import glob
import numpy as np

save_to = 'C:\\Users\\wefy2\\PycharmProjects\\CNN-Facial-Recognition-master1\\data'
all_faces = [img for img in glob.glob('C:\\Users\\wefy2\\PycharmProjects\\CNN-Facial-Recognition-master1\\data\\gt_db\\s*\\*.jpg')]

faces_x = []
faces_y = []

faceCascade = cv2.CascadeClassifier('data\\haarcascade_frontalface.xml')

for i, face in enumerate(all_faces):
    image = cv2.imread(face)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 1:
        x, y, w, h = faces[0]
        cropped_img = image[y:y + h, x:x + w]

        faces_x.append(cv2.resize(cropped_img, (128, 128)))
        faces_y.append(int(face.split('\\')[-2][1:]))

    print('Finished: ', i, ' Out of: ', len(all_faces))


faces_x, faces_y = np.array(faces_x), np.array(faces_y)

np.save(save_to + 'x_train', faces_x)
np.save(save_to + 'y_train', faces_y)

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# loading data
faces_x = np.load('datax_train.npy')
faces_y = np.load('datay_train.npy')
faces_x = tf.expand_dims(faces_x, axis=0)
faces_y = tf.expand_dims(faces_y, axis=0)
train_dataset = tf.data.Dataset.from_tensor_slices((faces_x, faces_y))
print('Faces were loaded successfully.')

print (tf.__version__)
# Construct the fully connected hashing layers
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                           activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                           activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2,
                           padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2,
                           padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='sigmoid')
])


# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss(margin=3.0))
print(model.summary())
print('Model Compiled Successfully.')


# Train the model
print('Training has started.')
history = model.fit(train_dataset, epochs=10, verbose=1)


# Save the model
model.save('models/face_id_model')
print('Training is finished.')

import numpy as np
import cv2
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class FaceID:
    def __init__(self):
        model = tf.keras.Sequential()
        net = tf.keras.applications.MobileNet(input_shape=(128, 128, 3), weights='imagenet', include_top=False)
        model.add(net)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.features_extractor = model

        self.x_holder = tf.placeholder(shape=[None, 1024], dtype=tf.float32)
        fc_1 = tf.layers.Dense(units=512, activation=tf.nn.relu)(self.x_holder)
        fc_2 = tf.layers.Dense(units=128, activation=tf.nn.sigmoid)(fc_1)

        self.face_id = fc_2

        self.sess = None

    def load_network(self, path='models\\face_id_model\\variables\\variables'):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, path)

    def get_id(self, imgs):
        imgs = imgs.reshape((-1, 128, 128, 3))
        features = self.features_extractor.predict(imgs)
        embeds = self.sess.run([self.face_id], feed_dict={self.x_holder: features})

        return embeds[0]


class FaceExtractor:
    def __init__(self, cascade_path='data\\haarcascade_frontalface.xml'):
        self.faceCascade = cv2.CascadeClassifier(cascade_path)

    def extract_single_face_from_path(self, img_path):
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cropped_img = image[y:y + h, x:x + w]
            return cv2.resize(cropped_img, (128, 128))
        else:
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 10)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                cropped_img = image[y:y + h, x:x + w]
                return cv2.resize(cropped_img, (128, 128))

        return None

    def faces_from_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.faceCascade.detectMultiScale(gray, 1.3, 5)


test_nn = FaceID()
face_ex = FaceExtractor()
test_nn.load_network()
ref_face = face_ex.extract_single_face_from_path("ref.jpg")
ref_face_hash = test_nn.get_id(ref_face)[0]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = face_ex.faces_from_image(frame)

    for face in faces:
        x, y, w, h = face
        cropped_face = cv2.resize(frame[y:y + h, x:x + w], (128, 128))
        cropped_hash = test_nn.get_id(cropped_face)[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), 1, 3)

        distance_1 = np.sum(np.power(ref_face_hash - cropped_hash, 2))

        if distance_1 <= 3:
            cv2.putText(frame, 'ref ', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Nan ', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 1, 2, cv2.LINE_AA)

    cv2.imshow('My FaceID', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ret, frame = cap.read()
        break