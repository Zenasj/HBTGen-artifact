import numpy as np
import tensorflow as tf
from tensorflow import keras

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

flatten = base_model.output
flatten = Flatten()(flatten)

# FC layer for bounding box prediction
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(bbox_dim1*bbox_dim2, activation="sigmoid")(bboxHead)
bboxHead = Reshape((bbox_dim1, bbox_dim2), name="bounding_box")(bboxHead)

# Second fully-connected layer head to predict the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(label_dim1*label_dim2, activation="softmax")(softmaxHead)
softmaxHead = Reshape((label_dim1, label_dim2), name="class_label")(softmaxHead)

# Create the model
model = Model(inputs=base_model.input, outputs=[bboxHead, softmaxHead])

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}

metrics = {
    "class_label": "categorical_accuracy",
    "bounding_box": tf.keras.metrics.IoU(num_classes, target_class_ids = [0, 1])
}

model.compile(optimizer= Adam(learning_rate=0.001), loss=losses, metrics=metrics)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=1,
    steps_per_epoch=len(train_images) // 10,
    validation_steps=len(test_images) // 10,
    verbose=1
)

class Dataloader():
    def __init__(self, annotation_path, image_path) -> None:
        self.annotation_path = annotation_path
        self.image_path = image_path

    def load(self):
        images = []
        labels = []
        bboxes = []

        #annotation loader
        for file in os.listdir(self.annotation_path):
            annotation = load_annotations(self.annotation_path, file)
            labels.append(tuple([d['class'] for d in annotation]))
            bboxes.append(tuple((d['x1'], d['y1'], d['x2'], d['y2']) for d in annotation))

        #image loader
        for file in os.listdir(self.image_path):
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(self.image_path, file))
                img = cv2.resize(img, (512,512))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

        categories = ['Hotspots', 'Hotstring', 'Bypass Diode active']
        labels = tf.ragged.constant(labels)
        layer = StringLookup(vocabulary=categories)
        indices = layer(labels)
        enc_layer = CategoryEncoding(num_tokens=3, output_mode="one_hot")
        indices = enc_layer(indices)
        labels = indices.to_tensor()
        bboxes = tf.ragged.constant(bboxes, dtype=tf.float32)
        bboxes = bboxes.to_tensor()

        return images, labels, bboxes


def data_generator(images, targets, batch_size=10):
    while True:
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_labels = targets[1][i:i+batch_size]
            batch_bboxes = targets[0][i:i+batch_size]
            yield np.array(batch_images), {"bounding_box": np.array(batch_bboxes), "class_label": np.array(batch_labels)}

def create_dataset(images, targets, batch_size=10):
    n, m = targets[0][0].shape
    p, q = targets[1][0].shape
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(images, targets, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32), 
            {
                "bounding_box": tf.TensorSpec(shape=(batch_size, n, m), dtype=tf.float32),
                "class_label": tf.TensorSpec(shape=(batch_size, p, q), dtype=tf.float32)
            }
        )
    )
    return dataset

train_dataset = create_dataset(train_images, train_targets, batch_size=10)
test_dataset = create_dataset(test_images, test_targets, batch_size=10)