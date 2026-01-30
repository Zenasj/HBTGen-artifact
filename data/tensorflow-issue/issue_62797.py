import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

@dataclass
class M:
    train = True
    results = True
    sensitivity = False
    filepath = str(pathlib.Path(__file__).parent.resolve())
    verbose = 1 # 1)Lots 3) Few
    Start = -300
    End = 51
    pixel_width = 334 
    pixel_height = 217
    pixel_horizontal_padding = 14

    target_size = (pixel_height, pixel_width-(2*pixel_horizontal_padding), 1)
    

##########################################################################################

def main():
    train_profiles, train_x, train_y, train_images, validation_profiles, validation_x, validation_y, validation_images, test_profiles, test_x, test_y, test_images, input_shape, output_shape = load_dataset()
    # model, earlyStopping, reduceLR = Model_Compiler(input_shape, output_shape)
    # model, history = Model_Trainer(model, train_images, validation_images, train_y, validation_y, earlyStopping, reduceLR)
    # histroy_plots(history)
    # save_model(model)
    
    model = load_model()
    model.summary()


    image_array = test_images[0]
    #prediction = Prediction(model, test_images)
    plt.imshow(image_array, cmap='gray')
    print(image_array.shape)
    
    grad_cam(model, image_array)

def load_and_preprocess_image(image_path):
    image = Image.open(f'Images/{image_path}.png')
    width, height = image.size
    image = image.crop((M.pixel_horizontal_padding, 0, (width - M.pixel_horizontal_padding), height))
    image = image.convert('L')
    image_array = np.array(image)
    image_array = image_array.reshape(M.target_size)
    image_array = image_array.astype('float32') / 255.0
    return image_array



def load_dataset():
    df = pd.read_csv('Coordinates.csv')
    images = [load_and_preprocess_image(i) for i in range(len(df))]
    profiles = df.iloc[:, 0].tolist()
    x_coordinates = df.iloc[:, 1:302].values.tolist()
    y_coordinates = df.iloc[:, -50:].values.tolist()
    
    data = list(zip(profiles, x_coordinates, y_coordinates, images))
    
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    train_data, validation_data = train_test_split(train_data, test_size=0.5, random_state=42)
    
    train_profiles, train_x, train_y, train_images = zip(*train_data)
    validation_profiles, validation_x, validation_y, validation_images = zip(*validation_data)
    test_profiles, test_x, test_y, test_images = zip(*test_data)
    
    train_profiles, train_x, train_y, train_images = np.array(train_profiles), np.array(train_x), np.array(train_y), np.array(train_images)
    validation_profiles, validation_x, validation_y, validation_images = np.array(validation_profiles), np.array(validation_x), np.array(validation_y), np.array(validation_images)
    test_profiles, test_x, test_y, test_images = np.array(test_profiles), np.array(test_x), np.array(test_y), np.array(test_images)
    
    input_shape = train_images.shape[1:]
    output_shape = train_y.shape[1:]
    
    return (
        train_profiles, train_x, train_y, train_images,
        validation_profiles, validation_x, validation_y, validation_images,
        test_profiles, test_x, test_y, test_images, input_shape, output_shape
    )
    

def Model_Compiler(input_shape, output_shape):
    checkpoint = ModelCheckpoint(M.filepath , monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    earlyStopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.00001)

    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(np.prod(output_shape)))
    model.add(layers.Dropout(0.2)) # Output layer with number of neurons corresponding to output shape
    model.add(layers.Reshape(output_shape))  # Reshape output to the target size
    model.compile(optimizer='adam', loss='mse')
    
    return model, earlyStopping, reduceLR


def Model_Trainer(model, x_train, x_val, y_train, y_val, earlyStopping, reduceLR):
    history = model.fit(
          x=x_train
        , y=y_train
        , validation_data=(x_val,y_val)
        , batch_size=4
        , epochs= 10000
        , callbacks=[earlyStopping, reduceLR]
        , verbose=M.verbose
    )
    return model, history


def Prediction(model, x_test):
    model.summary()
    prediction = model.predict(x_test)
    return prediction

def data_saver(prediction):
    
    
    
    
    for i in range (len(test_df)):
        predicted_values = pred_df[i]
        real_values = test_df[i]
        with open('CNN_2D_Results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(x_values)
            writer.writerow(predicted_values)
            writer.writerow(x_values)
            writer.writerow(real_values)


def display_images_in_loop(image_list):
    for index, image in enumerate(image_list):
        plt.imshow(image, cmap='gray')  # Set cmap to 'gray' for grayscale
        plt.axis('off')
        plt.title(f'Image {index + 1}')
        plt.show()
        input("Press Enter to continue...")

def histroy_plots(history):
    print(history.history.keys())
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model(): 
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    return model

def grad_cam(model, img_array):
    grad_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_5').output)

    with tf.GradientTape() as tape:
        last_conv_output = grad_model(img_array[np.newaxis, ...])
        tape.watch(last_conv_output)

        preds = model(img_array[np.newaxis, ...])
    

        # Get the top predicted class index
        top_class_index = tf.argmax(preds[0])

        grads = tape.gradient(preds[:, top_class_index], last_conv_output)


    # Compute the average gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    # Multiply each feature map by its importance score (obtained from the gradients)
    last_conv_output = last_conv_output.numpy()[0]
    heatmap = np.mean(last_conv_output * pooled_grads[..., np.newaxis], axis=-1)
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap


if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define a simple model for demonstration
def create_simple_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, name='conv2d_1'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Reshape((50,))
    ])
    return model

# Load and preprocess a sample image for Grad-CAM
def load_and_preprocess_image_for_grad_cam(image_path):
    # Replace this with your actual image loading code
    image = Image.fromarray(np.random.randint(0, 255, size=(217, 306)).astype('uint8'))
    image_array = np.array(image).reshape((217, 306, 1)).astype('float32') / 255.0
    return image_array

# Create a simple model
input_shape = (217, 306, 1)
model = create_simple_model(input_shape)

# Load a sample image for Grad-CAM
image_array = load_and_preprocess_image_for_grad_cam('sample_image')

# Function to generate Grad-CAM
def generate_grad_cam(model, img_array):
    grad_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_2').output)

    with tf.GradientTape() as tape:
        last_conv_output = grad_model(img_array[np.newaxis, ...])
        tape.watch(last_conv_output)

        preds = model(img_array[np.newaxis, ...])

        # Get the top predicted class index
        top_class_index = tf.argmax(preds[0])

        grads = tape.gradient(preds[:, top_class_index], last_conv_output)

    # Compute the average gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Multiply each feature map by its importance score (obtained from the gradients)
    last_conv_output = last_conv_output.numpy()[0]
    heatmap = np.mean(last_conv_output * pooled_grads[..., np.newaxis], axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# Generate and display Grad-CAM
heatmap = generate_grad_cam(model, image_array)

# Display the original image and Grad-CAM side by side
plt.subplot(1, 2, 1)
plt.imshow(image_array.squeeze(), cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(heatmap, cmap='viridis')
plt.title('Grad-CAM')

plt.show()

grads = tape.gradient(preds[:, top_class_index], last_conv_output,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)