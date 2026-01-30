import math
from tensorflow import keras
from tensorflow.keras import models

# main.py

import numpy as np
import tensorflow as tf
import imageio

from data_processing.data_processing import DatasetPreparer, DataLoader, num_to_char, char_to_num
from data_processing.mouth_detection import MouthDetector
from model.model import LipReadingModel

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


def main():
    base_dir = "data/A_U_EE_E/temp/"
    original_video_dir = base_dir + "videos"
    original_subtitle_dir = base_dir + "subtitles"
    output_dir = base_dir + "separated"
    video_dir = output_dir + "/videos"
    subtitle_dir = output_dir + "/subtitles"

    mouth_detector = MouthDetector()

    # Instantiate DataLoader (or appropriate class) and DatasetPreparer
    data_loader = DataLoader(detector=mouth_detector)  # Initialize with any necessary parameters

    data_loader.process_all_videos(original_video_dir, original_subtitle_dir, output_dir)

    dataset_preparer = DatasetPreparer(video_directory=video_dir, data_loader=data_loader)  # Provide data_loader here

    dataset = dataset_preparer.prepare_dataset()

    # Fetch a batch of data
    data_iterator = dataset.as_numpy_iterator()
    video_frames, subtitle_tokens = data_iterator.next()

    # # Process frames for saving as a GIF
    # processed_frames = []
    # for frame in video_frames[0]:  # Access the first video in the batch
    #     # Convert normalized frame to uint8
    #     frame = (frame.numpy() * 255).astype(np.uint8)
    #
    #     # Check and reshape if necessary
    #     if frame.shape[-1] == 1:  # Grayscale with singleton dimension
    #         frame = np.squeeze(frame, axis=-1)  # Remove the last dimension for display
    #
    #     processed_frames.append(frame)
    #
    # # Save frames as GIF
    # imageio.mimsave("./animation.gif", processed_frames, fps=30)

    # # Decode subtitle tokens to text for verification
    # decoded_subtitles = tf.strings.reduce_join(
    #     [tf.compat.as_str_any(x) for x in num_to_char(subtitle_tokens[0]).numpy()])
    # print("Decoded Subtitles:", decoded_subtitles.numpy().decode('utf-8'))


    model = LipReadingModel(char_to_num.vocabulary_size())

    yhat = model.predict(video_frames)
    print(tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x in yhat[0]]))
    print(model.input_shape)


if __name__ == "__main__":
    main()


# data_processing/data_processing.py
import csv
import os
import cv2
import tensorflow as tf
import pandas as pd
from data_processing.mouth_detection import MouthDetector

# Define vocabulary for character mapping
vocab = ["A", "U", "EE", "E", " "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


class DataLoader:
    def __init__(self, detector: MouthDetector):
        self.detector = detector

    def load_video(self, path: str) -> tf.Tensor:
        """
        Load video frames, apply mouth detection, convert to grayscale, and normalize.
        """
        cap = cv2.VideoCapture(path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detector.detect_and_crop_mouth(frame)  # Crop to mouth region
            if frame is not None:
                frame = tf.image.rgb_to_grayscale(frame)
                frames.append(frame)

        cap.release()
        if len(frames) == 0:
            raise ValueError(f"No valid frames found in video {path}")

        # Normalize frames
        # mean = tf.math.reduce_mean(frames)
        # std = tf.math.reduce_std(tf.cast(frames, tf.float32))
        # return tf.cast((frames - mean), tf.float32) / std

        # return tf.image.per_image_standardization(frames)

        mean = tf.math.reduce_mean(frames, axis=[0, 1, 2], keepdims=True)
        std = tf.math.reduce_std(tf.cast(frames, tf.float32), axis=[0, 1, 2], keepdims=True)
        frames = tf.cast(frames, tf.float32)
        normalized_frames = (frames - mean) / std
        return normalized_frames

    def load_subtitles(self, path: str) -> tf.Tensor:
        """
        Load subtitles and map them to character indices.
        """
        df = pd.read_csv(path, header=None, names=['start_time', 'end_time', 'subtitle'])
        tokens = []

        for _, row in df.iterrows():
            subtitle = row['subtitle'].strip().upper()
            if subtitle and subtitle != 'IDLE':
                tokens.extend(list(subtitle) + [' '])

        tokens = tokens[:-1]
        tokenized = tf.strings.unicode_split(tokens, input_encoding='UTF-8')
        return char_to_num(tf.reshape(tokenized, [-1]))

    def split_video_by_frames(self, video_path, subtitles_path, output_dir, max_frames=120):
        """
        Split video into chunks of `max_frames` or fewer, keeping word boundaries intact.
        """
        # Load video and subtitle data
        cap = cv2.VideoCapture(video_path)
        df = pd.read_csv(subtitles_path, header=None, names=['start_time', 'end_time', 'subtitle'])

        fps = cap.get(cv2.CAP_PROP_FPS)
        part_num = 1
        chunk_frames = []
        chunk_subtitles = []
        current_frame_count = 0
        start_time = 0

        for index, row in df.iterrows():
            start_ms, end_ms, subtitle = row['start_time'], row['end_time'], row['subtitle']

            # Convert start and end times to frame indices
            start_frame = int((start_ms / 1000) * fps)
            end_frame = int((end_ms / 1000) * fps)
            word_frame_count = end_frame - start_frame

            if current_frame_count + word_frame_count > max_frames:
                # Save current chunk if adding this word exceeds the max frame count
                self.save_chunk(chunk_frames, chunk_subtitles, video_path, output_dir, part_num, fps)
                part_num += 1
                chunk_frames = []
                chunk_subtitles = []
                current_frame_count = 0
                start_time = start_ms  # New start time for the new chunk

            # Add word frames and subtitle
            chunk_subtitles.append((start_ms - start_time, end_ms - start_time, subtitle))

            # Extract frames for this word
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(word_frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                chunk_frames.append(frame)
                current_frame_count += 1

        # Save the last chunk
        if chunk_frames:
            self.save_chunk(chunk_frames, chunk_subtitles, video_path, output_dir, part_num, fps)

        cap.release()

    def save_chunk(self, chunk_frames, chunk_subtitles, video_path, output_dir, part_num, fps):
        """
        Save a chunk of frames and its corresponding subtitles.
        """
        # Define output video and CSV paths
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, "videos", f"{file_name}_{part_num}.mp4")
        output_csv_path = os.path.join(output_dir, "subtitles", f"{file_name}_{part_num}.csv")

        # Save the video chunk
        height, width, _ = chunk_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        for frame in chunk_frames:
            out.write(frame)
        out.release()

        # Save the CSV chunk with adjusted timestamps
        with open(output_csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for start_ms, end_ms, subtitle in chunk_subtitles:
                writer.writerow([start_ms, end_ms, subtitle])

    def process_all_videos(self, video_directory, subtitles_directory, output_directory):
        """
        Process each video in the video_directory, split it, and generate output.
        """
        for video_file in os.listdir(video_directory):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(video_directory, video_file)
                subtitle_path = os.path.join(subtitles_directory, f"{os.path.splitext(video_file)[0]}.csv")
                self.split_video_by_frames(video_path, subtitle_path, output_directory)


class PreProcessor:
    @staticmethod
    def prepare_video_and_subtitles(video_path: tf.Tensor, data_loader: DataLoader):
        """
        Prepares video and subtitle tensors.
        """
        video_path = video_path.numpy().decode('utf-8')
        base_dir = os.path.dirname(os.path.dirname(video_path))
        file_name = os.path.splitext(os.path.basename(video_path))[0]

        subtitles_path = os.path.join(base_dir, 'subtitles', f'{file_name}.csv')
        video_tensor = data_loader.load_video(video_path)
        subtitle_tensor = data_loader.load_subtitles(subtitles_path)

        return video_tensor, subtitle_tensor

    @staticmethod
    def mappable_fn(video_path: tf.Tensor, data_loader: DataLoader):
        """
        A wrapper function that maps video path to frames and alignments.
        """
        return tf.py_function(lambda x: PreProcessor.prepare_video_and_subtitles(x, data_loader), [video_path], [tf.float32, tf.int64])


class Augmentor:
    @staticmethod
    def augment_video(frames: tf.Tensor) -> tf.Tensor:
        """
        Augment video frames by applying transformations such as flipping and concatenating.
        """
        if frames.shape.rank == 5:
            # Apply flipping to each frame in the video sequence
            flipped_frames = tf.map_fn(lambda x: tf.image.flip_left_right(x), frames)
        elif frames.shape.rank == 4:
            # Apply flipping directly if frames already have 4D shape
            flipped_frames = tf.image.flip_left_right(frames)
        else:
            raise ValueError("Expected frames to have 4 or 5 dimensions, got shape: {}".format(frames.shape))

        # Concatenate original and flipped frames along the batch dimension
        return tf.concat([frames, flipped_frames], axis=0)


class DatasetPreparer:
    def __init__(self, video_directory: str, data_loader: DataLoader):
        self.video_directory = video_directory
        self.data_loader = data_loader

    def prepare_dataset(self) -> tf.data.Dataset:
        """
        Prepare a dataset that reads videos and subtitles, applies augmentations, and batches data.
        """
        dataset = tf.data.Dataset.list_files(f"{self.video_directory}/*.mp4")
        dataset = dataset.shuffle(100)
        dataset = dataset.map(lambda path: PreProcessor.mappable_fn(path, self.data_loader))

        # 5400 frames in each training video (assuming 3 minutes 30 fps)
        # 240 tokens in each video (120 letters plus space token to separate)
        frames, alignments = dataset.as_numpy_iterator().next()
        print(frames.shape, alignments.shape)
        dataset = dataset.padded_batch(2, padded_shapes=([120, None, None, None], [12]))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(lambda frames, alignments: (Augmentor.augment_video(frames), alignments))

        return dataset


# data_processing/mouth_detection.py

import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class MouthDetector:
    def __init__(self, model_path='assets/face_landmarker.task', num_faces=1):
        base_options = python.BaseOptions(model_asset_path=model_path, delegate="GPU")
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=num_faces)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def expand_bounding_box(self, xmin, ymin, xmax, ymax, padding_ratio=0.4):
        width = xmax - xmin
        height = ymax - ymin
        pad_w = int(width * padding_ratio)
        pad_h = int(height * padding_ratio)
        xmin = max(xmin - pad_w, 0)
        ymin = max(ymin - pad_h, 0)
        xmax = xmax + pad_w
        ymax = ymax + pad_h
        return xmin, ymin, xmax, ymax

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image

    def crop_mouth_from_landmarks(self, rgb_image, detection_result, target_size=(250, 100)):
        if detection_result and detection_result.face_landmarks:
            try:
                face_landmarks = detection_result.face_landmarks[0]
                # These are the landmarks for the mouth
                mouth_landmarks = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                                   146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                                   78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                                   95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

                x_coords = [face_landmarks[landmark].x for landmark in mouth_landmarks]
                y_coords = [face_landmarks[landmark].y for landmark in mouth_landmarks]
                xmin, xmax = int(min(x_coords) * rgb_image.shape[1]), int(max(x_coords) * rgb_image.shape[1])
                ymin, ymax = int(min(y_coords) * rgb_image.shape[0]), int(max(y_coords) * rgb_image.shape[0])

                xmin, ymin, xmax, ymax = self.expand_bounding_box(xmin, ymin, xmax, ymax)
                cropped_mouth = rgb_image[ymin:ymax, xmin:xmax]
                return cv2.resize(cropped_mouth, target_size, interpolation=cv2.INTER_AREA)
            except cv2.error:
                return None
        return None

    def detect_and_crop_mouth(self, frame, target_size=(250, 100)):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image_input = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image_input)
        return self.crop_mouth_from_landmarks(mp_image_input.numpy_view(), detection_result, target_size=target_size)


# model/model.py

import tensorflow as tf
from tensorflow.keras import layers, models


class LipReadingModel:
    def __init__(self, input_shape=(250, 100), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        """
        Creates and compiles the lip-reading model using the Sequential API.
        """
        model = models.Sequential()

        model.add(layers.Conv3D(128, 3, input_shape=(120, 100, 250, 1), padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPool3D((1, 2, 2)))

        model.add(layers.Conv3D(256, 3, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPool3D((1, 2, 2)))

        model.add(layers.Conv3D(120, 3, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPool3D((1, 2, 2)))

        model.add(layers.TimeDistributed(layers.Flatten()))

        model.add(layers.Bidirectional(layers.LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
        model.add(layers.Dropout(.5))

        model.add(layers.Bidirectional(layers.LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
        model.add(layers.Dropout(.5))

        model.add(layers.Dense(self.num_classes + 1, kernel_initializer='he_normal', activation='softmax'))

        print(model.summary())

        return model

    def load(self, model_path):
        """
        Load a pre-trained model from a given path.
        """
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, frames):
        """
        Predict the sequence of characters from video frames.
        """
        # frames = frames / 255.0  # Normalize frames
        return self.model.predict(frames)

    def save(self, model_path):
        """
        Save the trained model to the specified path.
        """
        self.model.save(model_path)