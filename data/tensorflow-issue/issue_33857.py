import sys
import unittest

sys.path.append('./handlers')

from __litewave import LiteWaveHandle

import numpy as np
from tensorflow.compat.v2.audio import decode_wav
from tensorflow.compat.v2.io import read_file


class TestLiteWaveHandle(unittest.TestCase):

    def test_load_labels(self):
        lite_wave = LiteWaveHandle()

        labels = {0: '_silence_', 1: '_unknown_', 2: 'yes', 3: 'no', 4: 'up', 5: 'down', 6: 'left', 7: 'right', 8: 'on', 9: 'off', 10: 'stop', 11: 'go'}
        self.assertEqual(lite_wave.load_labels("./models/speech_commands_lite/conv_actions_labels.txt"), labels)

    def test_decode_wav(self):
        lite_wave_tf = LiteWaveHandle()
        lite_wave_np = LiteWaveHandle()


        # set the input tensor using tf wave decoder
        audio, sample_rate = decode_wav(
            read_file("./test/vectors/go/0a2b400e_nohash_0.wav"),
            desired_channels=1,
            desired_samples=16000,
        )
        lite_wave_tf.set_input_tensor_tf(audio)
        input_tensor_tf = lite_wave_tf.get_input_tensor()


        # set the input tensor using custom wave decoder
        audio = lite_wave_np.decode_wav('./test/vectors/go/0a2b400e_nohash_0.wav')
        lite_wave_np.set_input_tensor(audio)
        input_tensor_np = lite_wave_np.get_input_tensor() 

        self.assertTrue(np.array_equal(input_tensor_tf, input_tensor_np))

    def test_set_input_tensor(self):
        lite_wave = LiteWaveHandle()

        audio = lite_wave.decode_wav('./test/vectors/go/0a2b400e_nohash_0.wav')
        lite_wave.set_input_tensor(audio)
        self.assertTrue(np.array_equal(audio, lite_wave.get_input_tensor()))

    def test_get_output_tensor(self):
        lite_wave = LiteWaveHandle()

        output_details, output = lite_wave.get_output_tensor(0)
        self.assertEqual(output_details["name"], 'labels_softmax')

    def test_detect_keywords(self):
        lite_wave = LiteWaveHandle()

        audio = lite_wave.decode_wav('./test/vectors/go/0a2b400e_nohash_0.wav')
        lite_wave.set_input_tensor(audio)

        label_id, prob = lite_wave.detect_keywords()[0]
        self.assertEqual(lite_wave.labels[label_id], 'go')

import os

import re
import wave
import struct

from tflite_runtime.interpreter import Interpreter
import numpy as np

class LiteWaveHandle:
    """A class to handle lite inferences on wave data"""

    def __init__(self):
        self.labels = self.load_labels("./models/speech_commands_lite/conv_actions_labels.txt")

        self.__interpreter = Interpreter(model_path="./models/speech_commands_lite/conv_actions_frozen.tflite")
        self.__interpreter.allocate_tensors()

    def load_labels(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
        return labels

    def decode_wav(self, file_path):
        """wave file to a numpy array of floats limited size to 16000"""
        with wave.open(file_path) as wf:
            astr = wf.readframes(wf.getnframes())
            # convert binary chunks to short 
            audio = struct.unpack("%ih" % (wf.getnframes()* wf.getnchannels()), astr)
            audio = [float(val) / pow(2, 15) for val in audio]
            audio = np.asarray(audio)
            return np.reshape(audio, (16000, 1))

    def set_input_tensor(self, audio):
        """Sets the input tensor."""
        tensor_index = self.__interpreter.get_input_details()[0]['index']
        input_tensor = self.__interpreter.tensor(tensor_index)()
        input_tensor[:] = audio

    def set_input_tensor_tf(self, audio):
        tensor_index = self.__interpreter.get_input_details()[0]['index']
        input_tensor = self.__interpreter.tensor(tensor_index)()[0]
        self.__interpreter.set_tensor(tensor_index, audio)

    def get_input_tensor(self):
        tensor_index = self.__interpreter.get_input_details()[0]['index']
        return self.__interpreter.get_tensor(tensor_index)

    def get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self.__interpreter.get_output_details()[index]
        output = np.squeeze(self.__interpreter.get_tensor(output_details['index']))
        return output_details, output

    def detect_keywords(self, top_k=1):
        """Returns a list of detection results, each a dictionary of object info."""
        self.__interpreter.invoke()

        output_details, output = self.get_output_tensor(0)

        ordered = np.argpartition(-output, top_k)
        return [(i, output[i]) for i in ordered[:top_k]]