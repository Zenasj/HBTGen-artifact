from tensorflow import keras

import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

def set_visible_devices(device_idx: str, device_type: str, use_tf: bool):
    """
    Limit visibility of devices available in the cluster for the current application.

    @param device_idx: comma-delimited indexes or "ALL"
    @param device_type: [XLA_](CPU|GPU), e.g., "GPU"
    @param use_tf: set visibility of devices through tensorflow APIs or no

    @return: list of selected devices
    """

    if use_tf:
      filtered_type_devices = tf.config.list_physical_devices(device_type=device_type)
      visible_devices = select_devices_by_index(device_idx, filtered_type_devices)

      # always add CPU
      has_cpu = device_type[-3:] == "CPU"
      if not has_cpu:
          visible_devices += tf.config.list_physical_devices(
              device_type="{}CPU".format(device_type[:-3]).upper()
          )

      tf.config.set_visible_devices(visible_devices)
    elif device_type == "GPU":
      os.environ["CUDA_VISIBLE_DEVICES"]=device_idx
    else:
      raise NotImplementedError()

def select_devices_by_index(device_idx: str, devices: list):
    """
    @param device_idx: comma-delimited indexes or "ALL"
    @param devices: list of PhysicalDevices properties
    @return: subset of a list corresponding to indexes
    """
    assert len(devices) > 0
    filtered_devices = []
    if device_idx.upper() == "ALL":
        filtered_devices = devices
    else:
        device_idx = device_idx.split(",")
        assert len(devices) >= len(device_idx)
        for i in device_idx:
            for d in devices:
                if i == d.name.split(":")[-1]:
                    filtered_devices.append(d)

    print("{} devices selected:".format(len(filtered_devices)))
    for d in filtered_devices:
        print(d.name)

    return filtered_devices

def build_model():
    # build dummy classification model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

if __name__ == "__main__":
  use_tf = sys.argv[1].upper() == "TRUE"
  set_visible_devices(device_idx="0,1", device_type="GPU", use_tf = use_tf)
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
      model = build_model()