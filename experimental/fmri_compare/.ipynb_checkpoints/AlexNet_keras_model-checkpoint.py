# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#Dependencies
import keras
from keras.models import Sequential
# from keras.layers import Dense
from keras import layers
import numpy as np
import tensorflow as tf

path = '/home/mor/NDS_project/imported_data'

# -

gpu_n = 3
# allocating memory and creating the Keras net
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[gpu_n], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[gpu_n], True)

layers.Resizing(
    227, 227, interpolation="bilinear", crop_to_aspect_ratio=False)
# Neural network building
model = Sequential(
    [
    keras.Input(shape=(227, 227, 3)),
    layers.Conv2D(96, 11, strides=(4,4), padding = "same", groups=1,
                  activation="relu", name="conv1"),
    layers.Lambda(lambda x: tf.nn.local_response_normalization(x,
                  depth_radius=2, alpha=2e-05, beta=0.75,
                    bias=1.0), name = "norm1"),
    layers.MaxPooling2D(
    pool_size=(3,3), strides=(2, 2), padding="valid",
        data_format=None, name = "pool1"),
    layers.Conv2D(256, 5, padding = "same", groups = 2,
                  activation="relu", name="conv2"),
    layers.Lambda(lambda x: tf.nn.local_response_normalization(x,
              depth_radius=2, alpha=2e-05, beta=0.75,
                bias=1.0), name = "norm2"),
    layers.MaxPooling2D(
    pool_size=(3,3), strides=(2, 2), padding="valid",
        data_format=None, name = "pool2"),
    layers.Conv2D(384, 3, padding = "same", groups=1,
                  activation="relu", name="conv3"),
    layers.Conv2D(384, 3,  padding = "same", groups=2,
                  activation="relu", name="conv4"),
    layers.Conv2D(256, 3, padding = "same", groups=2,
                  activation="relu", name="conv5"),
    layers.MaxPooling2D(
    pool_size=(3,3), strides=(2, 2), padding="valid",
        data_format=None, name = "pool5"),
    layers.Flatten(),
    layers.Dense(4096, activation='relu', name='fc6'),
    layers.Dense(4096, activation='relu', name='fc7'),
    layers.Dense(1000, name='fc8'),
    layers.Softmax(name='prob')
    ], name = "Alex"
)

# Weights insertion
W = np.load(open(path + "/bvlc_alexnet.npy", "rb"), encoding="latin1",
            allow_pickle=True).item()
len(W)
for layer in model.layers:
    try:
        layer.set_weights(W[layer.name])
    except KeyError:
        pass
model.layers[11].get_weights()

model.build()

# +
# model.save('/home/mor/wisdrc/Alexnetmodel')
# -

model.summary()
