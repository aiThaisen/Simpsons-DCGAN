import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, LeakyReLU, Conv2DTranspose, BatchNormalization
from keras.optimizers import RMSprop
from keras.initializers import TruncatedNormal

IMAGE_SIZE = 128
NOISE_SIZE = 100
LR_D = 0.00004
LR_G = 0.0004
BATCH_SIZE = 64
EPOCHS = 300
BETA1 = 0.5
WEIGHT_INIT_STDDEV = 0.02
EPSILON = 0.00005
SAMPLES_TO_SHOW = 5


def generator(z, output_channel_dim, training):
    model = Sequential()
    # 8x8x1024
    model.add(Dense(8 * 8 * 1024))
    model.add(Reshape((-1, 8, 8, 1024)))
    model.add(LeakyReLU())
    # 8x8x1024 -> 16x16x512
    model.add(Conv2DTranspose(filters=512, kernel_size=[5, 5], strides=[2, 2], padding='same', kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    model.add(BatchNormalization(training=training, epsilon=EPSILON))
    model.add(LeakyReLU())
    # 16x16x512 -> 32x32x256
    model.add(Conv2DTranspose(filters=256, kernel_size=[5, 5], strides=[2, 2], padding='same', kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    model.add(BatchNormalization(training=training, epsilon=EPSILON))
    model.add(LeakyReLU())
    # 32x32x256 -> 64x64x128
    model.add(Conv2DTranspose(filters=128, kernel_size=[5, 5], strides=[2, 2], padding='same', kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    model.add(BatchNormalization(training=training, epsilon=EPSILON))
    model.add(LeakyReLU())
    # 64x64x128 -> 128x128x64
    model.add(Conv2DTranspose(filters=64, kernel_size=[5, 5], strides=[2, 2], padding='same', kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    model.add(BatchNormalization(training=training, epsilon=EPSILON))
    model.add(LeakyReLU())
    # 128x128x64 -> 128x128x3
    model.add(Conv2DTranspose(filters=64, kernel_size=[5, 5], strides=[1, 1], padding='same', kernel_initializer=TruncatedNormal(stddev=WEIGHT_INIT_STDDEV)))
    model.add(Activation('tanh'))