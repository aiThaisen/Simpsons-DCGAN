from __future__ import print_function, division

import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import random
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob


import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        optimizer = Adam(lr=0.00004, beta_1=0.5)
        optimizer_gen = Adam(lr=0.0004, beta_1=0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(8 * 8 * 1024, activation="linear", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 1024)))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=512, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=256, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=128, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=64, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=self.channels, kernel_size=[5, 5], strides=[1, 1],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(Activation("tanh"))

        print("Generator")
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(
            Conv2D(filters=64, kernel_size=[5, 5], strides=[2, 2],
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                   input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(
            Conv2D(filters=128, kernel_size=[5, 5], strides=[2, 2],
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                   padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(
            Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2],
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                   padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(
            Conv2D(filters=512, kernel_size=[5, 5], strides=[1, 1],
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                   padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(
            Conv2D(filters=1024, kernel_size=[5, 5], strides=[2, 2],
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                   padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        model.add(Activation("sigmoid"))

        print("Discriminator")
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def get_batches(self, data, batch_size):
        batches = []
        for i in range(int(data.shape[0] // batch_size)):
            batch = data[i * batch_size:(i + 1) * batch_size]
            augmented_images = []
            for img in batch:
                image = Image.fromarray(img)
                if random.choice([True, False]):
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                augmented_images.append(np.asarray(image))
            batch = np.asarray(augmented_images)
            normalized_batch = (batch / 127.5) - 1.0
            batches.append(normalized_batch)
        return batches

    def train(self, epochs, batch_size=128):

        x_train = np.asarray([np.asarray(Image.open(file).resize((self.img_rows, self.img_cols))) for file in
                              glob(INPUT_DATA_DIR + '*')])

        print("Input: " + str(x_train.shape))

        valid = np.ones((batch_size, 1)) * random.uniform(0.9, 1.0)
        fake = np.zeros((batch_size, 1))

        epoch_n = 0
        d_losses = []
        g_losses = []

        for epoch in range(epochs):
            epoch_n += 1
            batch = 0
            for imgs in self.get_batches(x_train, batch_size):
                batch += 1
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                self.discriminator.trainable = True
                d_loss_real =  self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                self.discriminator.trainable = False
                g_loss = self.combined.train_on_batch(noise, valid)
                d_losses.append(d_loss)
                g_losses.append(g_loss)

            print("Epoch " + str(epoch_n) + " finished")
            self.save_imgs(epoch)
            plt.plot(d_losses, label='Discriminator', alpha=0.6)
            plt.plot(g_losses, label='Generator', alpha=0.6)
            plt.title("Losses")
            plt.legend()
            plt.savefig("images/" + "losses_" + str(epoch) + ".png")
            plt.close()

    def show_samples(self, sample_images, name, epoch):
        figure, axes = plt.subplots(1, len(sample_images), figsize=(128, 128))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = sample_images[index]
            axis.imshow(image_array)
            image = Image.fromarray(image_array)
            image.save(name + "simpsons_" + str(epoch) + "_" + str(index) + ".png")
        plt.close()

    def save_imgs(self, epoch):
        r = 5
        noise = np.random.uniform(-1, 1, (r, self.latent_dim))
        samples = self.generator.predict(noise)
        sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]
        self.show_samples(sample_images, "./images/", epoch)


EPSILON = 0.00005

INPUT_DATA_DIR = "C:/Users/tony/PycharmProjects/untitled/cropped/"

dcgan = DCGAN()
dcgan.train(epochs=500, batch_size=64)
