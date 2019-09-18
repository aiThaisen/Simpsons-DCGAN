from __future__ import print_function, division

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal, TruncatedNormal
import random
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from numpy.random import randn


import sys

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(8 * 8 * 1024, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 1024)))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=512, kernel_size=[5,5], strides=[2,2], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=256, kernel_size=[5,5], strides=[2,2], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=128, kernel_size=[5,5], strides=[2,2], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=64, kernel_size=[5,5], strides=[2,2], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(filters=self.channels, kernel_size=[5,5], strides=[1,1], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(Activation("tanh"))

        print("Generator")
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=[5,5], strides=[2,2], kernel_initializer=TruncatedNormal(stddev=0.02), input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=128, kernel_size=[5,5], strides=[2,2], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=256, kernel_size=[5,5], strides=[2,2], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=512, kernel_size=[5, 5], strides=[1, 1], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=1024, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=TruncatedNormal(stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=EPSILON))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Reshape((-1, 8 * 8 * 1024)))

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

        x_train = np.asarray([np.asarray(Image.open(file).resize((self.img_rows, self.img_cols))) for file in glob(INPUT_DATA_DIR + '*')])

        print("Input: " + str(x_train.shape))

        valid = np.ones((batch_size, 1)) * random.uniform(0.9, 1.0)
        fake = np.zeros((batch_size, 1))

        epoch_n = 0

        for epoch in range(epochs):
            epoch_n += 1
            batch = 0
            for imgs in self.get_batches(x_train, batch_size):
                batch +=1
                # Sample noise and generate a batch of new images
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                self.discriminator.trainable = True
                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                self.discriminator.trainable = False
                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print("Batch " + str(batch) + " processed")
                # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            print("Epoch " + str(epoch_n) + " finished")
            self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.uniform(-1, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/simpsons_%d.png" % epoch)
        plt.close()


INPUT_DATA_DIR = "/Users/edwardhyde/PycharmProjects/gan/cropped/"
EPSILON = 0.00005

dcgan = DCGAN()
dcgan.train(epochs=300, batch_size=64)