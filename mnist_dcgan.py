import os, sys, random, glob, h5py
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential, load_model
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras import initializations

K.set_image_dim_ordering('th')

# Deterministic
np.random.seed(1000)

# MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = np.concatenate((X_train, X_test), axis=0)
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]
# X_train = X_train.astype('float32')
# X_train /= 255

def initNormal(shape, name=None):
    return initializations.normal(shape, scale=0.02, name=name)

'''
# Convolutional Generator
nodes = 64
generator = Sequential()
# generator.add(Dense(1024, input_shape=(100,), activation='relu'))
# generator.add(Dense(nodes * 7 * 7, activation='relu'))
generator.add(Dense(nodes * 14 * 14, input_shape=(100,), activation='relu'))
# Upsample to (..., 14, 14)
generator.add(Reshape((nodes, 14, 14)))
# Upsample to (..., 28, 28)
generator.add(UpSampling2D(size=(2, 2)))
# generator.add(Convolution2D(nodes/2, 3, 3, border_mode='same', activation='relu', init='glorot_normal'))
generator.add(Convolution2D(nodes/4, 3, 3, border_mode='same', activation='relu', init='glorot_normal'))
generator.add(Convolution2D(1, 2, 2, border_mode='same', activation='sigmoid', init='glorot_normal'))
generator.compile(loss='binary_crossentropy', optimizer='adam')

# Build discriminator model
nodes = 32
discriminator = Sequential()
discriminator.add(Convolution2D(nodes, 3, 3, input_shape=(1, 28, 28), border_mode='same', subsample=(2, 2)))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))
discriminator.add(Convolution2D(nodes*2, 3, 3, border_mode='same', subsample=(1, 1)))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))
discriminator.add(Convolution2D(nodes*4, 3, 3, border_mode='same', subsample=(2, 2)))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.3))
# discriminator.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
# discriminator.add(LeakyReLU())
# discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

# Combined network
gan = Sequential()
gan.add(generator)
discriminator.trainable = False
gan.add(discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')
'''

adam = Adam(lr=0.0002, beta_1=0.5)
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=100, init=initNormal))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 7, 7)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(64, 5, 5, border_mode='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(1, 5, 5, border_mode='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator = Sequential()
discriminator.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2, 2), input_shape=(1, 28, 28), init=initNormal))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(2, 2)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(100,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(input=ganInput, output=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

'''
# Pretrain discriminator
count = 10000
# Get a random set of input noise and models
# noise = np.random.normal(0, 1, size=[count, 10])
# noise = np.random.normal(0, 1, size=[count, 100])
noise = np.random.uniform(-1, 1, size=[count, 100])
imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=count)]

# Generate random models
generatedImages = generator.predict(noise)
X = np.concatenate([imageBatch, generatedImages])
y = np.zeros(2*count)
# y[:count] = 0.9
y[:count] = 1

# Train discriminator
discriminator.trainable = True
discriminator.fit(X, y, nb_epoch=1, batch_size=32)
# discriminator.fit(imageBatch, y[:count], nb_epoch=1, batch_size=32)
# discriminator.fit(generatedImages, y[count:], nb_epoch=1, batch_size=32)
print 'Discriminator pretraining complete'
'''

dLoss = []
gLoss = []

def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLoss, label='Discriminitive loss')
    plt.plot(gLoss, label='Generative loss')
    plt.legend()
    plt.savefig('images/loss_epoch_%d.png' % epoch)

def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    # noise = np.random.normal(0, 1, size=[examples, 10])
    noise = np.random.normal(0, 1, size=[examples, 100])
    # noise = np.random.uniform(-1, 1, size=[examples, 100])
    generatedImages = generator.predict(noise)
    # Add a row of real images
    # generatedImages = np.concatenate([generatedImages, X_train[np.random.randint(0, X_train.shape[0], size=10)]])

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        # plt.imshow(generatedImages[i, 0], cmap='gray')
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/generated_image_epoch_%d.png' % epoch)

def saveModels(epoch):
    generator.save('models/generator_epoch_%d.h5' % epoch)
    discriminator.save('models/discriminator_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] / batchSize
    print 'Epochs:', epochs
    print 'Batch size:', batchSize
    print 'Batches per epoch:', batchCount

    for e in xrange(1, epochs+1):
        print '-'*15, 'Epoch %d' % e, '-'*15
        for _ in tqdm(xrange(batchCount)):
            # Get a random set of input noise and images
            # noise = np.random.normal(0, 1, size=[batchSize, 10])
            noise = np.random.normal(0, 1, size=[batchSize, 100])
            # noise = np.random.uniform(-1, 1, size=[batchSize, 100])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate models
            generatedImages = generator.predict(noise)
            # generatedModels += np.random.uniform(0, 0.1, size=generatedModels.shape)
            X = np.concatenate([imageBatch, generatedImages])

            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9
            # yDis[:batchSize] = 1
            X, yDis = shuffle(X, yDis)

            # Train discriminator
            discriminator.trainable = True
            dLoss.append(discriminator.train_on_batch(X, yDis))
            # discriminator.train_on_batch(imageBatch, yDis[:batchSize])
            # dLoss.append(discriminator.train_on_batch(generatedImages, yDis[batchSize:]))

            # Train full GAN
            # noise = np.random.normal(0, 1, size=[batchSize, 10])
            noise = np.random.normal(0, 1, size=[batchSize, 100])
            yGen = np.ones(batchSize)
            # noise = np.random.uniform(-1, 1, size=[batchSize, 100])
            discriminator.trainable = False
            gLoss.append(gan.train_on_batch(noise, yGen))

        if e == 1 or e % 5 == 0:
            plotLoss(e)
            plotGeneratedImages(e)
            saveModels(e)

if __name__ == '__main__':
    train(10, 128)

