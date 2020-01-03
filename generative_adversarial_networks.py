import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

# this follows the tutorial at https://www.datacamp.com/community/tutorials/generative-adversarial-networks

# lets keras know we are using tensorflow as backend
os.environ['KERAS_BACKEND'] = "tensorflow"

# this is set constant for reproducible results
np.random.seed(10)

# dimension of the random noise vector
random_dim = 100

def load_data():

    # loads dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize to range [-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    #convert x_train with shape (60000, 28, 28) to (60000, 784) so there's 784 columns per row
    x_train = x_train.reshape(60000, 784)

    return x_train, y_train, x_test, y_test

# Set Adam as the optimizer
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):

    # selects keras Sequential API which builds model layer-by-layer. Functional() is more flexible.
    gen = Sequential()

    # creates a new Dense layer neural network.
    gen.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

    # creates a new leaky ReLU, which means a small gradient can exist when unit is not active.
    gen.add(LeakyReLU(0.2))

    # does the same as above, but in 512 and 1024 sizes
    gen.add(Dense(512))
    gen.add(LeakyReLU(0.2))

    gen.add(Dense(1024))
    gen.add(LeakyReLU(0.2))

    # steps the size down a little bit and adds the activation function
    gen.add(Dense(784, activation='tanh'))

    # configures the model for training
    gen.compile(loss='binary_crossentropy', optimizer=optimizer)

    return gen

def get_discriminator(optimizer):

    # Selects keras Sequential API
    disc = Sequential()

    # adds a 1024 size Dense neural net with 784 inputs
    disc.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

    # adds leakyReLU
    disc.add(LeakyReLU(0.2))

    # adds Dropout, which sets 30% of inputs to zero to prevent overfitting
    disc.add(Dropout(0.3))

    # does the same as above but steps it down to 512 and 256 sizes
    disc.add(Dense(512))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.3))

    disc.add(Dense(256))
    disc.add(LeakyReLU(0.2))
    disc.add(Dropout(0.3))

    # steps the size down to 1 and adds the activation function
    disc.add(Dense(1, activation='sigmoid'))

    # configures the model for training
    disc.compile(loss='binary_crossentropy', optimizer=optimizer)

    return disc

def get_gan_network(disc, random_dim, gen, optimizer):

    # turn off training disc since we want to train only 1 at a time
    disc.trainable = False

    # gan input, which is noise, will be 100 dimensional vector
    gan_input = Input(shape=(random_dim,))

    # get the output of the generator (an image)
    x = gen(gan_input)

    # get the output of the disc, which is a probability of if the image is real or not
    gan_output = disc(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return gan


# This function is copied verbatim, as it relates to plotting and not the GAN itself
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
    plt.close()

def train(epochs = 1, batch_size=128):

    # get the training and testing data
    x_train, y_train, x_test, y_test = load_data()

    # split the training into batches of 128
    batch_count = x_train.shape[0] // batch_size

    # create all the elements
    Adam = get_optimizer()
    gen = get_generator(Adam)
    disc = get_discriminator(Adam)
    gan = get_gan_network(disc, random_dim, gen, Adam)

    # loop through once for each epoch
    for e in range(1, epochs+1):

        # print the current epoch
        print('-'*15, 'Epoch %d' % e, '-'*15)

        # this is to loop through each batch of 128
        for _ in tqdm(range(batch_count)):

            # get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # generate fake MNIST images
            gend_images = gen.predict(noise)
            X = np.concatenate([image_batch, gend_images])

            # Label stuff for graphing
            y_dis = np.zeros(2*batch_size)

            # one sided label smoothing
            y_dis[:batch_size] = 0.9

            # train the descriminator
            disc.trainable = True
            disc.train_on_batch(X, y_dis)

            # train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            disc.trainable = False
            gan.train_on_batch(noise, y_gen)

            # plot generated images
            if e == 1 or e % 2 == 0:
                plot_generated_images(e, gen)

train(2,128)








