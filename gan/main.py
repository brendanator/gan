import tensorflow as tf
from tensorflow.contrib import learn
from gan import GenerativeAdversarialNetwork

def main(_):
  with tf.Session() as session:
    batch_size = 32
    noise_size = 1
    dtype = tf.float32
    epochs = 1000000
    images = learn.datasets.load_dataset('mnist').train.images
    image_shape = [28, 28, 1]

    gan = GenerativeAdversarialNetwork(noise_size, image_shape, dtype)
    gan.train(session, images, epochs, batch_size)

if __name__ == '__main__':
  tf.app.run()
