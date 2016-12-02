import tensorflow as tf
from tensorflow.contrib import learn
from gan import GenerativeAdversarialNetwork

def main(_):
  with tf.Session() as session:
    batch_size = 32
    noise_size = 1
    dtype = tf.float32
    training_iterations = 1000000
    mnist = learn.datasets.load_dataset('mnist')

    gan = GenerativeAdversarialNetwork(mnist.train.images, batch_size, noise_size, dtype)
    gan.train(session, training_iterations)

if __name__ == '__main__':
  tf.app.run()
