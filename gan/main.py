import tensorflow as tf
from tensorflow.contrib import learn
from gan import GenerativeAdversarialNetwork

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('noise_size', 2, 'Noise size')
flags.DEFINE_integer('steps', 100000, 'Maximum training steps')
flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate for AdamOptimizer')
flags.DEFINE_float('beta1', 0.5, 'beta1 for AdamOptimizer')
FLAGS = flags.FLAGS

def main(_):
  with tf.Session() as session:
    images = learn.datasets.load_dataset('mnist').train.images
    image_shape = [28, 28, 1]

    gan = GenerativeAdversarialNetwork(FLAGS.noise_size, image_shape, dtype=tf.float32)
    gan.train(session, images, FLAGS.steps, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.beta1)

if __name__ == '__main__':
  tf.app.run()
