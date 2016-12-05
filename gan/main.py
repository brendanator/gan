import tensorflow as tf
from tensorflow.contrib import learn
from gan import GenerativeAdversarialNetwork
from utils import RunDirectories

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', '[train evaluate]')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('noise_size', 2, 'Noise size')
flags.DEFINE_integer('steps', 100000, 'Maximum training steps for generator and discriminator combined')
flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate for AdamOptimizer')
flags.DEFINE_float('beta1', 0.5, 'beta1 for AdamOptimizer')
flags.DEFINE_string('run_dirs', 'runs', 'Directory to save/load data for all runs')
flags.DEFINE_string('restore_run', None, 'Run to restore: num or latest')
FLAGS = flags.FLAGS

def main(_):
  with tf.Session() as session:
    image_shape = [28, 28, 1]
    copy_source = FLAGS.mode == 'train'
    run_dirs = RunDirectories(FLAGS.run_dirs, copy_source)
    gan = GenerativeAdversarialNetwork(FLAGS.noise_size, image_shape, tf.float32, run_dirs)

    if FLAGS.mode == 'train':
      images = learn.datasets.load_dataset('mnist').train.images
      gan.train(session, images, FLAGS.steps, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.beta1, FLAGS.restore_run)
    elif FLAGS.mode == 'evaluate':
      gan.restore(session, FLAGS.restore_run)
      gan.show_generated_images(session, 10)
    else:
      raise 'Unknown mode %s' % FLAGS.mode

if __name__ == '__main__':
  tf.app.run()
