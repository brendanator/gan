import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn, layers
from utils import RunDirectories
import matplotlib.pyplot as plt
import matplotlib.cm as cm

mnist = learn.datasets.load_dataset('mnist')
num_real_images = len(mnist.train.images)

run_dirs = RunDirectories()

dtype = tf.float32
batch_size = 32
training_iterations = 1000000
dsteps = 1
gsteps = 2

with tf.variable_scope('generator') as scope:
  generator_input_shape = [None, 2]
  generator_input = tf.placeholder(dtype, generator_input_shape)
  generator_labels = tf.placeholder(dtype, shape=[None])
  generator_hidden_1 = layers.layer_norm(layers.relu(generator_input, 100))
  generator_hidden_2 = layers.layer_norm(layers.fully_connected(generator_hidden_1, 100, activation_fn=tf.sigmoid))
  generator_hidden_3 = layers.layer_norm(layers.relu(generator_hidden_2, 100))
  generator_image = layers.fully_connected(generator_hidden_1, 784, activation_fn=tf.sigmoid)

with tf.variable_scope('discriminator') as scope:
  discriminator_input_shape = [None, 784]
  discriminator_input = generator_image
  discriminator_labels = tf.placeholder(dtype, shape=[None])
  discriminator_hidden_layer_1 = layers.layer_norm(layers.fully_connected(discriminator_input, 100, tf.nn.relu))
  discriminator_hidden_layer_2 = layers.layer_norm(layers.fully_connected(discriminator_hidden_layer_1, 100, tf.nn.relu))
  discriminator_logits = tf.squeeze(layers.fully_connected(discriminator_hidden_layer_2, num_outputs=1, activation_fn=None))
  discriminator_output = tf.sigmoid(discriminator_logits)

generator_loss = tf.reduce_mean(tf.log(1 - discriminator_output))
generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
generator_train = tf.train.AdamOptimizer() \
                          .minimize(generator_loss, var_list=generator_variables)

discriminator_predict = tf.round(discriminator_output)
discriminator_accuracy = tf.reduce_mean(tf.to_float(tf.equal(discriminator_predict, discriminator_labels)))
discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(discriminator_logits, discriminator_labels))
discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
discriminator_train = tf.train.AdamOptimizer() \
                              .minimize(discriminator_loss, var_list=discriminator_variables)

generator_loss_summary = tf.scalar_summary('generator_loss', generator_loss)
generator_image_summary = tf.image_summary('generator_image', tf.reshape(generator_image, shape=[batch_size, 28, 28, 1]))
generator_summary = tf.merge_summary([generator_loss_summary, generator_image_summary])

discriminator_loss_summary = tf.scalar_summary('discriminator_loss', discriminator_loss)
discriminator_accuracy_summary = tf.scalar_summary('discriminator_accuracy', discriminator_accuracy)
discriminator_summary = tf.merge_summary([discriminator_loss_summary, discriminator_accuracy_summary])

fake_labels = np.zeros(shape=batch_size)
real_labels = np.ones(shape=batch_size)
all_labels = np.concatenate((fake_labels, real_labels))

with tf.Session() as session:
  tf.initialize_all_variables().run()

  saver = tf.train.Saver()
  writer = tf.train.SummaryWriter(run_dirs.summaries(), session.graph)

  for iteration in range(training_iterations):
    for dstep in range(dsteps):
      fake_images = session.run(generator_image,
                                {generator_input: np.random.normal(size=[batch_size, 2])})
      real_images = mnist.train.images[np.random.choice(num_real_images, batch_size)]

      all_images = np.concatenate((fake_images, real_images))
      _, summary = session.run([discriminator_train, discriminator_summary],
                               {discriminator_input: all_images, discriminator_labels: all_labels})
    writer.add_summary(summary, iteration)

    for gstep in range(gsteps):
      _, summary = session.run([generator_train, generator_summary],
                               {generator_input: np.random.normal(size=[batch_size, 2])})
    writer.add_summary(summary, iteration)

    if iteration % 10000 == 0:
      print('Evaluation at iteration %d' % iteration)
      fake_image = session.run(tf.reshape(generator_image, [28, 28]),
                               {generator_input: np.random.normal(size=[1, 2])})

      plt.imshow(fake_image, cmap=cm.Greys)
      plt.suptitle('iteration %d' % iteration)
      plt.savefig(run_dirs.images() + 'iteration-%d.svg' % iteration)

      saver.save(session, run_dirs.checkpoints() + 'model.ckpt', global_step=iteration)

saver.save(session, run_dirs.checkpoints() + 'final-model.ckpt')
