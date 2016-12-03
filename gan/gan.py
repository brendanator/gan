import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from utils import RunDirectories
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GenerativeAdversarialNetwork():
  def __init__(self, noise_size, image_shape, dtype):
    self.noise_size = noise_size
    self.run_dirs = RunDirectories()
    self.build_model(image_shape, dtype)

  def build_model(self, image_size, dtype):
    self.generator_input = tf.placeholder(dtype, [None, self.noise_size])
    self.generator_labels = tf.placeholder(dtype, shape=[None])
    self.generator_output, self.generator_image = self.generator(self.generator_input, image_size)

    self.discriminator_input = self.generator_output
    self.discriminator_labels = tf.placeholder(dtype, shape=[None])
    self.discriminator_logits, self.discriminator_output = self.discriminator(self.generator_output)

  def generator(self, z, image_shape):
    with tf.variable_scope('generator') as scope:
      hidden_1 = layers.layer_norm(layers.relu(z, 100))
      hidden_2 = layers.layer_norm(layers.fully_connected(hidden_1, 100, activation_fn=tf.sigmoid))
      hidden_3 = layers.layer_norm(layers.relu(hidden_2, 100))
      image_size = int(np.prod(image_shape))
      generator_output = layers.fully_connected(hidden_3, image_size, activation_fn=tf.sigmoid)
      generator_image = tf.reshape(generator_output, [-1] + image_shape)
    return generator_output, generator_image

  def discriminator(self, image):
    with tf.variable_scope('discriminator') as scope:
      hidden_layer_1 = layers.layer_norm(layers.fully_connected(image, 100, tf.nn.relu))
      hidden_layer_2 = layers.layer_norm(layers.fully_connected(hidden_layer_1, 100, tf.nn.relu))
      logits = tf.squeeze(layers.fully_connected(hidden_layer_2, num_outputs=1, activation_fn=None))
    return logits, tf.sigmoid(logits)

  def train(self, session, images, epochs, batch_size):
    generator_loss = tf.reduce_mean(tf.log(1 - self.discriminator_output))
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    generator_train = tf.train.AdamOptimizer() \
                                   .minimize(generator_loss, var_list=generator_variables)

    discriminator_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.round(self.discriminator_output), self.discriminator_labels)))
    discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.discriminator_logits, self.discriminator_labels))
    discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    discriminator_train = tf.train.AdamOptimizer() \
                                  .minimize(discriminator_loss, var_list=discriminator_variables)

    generator_loss_summary = tf.scalar_summary('generator_loss', generator_loss)
    generator_image_summary = tf.image_summary('generator_image', self.generator_image)
    generator_summary = tf.merge_summary([generator_loss_summary, generator_image_summary])

    discriminator_loss_summary = tf.scalar_summary('discriminator_loss', discriminator_loss)
    discriminator_accuracy_summary = tf.scalar_summary('discriminator_accuracy', discriminator_accuracy)
    discriminator_summary = tf.merge_summary([discriminator_loss_summary, discriminator_accuracy_summary])

    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    writer = tf.train.SummaryWriter(self.run_dirs.summaries(), session.graph)

    fake_labels = np.zeros(shape=batch_size)
    real_labels = np.ones(shape=batch_size)
    all_labels = np.concatenate((fake_labels, real_labels))

    for epoch in range(epochs):
      accuracy = 0.0
      while accuracy < 0.4:
        fake_images = session.run(self.generator_output, {self.generator_input: self.generator_noise(batch_size)})
        real_images = self.random_real_images(images, batch_size)
        all_images = np.concatenate((fake_images, real_images))

        _, accuracy, summary = session.run([discriminator_train, discriminator_accuracy, discriminator_summary],
                                {self.discriminator_input: all_images, self.discriminator_labels: all_labels})
      writer.add_summary(summary, epoch)

      accuracy = 1.0
      while accuracy > 0.6:
        _, accuracy, summary = session.run([generator_train, discriminator_accuracy, generator_summary],
                                {self.generator_input: self.generator_noise(batch_size), self.discriminator_labels: fake_labels})
      writer.add_summary(summary, epoch)

      if epoch % 1000 == 0:
        print('Outputting image at epoch %d' % epoch)
        fake_image = session.run(tf.squeeze(self.generator_image), {self.generator_input: self.generator_noise(1)})

        plt.imshow(fake_image, cmap=cm.Greys)
        plt.suptitle('epoch %d' % epoch)
        plt.savefig(self.run_dirs.images() + 'epoch-%d.svg' % epoch)

        saver.save(session, self.run_dirs.checkpoints() + 'model.ckpt', global_step=epoch)

    saver.save(session, self.run_dirs.checkpoints() + 'final-model.ckpt')

  def generator_noise(self, batch_size):
    return np.random.uniform(size=[batch_size, self.noise_size])

  def random_real_images(self, images, batch_size):
    num_real_images = len(images)
    return images[np.random.choice(num_real_images, batch_size)]
