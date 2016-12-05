import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from utils import RunDirectories
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GenerativeAdversarialNetwork():
  def __init__(self, noise_size, image_shape, dtype, run_dirs):
    self.noise_size = noise_size
    self.run_dirs = run_dirs
    self.build_model(image_shape, dtype)

  def build_model(self, image_size, dtype):
    self.generator_input = tf.placeholder(dtype, [None, self.noise_size])
    self.generator_labels = tf.placeholder(dtype, shape=[None])
    self.generator_output, self.generator_image, self.generator_activations = self.generator(self.generator_input, image_size)

    self.discriminator_input = self.generator_output
    self.discriminator_labels = tf.placeholder(dtype, shape=[None])
    self.discriminator_logits, self.discriminator_output, self.discriminator_activations = self.discriminator(self.generator_output)

  def generator(self, z, image_shape):
    with tf.variable_scope('generator') as scope:
      hidden_1 = layers.layer_norm(layers.relu(z, 100))
      hidden_2 = layers.layer_norm(layers.fully_connected(hidden_1, 100, activation_fn=tf.sigmoid))
      hidden_3 = layers.layer_norm(layers.relu(hidden_2, 100))
      image_size = int(np.prod(image_shape))
      output = layers.fully_connected(hidden_3, image_size, activation_fn=tf.sigmoid)
      image = tf.reshape(output, [-1] + image_shape, name='image')
      activations = [hidden_1, hidden_2, hidden_3, output]
    return output, image, activations

  def discriminator(self, image):
    with tf.variable_scope('discriminator') as scope:
      hidden_1 = layers.layer_norm(layers.fully_connected(image, 100, tf.nn.relu))
      hidden_2 = layers.layer_norm(layers.fully_connected(hidden_1, 100, tf.nn.relu))
      logits = tf.squeeze(layers.fully_connected(hidden_2, num_outputs=1, activation_fn=None), name='logits')
      output = tf.sigmoid(logits, name='output')
      activations = [hidden_1, hidden_2, output]
    return logits, output, activations

  def train(self, session, images, steps, batch_size, learning_rate, beta1, restore_run):
    generator_loss = tf.reduce_mean(tf.log(1 - self.discriminator_output))
    generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    generator_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
    generator_gradients = generator_optimizer.compute_gradients(generator_loss, var_list=generator_variables)
    generator_train = generator_optimizer.apply_gradients(generator_gradients)

    discriminator_accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.round(self.discriminator_output), self.discriminator_labels)))
    discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.discriminator_logits, self.discriminator_labels))
    discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
    discriminator_gradients = discriminator_optimizer.compute_gradients(discriminator_loss, var_list=discriminator_variables)
    discriminator_train = discriminator_optimizer.apply_gradients(discriminator_gradients)

    generator_loss_summary = tf.scalar_summary('generator_loss', generator_loss)
    generator_image_summary = tf.image_summary('generator_image', self.generator_image)
    generator_activation_summaries = [tf.histogram_summary(act.op.name + '/activation', act) for act in self.generator_activations]
    generator_gradient_summaries = [tf.histogram_summary(var.op.name + '/gradient', grad) for grad, var in generator_gradients]
    generator_summary = tf.merge_summary([generator_loss_summary, generator_image_summary] + generator_activation_summaries + generator_gradient_summaries)

    discriminator_loss_summary = tf.scalar_summary('discriminator_loss', discriminator_loss)
    discriminator_accuracy_summary = tf.scalar_summary('discriminator_accuracy', discriminator_accuracy)
    discriminator_activation_summaries = [tf.histogram_summary(act.op.name + '/activation', act) for act in self.discriminator_activations]
    discriminator_gradient_summaries = [tf.histogram_summary(var.op.name + '/gradient', grad) for grad, var in discriminator_gradients]
    discriminator_summary = tf.merge_summary([discriminator_loss_summary, discriminator_accuracy_summary] + discriminator_activation_summaries + discriminator_gradient_summaries)

    tf.initialize_all_variables().run()

    if restore_run:
      self.restore(session, restore_run)

    saver = tf.train.Saver()
    writer = tf.train.SummaryWriter(self.run_dirs.summaries(), session.graph)

    fake_labels = np.zeros(shape=batch_size)
    real_labels = np.ones(shape=batch_size)
    all_labels = np.concatenate((fake_labels, real_labels))

    step, output_step = 0, 0
    while step < steps:
      start_step, accuracy = step, 0.0
      while step <= start_step+10 and accuracy < 0.4:
        step += 1
        fake_images = session.run(self.generator_output, {self.generator_input: self.generator_noise(batch_size)})
        real_images = self.random_real_images(images, batch_size)
        all_images = np.concatenate((fake_images, real_images))

        _, accuracy, summary = session.run([discriminator_train, discriminator_accuracy, discriminator_summary],
                                {self.discriminator_input: all_images, self.discriminator_labels: all_labels})
      writer.add_summary(summary, step)

      start_step, accuracy = step, 1.0
      while step <= start_step+10 and accuracy > 0.6:
        step += 1
        _, accuracy, summary = session.run([generator_train, discriminator_accuracy, generator_summary],
                                {self.generator_input: self.generator_noise(batch_size), self.discriminator_labels: fake_labels})
      writer.add_summary(summary, step)

      if step >= output_step+100:
        output_step = step - (step%100)
        print('Outputting image at step %d' % output_step)
        fake_image = session.run(tf.squeeze(self.generator_image), {self.generator_input: self.generator_noise(1)})

        plt.suptitle('step %d' % output_step)
        plt.imshow(fake_image, cmap=cm.Greys)
        plt.savefig(self.run_dirs.images() + 'step-%06d.svg' % output_step)

        if output_step % 1000 == 0:
          saver.save(session, self.run_dirs.checkpoints() + 'model.ckpt', global_step=output_step)

    saver.save(session, self.run_dirs.checkpoints() + 'model.ckpt')

  def generator_noise(self, batch_size):
    return np.random.uniform(size=[batch_size, self.noise_size])

  def random_real_images(self, images, batch_size):
    num_real_images = len(images)
    return images[np.random.choice(num_real_images, batch_size)]

  def restore(self, session, restore_run):
    latest_checkpoint_dir = self.run_dirs.latest_checkpoint(restore_run)
    checkpoint = tf.train.get_checkpoint_state(latest_checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
      tf.train.Saver().restore(session, checkpoint.model_checkpoint_path)

  def show_generated_images(self, session, num_images):
    images = session.run(tf.squeeze(self.generator_image), {self.generator_input: self.generator_noise(num_images)})
    for i, image in enumerate(images):
      plt.figure(i)
      plt.imshow(image, cmap=cm.Greys)
    plt.show()
