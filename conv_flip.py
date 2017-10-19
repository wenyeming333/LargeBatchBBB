from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.examples.tutorials.mnist import input_data
import tabular_logger as tlogger
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import os

def normal(x, mu, sigma):
  pi = tf.constant(np.pi)
  Z = (2*pi*sigma**2)**0.5
  return tf.exp(-0.5*(x - mu)**2 / sigma**2) / Z

def log_normal(x, mu, sigma):
  pi = tf.constant(np.pi)
  return -0.5*tf.log(2*pi) - tf.log(sigma) - 0.5*tf.square(x-mu) / (sigma**2)

def log_prior(x, isScaled=True):
  if isScaled:
    sigma1 = tf.exp(-1.0)
    sigma2 = tf.exp(-7.0)
    pi = 0.5
    return tf.log(pi*normal(x,0.0,sigma1)+(1-pi)*normal(x,0.0,sigma2))
  else:
    sigma = tf.exp(-1.0)
    return log_normal(x, 0.0, sigma)

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class bayes():
  def __init__(self, args):
    self.args = args
    self.data_dir = args.data_dir
    self.mnist = input_data.read_data_sets(self.data_dir,one_hot=True,fake_data=False)
    self.batch_size = args.batch_size
    self.log_dir = args.log_dir
    self.model = 'models'
    self.scale = args.scale
    self.w_prior_std = 1.0
    self.isFlip = args.isFlip
    self.LRT = args.LRT
    if args.lr_decay:
      self.learning_rate = tf.Variable(0.0, trainable=False)
    else:
      self.learning_rate = args.learning_rate

  def bayesian_nn_layer(self, input_tensor, input_dim, output_dim, isTrain, layer_name, nonlinearity=None):
    eps = 1e-35
    with tf.name_scope(layer_name):
      # Initialize the variational parameters. Reference: weight uncertainty in neural networks.
      with tf.name_scope('weights_mean'):
        mu_w = tf.Variable(tf.random_normal([input_dim, output_dim], stddev=0.1))
      with tf.name_scope('weights_sd'):
        rho_w = tf.Variable(tf.random_normal([input_dim, output_dim], mean=-3.0, stddev=0.1))
        sigma_w = tf.log(1+tf.exp(rho_w))
      with tf.name_scope('bias_mean'):
        biases = tf.Variable(tf.zeros([output_dim]))

      def closed_form_kl():
        dim = input_dim * output_dim
        return (tf.log(self.w_prior_std)*dim - \
          tf.reduce_sum(tf.log(sigma_w+eps)) + \
          0.5*(-dim+1.0/(self.w_prior_std**2)*(tf.reduce_sum(sigma_w**2) +\
          tf.reduce_sum(mu_w**2))))

      def train_forward():
        epsilon_w = tf.random_normal([input_dim, output_dim], stddev=1.0)
        weights = mu_w + tf.multiply(sigma_w, epsilon_w)
        with tf.name_scope('Wx_plus_b'):
          preactivate = tf.matmul(input_tensor, weights) + biases
        if nonlinearity is not None:
          preactivate = nonlinearity(preactivate)
        return preactivate, closed_form_kl()

      def map_inference():
        weights = mu_w
        # biases = mu_b
        with tf.name_scope('Wx_plus_b'):
          preactivate = tf.matmul(input_tensor, weights) + biases
        if nonlinearity is not None:
          preactivate = nonlinearity(preactivate)
        return preactivate, closed_form_kl()
      return tf.cond(isTrain, lambda: train_forward(), lambda: map_inference())

  # Only be called during training.
  def flipoutFC(self, x, W_0, delta_W):
    weight_dim = W_0.shape.as_list()
    # x is n*m where m is the dimension, n is the mini-batch size.
    # W_0 is m*h where h is the num of hidden units.
    epsilon = tf.random_normal(weight_dim, stddev=1.0)
    def generate_flipping_factor(dim):
      shape = tf.stack([tf.shape(x)[0], dim])
      random = tf.random_normal(shape)
      positives = tf.ones(shape)
      negatives = tf.zeros(shape)-1
      return tf.where(random>0, positives, negatives)
    E1 = generate_flipping_factor(weight_dim[1])
    E2 = generate_flipping_factor(weight_dim[0])
    pert_x = tf.multiply(tf.matmul(tf.multiply(x, E2), delta_W), E1)
    return (tf.matmul(x, W_0) + pert_x)

  def bayesian_nn_layer_flip(self, input_tensor, input_dim, output_dim, isTrain, layer_name, nonlinearity=None):
    eps = 1e-35
    with tf.name_scope(layer_name):
      # Initialize the variational parameters. Reference: weight uncertainty in neural networks.
      with tf.name_scope('weights_mean'):
        mu_w = tf.Variable(tf.random_normal([input_dim, output_dim], stddev=0.1))
      with tf.name_scope('weights_sd'):
        rho_w = tf.Variable(tf.random_normal([input_dim, output_dim], mean=-3.0, stddev=0.1))
        sigma_w = tf.log(1+tf.exp(rho_w))

      with tf.name_scope('bias_mean'):
        biases = tf.Variable(tf.zeros([output_dim]))

      def closed_form_kl():
        dim = input_dim * output_dim
        return (tf.log(self.w_prior_std)*dim - \
          tf.reduce_sum(tf.log(sigma_w+eps)) + \
          0.5*(-dim+1.0/(self.w_prior_std**2)*(tf.reduce_sum(sigma_w**2) +\
          tf.reduce_sum(mu_w**2))))

      def train_forward():
        with tf.name_scope('perturbation'):
          epsilon_w = tf.random_normal([input_dim, output_dim], stddev=1.0)
          delta_W = tf.multiply(sigma_w, epsilon_w)
        with tf.name_scope('flipout'):
          flipoutFC = self.flipoutFC(input_tensor, mu_w, delta_W)
        with tf.name_scope('Wx_plus_b'):
          preactivate = flipoutFC + biases
        if nonlinearity is not None:
          preactivate = nonlinearity(preactivate)
        return preactivate, closed_form_kl()

      def map_inference():
        weights = mu_w
        with tf.name_scope('Wx_plus_b'):
          preactivate = tf.matmul(input_tensor, weights) + biases
        if nonlinearity is not None:
          preactivate = nonlinearity(preactivate)
        return preactivate, closed_form_kl()
      return tf.cond(isTrain, lambda: train_forward(), lambda: map_inference())


  def conv2dFlipout(self, x, W_0, delta_W, weight_shape):
    num_channels = weight_shape[2]
    num_filters = weight_shape[3]
    conv = tf.nn.conv2d(x, W_0, strides=[1, 1, 1, 1], padding='SAME')

    def generate_flipping_factor(dim):
      shape = tf.stack([tf.shape(x)[0], dim])
      random = tf.random_normal(shape)
      positives = tf.ones(shape)
      negatives = tf.zeros(shape)-1
      return tf.where(random>0, positives, negatives)

    E1 = generate_flipping_factor(num_filters)
    E2 = generate_flipping_factor(num_channels)

    E1 = tf.reshape(E1, [-1, 1, 1, num_filters])
    E2 = tf.reshape(E2, [-1, 1, 1, num_channels])

    flip_x = tf.multiply(x, tf.tile(E2, [1, tf.shape(x)[1], tf.shape(x)[2], 1]))
    pert_conv = tf.multiply(tf.nn.conv2d(flip_x, delta_W, strides=\
      [1, 1, 1, 1], padding='SAME'), tf.tile(E1, [1, tf.shape(conv)[1], \
      tf.shape(conv)[2], 1]))
    return pert_conv + conv

  def bayesian_conv_layer_flip(self, inputs, weight_shape, isTrain, layer_name, nonlinearity=tf.nn.relu):
    eps = 1e-35
    with tf.variable_scope(layer_name):
      mu_w = tf.get_variable('weights_mean', shape=weight_shape, initializer=initializers.xavier_initializer())
      rho_w = tf.get_variable('weights_rho', shape=weight_shape, initializer=tf.random_normal_initializer(-3.0, 0.05))
      biases = tf.get_variable('biases', initializer=tf.zeros([weight_shape[-1]]))
      with tf.name_scope('weights_std'):
        sigma_w = tf.log(1+tf.exp(rho_w))

      def closed_form_kl():
        dim = tf.cast(tf.size(mu_w), tf.float32)
        return (tf.log(self.w_prior_std)*dim - \
          tf.reduce_sum(tf.log(sigma_w+eps)) + \
          0.5*(-dim+1.0/(self.w_prior_std**2)*(tf.reduce_sum(sigma_w**2) +\
          tf.reduce_sum(mu_w**2))))

      def train_forward():
        epsilon_w = tf.random_normal(weight_shape, stddev=1.0)
        delta_W = tf.multiply(sigma_w, epsilon_w)
        preactivate = self.conv2dFlipout(inputs, mu_w, delta_W, weight_shape) + biases
        res = nonlinearity(preactivate)
        return res, closed_form_kl()

      def map_inference():
        preactivate = tf.nn.conv2d(inputs, mu_w, strides=[1, 1, 1, 1], padding='SAME') + biases
        res = nonlinearity(preactivate)
        return res, closed_form_kl()
      return tf.cond(isTrain, lambda: train_forward(), lambda: map_inference())


  def bayesian_conv_layer(self, inputs, weight_shape, isTrain, layer_name, nonlinearity=tf.nn.relu):
    eps = 1e-35
    with tf.variable_scope(layer_name):
      mu_w = tf.get_variable('weights_mean', shape=weight_shape, initializer=initializers.xavier_initializer())
      rho_w = tf.get_variable('weights_rho', shape=weight_shape, initializer=tf.random_normal_initializer(-3.0, 0.05))
      biases = tf.get_variable('biases', initializer=tf.zeros([weight_shape[-1]]))
      with tf.name_scope('weights_std'):
        sigma_w = tf.log(1+tf.exp(rho_w))

      def closed_form_kl():
        dim = tf.cast(tf.size(mu_w), tf.float32)
        return (tf.log(self.w_prior_std)*dim - \
          tf.reduce_sum(tf.log(sigma_w+eps)) + \
          0.5*(-dim+1.0/(self.w_prior_std**2)*(tf.reduce_sum(sigma_w**2) +\
          tf.reduce_sum(mu_w**2))))

      def train_forward():
        epsilon_w = tf.random_normal(weight_shape, stddev=1.0)
        delta_W = tf.multiply(sigma_w, epsilon_w)
        preactivate = tf.nn.conv2d(inputs, mu_w+delta_W, strides=[1, 1, 1, 1], padding='SAME') + biases
        res = nonlinearity(preactivate)
        return res, closed_form_kl()

      def map_inference():
        preactivate = tf.nn.conv2d(inputs, mu_w, strides=[1, 1, 1, 1], padding='SAME') + biases
        res = nonlinearity(preactivate)
        return res, closed_form_kl()
      return tf.cond(isTrain, lambda: train_forward(), lambda: map_inference())


  def build_model(self):
    self.isTrain = tf.placeholder(tf.bool, name='isTrain')
    with tf.name_scope('input'):
      self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
      self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
      self.M = tf.placeholder(tf.float32, shape=(), name='number_mini_batches')
      self.n = tf.placeholder(tf.float32, shape=(), name='mini_batch_size')

    x = tf.reshape(self.x, [-1, 28, 28, 1])
    
    if self.isFlip:
      x, kl1 = self.bayesian_conv_layer_flip(x, [5,5,1,32], self.isTrain, layer_name='conv1')
      x = max_pool_2x2(x)
      x, kl2 = self.bayesian_conv_layer_flip(x, [5,5,32,64], self.isTrain, layer_name='conv2')
      x = max_pool_2x2(x)
      x = tf.reshape(x, [-1, 7*7*64])
      x, kl3 = self.bayesian_nn_layer_flip(x, 7*7*64, 1024, self.isTrain, 'fc1', nonlinearity=tf.nn.relu)
      y, kl4 = self.bayesian_nn_layer_flip(x, 1024, 10, self.isTrain, 'fc2')
    else:
      x, kl1 = self.bayesian_conv_layer(x, [5,5,1,32], self.isTrain, layer_name='conv1')
      x = max_pool_2x2(x)
      x, kl2 = self.bayesian_conv_layer(x, [5,5,32,64], self.isTrain, layer_name='conv2')
      x = max_pool_2x2(x)
      x = tf.reshape(x, [-1, 7*7*64])
      x, kl3 = self.bayesian_nn_layer(x , 7*7*64, 1024, self.isTrain, 'fc1', nonlinearity=tf.nn.relu)
      y, kl4 = self.bayesian_nn_layer(x, 1024, 10, self.isTrain, 'fc2')

    with tf.name_scope('cross-entropy'):
      self.cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y))

    with tf.name_scope('KL'):
      self.KL = (kl1 + kl2 + kl3 + kl4) / self.M
      # tf.summary.scalar('KL', self.KL)

    with tf.name_scope('loss'):
      self.loss = self.scale * self.KL + self.cross_entropy
      # tf.summary.scalar('loss', self.loss)

    with tf.name_scope('train'):
      self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
          self.loss)
      # if self.args.lr_decay:
        # tf.summary.scalar('lr', self.learning_rate)

    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))

    with tf.name_scope('accuracy'):
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.summary.scalar('accuracy', self.accuracy)

  def run_model(self):
    self.build_model()
    mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
    log_dir = self.log_dir + '/Flip{}batch{}scale{}lr{}'.format(self.isFlip, self.batch_size, self.scale, self.learning_rate)
    tlogger.start(log_dir)
    for k, v in self.args.__dict__.items():
      tlogger.log('{}: {}'.format(k, v))

    with tf.Session() as sess:
      # merged = tf.summary.merge_all()
      # train_writer = tf.summary.FileWriter(self.log_dir + '/Flip{}train{}scale{}lr{}'
      #   .format(self.isFlip, self.batch_size, self.scale, self.learning_rate), sess.graph)
      # test_writer = tf.summary.FileWriter(self.log_dir + '/Flip{}test{}scale{}lr{}'
      #   .format(self.isFlip, self.batch_size, self.scale, self.learning_rate))
      # saver = tf.train.Saver(max_to_keep=40)

      sess.run(tf.global_variables_initializer())
      # M = mnist.train.labels.shape[0] // self.batch_size
      M = 55000

      tstart = time.time()
      for i in range(self.args.num_iterations):
        start = time.time()
        if self.args.lr_decay:
          step_size = self.piecewise_learning_rate(i)
          sess.run(tf.assign(self.learning_rate, step_size))
        batch = mnist.train.next_batch(self.batch_size)
        _, train_KL, train_accuracy, train_loss, train_cross = sess.run([self.train_step, 
          self.KL, self.accuracy, self.loss, self.cross_entropy], feed_dict={self.x: batch[0], 
          self.y_: batch[1], self.M: M, self.n: batch[0].shape[0], self.isTrain: True})

        if i % 100 == 0:
          # train_writer.add_summary(train_summary, i)
          tlogger.log('********** Iteration {} **********'.format(i))
          tlogger.record_tabular("train_loss", train_loss)
          tlogger.record_tabular("train_cross", train_cross)
          tlogger.record_tabular("train_KL", train_KL)
          tlogger.record_tabular("train_acc", train_accuracy)
          # print('Train accuracy, Loss at step %s: %s, %s' % (i, train_accuracy, train_loss))

          xs, ys = mnist.test.images, mnist.test.labels
          test_accuracy, test_loss, test_KL, test_cross = sess.run([self.accuracy, self.loss, 
            self.KL, self.cross_entropy], feed_dict={ self.x: xs, self.y_: ys, self.M: M, 
            self.n: xs.shape[0], self.isTrain: False})
          # test_writer.add_summary(test_summary, i)
          # print('Test accuracy at step %s: %s' % (i, test_accuracy))
          tlogger.record_tabular("test_loss", test_loss)
          tlogger.record_tabular("test_cross", test_cross)
          tlogger.record_tabular("test_KL", test_KL)
          tlogger.record_tabular("test_acc", test_accuracy)
          tlogger.record_tabular("TimeElapsed", time.time() - tstart)
          tlogger.dump_tabular()
      tlogger.stop()

      # print('test accuracy %g' % self.accuracy.eval(feed_dict={
      #   self.x: mnist.test.images, self.y_: mnist.test.labels, \
      #       self.M: 1.0, self.n: mnist.test.images.shape[0], self.isTrain: False}))

      # if os.path.exists(self.model):
      # saver.save(sess, os.path.join(self.model, '{}{}scale{}'.format(self.batch_size, self.scale)), global_step=i)
      # train_writer.close()
      # test_writer.close()

  def piecewise_learning_rate(self, step):
    init_lr = self.args.learning_rate
    num_iterations = self.args.num_iterations
    if step <= (num_iterations/3):
      return init_lr
    elif (num_iterations/3)< step <= (2*num_iterations/3):
      return 0.1*init_lr
    else:
      return 0.1*0.1*init_lr

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data_dir', type=str, default='/ais/gobi5/wenyemin/summerproject/data/mnist',
                      help='data file dir')
  parser.add_argument('--log_dir', type=str, default='NEW_log',
                      help='log file dir')
  parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='learning rate')
  parser.add_argument('--scale', type=float, default=0.1,
                      help='scale the KL term')
  parser.add_argument('--batch_size', type=int, default=128,
                      help='minibatch size')
  parser.add_argument('--num_iterations', type=int, default=6000,
                      help='number of iterations')
  parser.add_argument('--isFlip', action='store_true', default=False,
                      help='whether use flipout')
  parser.add_argument('--lr_decay', action='store_true', default=False,
                      help='whether use learning rate decay')
  parser.add_argument('--LRT', action='store_true', default=False,
                      help='whether use local reparametrization tric')
  args = parser.parse_args()
  BBB = bayes(args)
  BBB.run_model()