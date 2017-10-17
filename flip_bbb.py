from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

class bayes():
  def __init__(self, args):
    self.args = args
    self.data_dir = '/ais/gobi5/wenyemin/summerproject/data/mnist'
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
        mu_w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
      with tf.name_scope('weights_sd'):
        rho_w = tf.Variable(tf.truncated_normal([input_dim, output_dim], mean=-3.0, stddev=0.1))
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
        epsilon_w = tf.truncated_normal([input_dim, output_dim], stddev=1.0)
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
  def flipoutlayerFC(self, x, W_0, delta_W):
    weight_dim = W_0.shape.as_list()
    # x is n*m where m is the dimension, n is the mini-batch size.
    # W_0 is m*h where h is the num of hidden units.
    epsilon = tf.truncated_normal(weight_dim, stddev=1.0)
    def generate_flipping_factor(dim):
      shape = tf.stack([tf.shape(x)[0], dim])
      random = tf.truncated_normal(shape)
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
        mu_w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
      with tf.name_scope('weights_sd'):
        rho_w = tf.Variable(tf.truncated_normal([input_dim, output_dim], mean=-3.0, stddev=0.1))
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
          epsilon_w = tf.truncated_normal([input_dim, output_dim], stddev=1.0)
          delta_W = tf.multiply(sigma_w, epsilon_w)
          weights = mu_w + delta_W
        with tf.name_scope('flipout'):
          flipoutFC = self.flipoutlayerFC(input_tensor, mu_w, delta_W)
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

  def bayesian_nn_layer_LRT(self, input_tensor, input_dim, output_dim, isTrain, layer_name, nonlinearity=None):
    eps = 1e-35
    with tf.name_scope(layer_name):
      with tf.name_scope('weights_mean'):
        mu_w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
      with tf.name_scope('weights_sd'):
        rho_w = tf.Variable(tf.truncated_normal([input_dim, output_dim], mean=-3.0, stddev=0.1))
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
        mu_b = tf.matmul(input_tensor, mu_w)
        sigma_b = tf.sqrt(tf.matmul(tf.square(input_tensor), tf.square(sigma_w))+eps)
        output_shape = tf.stack([tf.shape(input_tensor)[0], output_dim])
        epsilon = tf.truncated_normal(output_shape, stddev=1.0)
        preactivate = mu_b + tf.multiply(sigma_b, epsilon)
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

  def build_model(self):
    self.isTrain = tf.placeholder(tf.bool, name='isTrain')
    with tf.name_scope('input'):
      self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
      self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
      self.M = tf.placeholder(tf.float32, shape=(), name='number_mini_batches')
      self.n = tf.placeholder(tf.float32, shape=(), name='mini_batch_size')

    if self.args.LRT:
      hidden1, kl1 = self.bayesian_nn_layer_LRT(self.x , 784, 400, self.isTrain, 'layer1', nonlinearity=tf.nn.relu)
      hidden2, kl2 = self.bayesian_nn_layer_LRT(hidden1, 400, 400, self.isTrain, 'layer2', nonlinearity=tf.nn.relu)
      y, kl3 = self.bayesian_nn_layer_LRT(hidden2, 400, 10, self.isTrain, 'layer3')
    elif self.isFlip:
      hidden1, kl1 = self.bayesian_nn_layer_flip(self.x , 784, 400, self.isTrain, 'layer1', nonlinearity=tf.nn.relu)
      hidden2, kl2 = self.bayesian_nn_layer_flip(hidden1, 400, 400, self.isTrain, 'layer2', nonlinearity=tf.nn.relu)
      y, kl3 = self.bayesian_nn_layer_flip(hidden2, 400, 10, self.isTrain, 'layer3')
    else:
      hidden1, kl1 = self.bayesian_nn_layer(self.x , 784, 400, self.isTrain, 'layer1', nonlinearity=tf.nn.relu)
      hidden2, kl2 = self.bayesian_nn_layer(hidden1, 400, 400, self.isTrain, 'layer2', nonlinearity=tf.nn.relu)
      y, kl3 = self.bayesian_nn_layer(hidden2, 400, 10, self.isTrain, 'layer3')

    with tf.name_scope('cross-entropy'):
      self.cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y))

    with tf.name_scope('KL'):
      self.KL = (kl1 + kl2 + kl3) / self.M
      # tf.summary.scalar('KL', self.KL)

    with tf.name_scope('loss'):
      self.loss = self.KL + self.cross_entropy
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
  parser.add_argument('--log_dir', type=str, default='NEW_log',
                      help='log file dir')
  parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='learning rate')
  parser.add_argument('--scale', type=float, default=0.1,
                      help='scale the KL term')
  parser.add_argument('--batch_size', type=int, default=128,
                      help='minibatch size')
  parser.add_argument('--num_iterations', type=int, default=30000,
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