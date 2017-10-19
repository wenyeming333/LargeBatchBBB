from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cPickle as pickle
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

def log_prior(x, isScaled=False):
    if isScaled:
        sigma1 = tf.exp(-1.0)
        sigma2 = tf.exp(-7.0)
        pi = 0.5
        return tf.log(pi*normal(x,0.0,sigma1)+(1-pi)*normal(x,0.0,sigma2))
    else:
        sigma = tf.exp(-1.0)
        return log_normal(x, 0.0, sigma)

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class bayes():
    def __init__(self, isFlipout, mini_batch_size, isContorl):
        self.data_dir = '/ais/gobi5/wenyemin/summerproject/data/mnist'
        self.learning_rate = 0.0001
        self.log_dir = 'log_map'
        self.model = 'tf_models'
        self.isFlipout = isFlipout
        self.mini_batch_size = mini_batch_size
        self.isContorl = isContorl

    def flipoutlayerFC(self, x, W_0, delta_W):
        weight_dim = W_0.shape.as_list()
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
        return pert_x

    def bayesian_nn_layer_flip(self, input_tensor, input_dim, output_dim, isTrain, layer_name):
        with tf.name_scope(layer_name):
            mu_w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            rho_w = tf.Variable(tf.truncated_normal([input_dim, output_dim], mean=-3.0, stddev=0.1))
            biases = tf.Variable(tf.zeros([output_dim]))
            if self.isFlipout:
                if self.isContorl:
                    ctrl_dep = []
                    main_part = tf.matmul(input_tensor, mu_w)
                    ctrl_dep.append(main_part)
                    with tf.control_dependencies(ctrl_dep):
                        sigma_w = tf.log(1+tf.exp(rho_w))
                        epsilon_w = tf.truncated_normal([input_dim, output_dim], stddev=1.0)
                        delta_W = tf.multiply(sigma_w, epsilon_w)
                        weights = mu_w + delta_W
                        flipoutFC = self.flipoutlayerFC(input_tensor, mu_w, delta_W)
                    preactivate = main_part + flipoutFC + biases
                    return tf.nn.relu(preactivate)                  
                else:
                    main_part = tf.matmul(input_tensor, mu_w)
                    sigma_w = tf.log(1+tf.exp(rho_w))
                    epsilon_w = tf.truncated_normal([input_dim, output_dim], stddev=1.0)
                    delta_W = tf.multiply(sigma_w, epsilon_w)
                    weights = mu_w + delta_W
                    flipoutFC = self.flipoutlayerFC(input_tensor, mu_w, delta_W)
                    preactivate = main_part + flipoutFC + biases
                    return tf.nn.relu(preactivate)
            else:
                sigma_w = tf.log(1+tf.exp(rho_w))
                epsilon_w = tf.truncated_normal([input_dim, output_dim], stddev=1.0)
                delta_W = tf.multiply(sigma_w, epsilon_w)
                weights = mu_w + delta_W
                return tf.nn.relu(tf.matmul(input_tensor, weights)+biases)

    def build_model(self):

        self.isTrain = tf.placeholder(tf.bool, name='isTrain')

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

            self.M = tf.placeholder(tf.float32, shape=(), name='number_mini_batches')

        hidden1 = self.bayesian_nn_layer_flip(self.x , 784, 400, self.isTrain, 'layer1')
        hidden2 = self.bayesian_nn_layer_flip(hidden1, 400, 400, self.isTrain, 'layer2')
        self.logits = self.bayesian_nn_layer_flip(hidden2, 400, 10, self.isTrain, 'layer3')

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
        params = tf.trainable_variables()
        self.grads = tf.gradients(cross_entropy, params)

    def run_model(self):
        self.build_model()
        np.random.seed(seed=1) # Seed 1 or 100
        mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
        M = 55000.0 # Number of training examples.

        with open('tf55000_x.npy','rb') as f:
            data = np.load(f)
        with open('tf55000_y.npy','rb') as f:
            target = np.load(f)

        model_path = os.path.join(self.model,'model_factor10.ckpt')
        K = self.mini_batch_size

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=40)
            sess.run(tf.global_variables_initializer())
            for i in range(5):
                start_time = time.time()
                for j in range(200):
                    idx = np.random.randint(0,47000)
                    x, y = data[idx:idx+K,:], target[idx:idx+K,:]
                    _ = sess.run(self.logits, feed_dict={self.x: x, self.y_: \
                        y, self.M: M, self.isTrain: True})
                print('{} test: batch size {} 200 forward passes takes {} seconds'.format(i, K, time.time()-start_time))

            for i in range(5):
                start_time = time.time()
                for j in range(200):
                    idx = np.random.randint(0,47000)
                    x, y = data[idx:idx+K,:], target[idx:idx+K,:]
                    _ = sess.run(self.grads, feed_dict={self.x: x, self.y_: \
                        y, self.M: M, self.isTrain: True})
                print('{} test: batch size {} 200 forward-backward passes takes {} seconds'.format(i, K, time.time()-start_time))
        tf.reset_default_graph()

if __name__ == '__main__':
    for K in [128,1024,4096,8192,12800]:
        print('=========Flipout =========')
        FlipBBB = bayes(isFlipout=True, mini_batch_size=K, isContorl=False)
        FlipBBB.run_model()
        print('=========NoFlip=========')
        FlipBBB = bayes(isFlipout=False, mini_batch_size=K, isContorl=False)
        FlipBBB.run_model()