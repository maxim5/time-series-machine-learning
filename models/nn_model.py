#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import tensorflow as tf

from nn_ops import ACTIVATIONS, COST_FUNCTIONS, dropout, batch_normalization

from model import Model
from util import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CPU_ONLY = False


class NeuralNetworkModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)

    self._batch_size = params.get('batch_size', 1024)
    self._epochs = params.get('epochs', 80)
    self._layers = params.get('layers', [])
    self._learning_rate = params.get('learning_rate', 0.001)
    self._init_sigma = params.get('init_sigma', 0.001)
    self._lambda = params.get('lambda', 0.005)
    self._cost_func = COST_FUNCTIONS[params.get('cost_func', 'l2')]

    self._graph = None
    self._session = None
    self._compile()


  def _compile(self):
    with tf.Graph().as_default() as self._graph:
      x = tf.placeholder(tf.float32, shape=[None, self._features], name='x')
      y = tf.placeholder(tf.float32, shape=[None], name='y')
      mode = tf.placeholder(tf.string, name='mode')

      rand_init = lambda shape: tf.random_normal(shape=shape, stddev=self._init_sigma)

      layer = x
      dimension = self._features
      reg = 0
      for idx, layer_params in enumerate(self._layers):
        with tf.variable_scope('l_%d' % idx):
          size = layer_params.get('size', 50)
          W = tf.Variable(rand_init([dimension, size]), name='W%d' % idx)
          b = tf.Variable(rand_init([size]), name='b%d' % idx)
          layer = tf.matmul(layer, W) + b

          batchnorm = layer_params.get('batchnorm', False)
          if batchnorm:
            batch_normalization(layer, tf.equal(mode, 'train'))

          activation_func = ACTIVATIONS[layer_params.get('activation_func', 'relu')]
          layer = activation_func(layer)

          dropout_prob = layer_params.get('dropout', 0.5)
          layer = dropout(layer, tf.equal(mode, 'train'), keep_prob=dropout_prob)

          reg += self._lambda * tf.nn.l2_loss(W)
          dimension = size

      with tf.variable_scope('l_out'):
        W_out = tf.Variable(rand_init([dimension, 1]), name='W')
        b_out = tf.Variable(rand_init([1]), name='b')
        output_layer = tf.matmul(layer, W_out) + b_out
        reg += self._lambda * tf.nn.l2_loss(W_out)

      cost = self._cost_func(output_layer, y) + reg
      optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(cost)

      init = tf.global_variables_initializer()

    self._x = x
    self._y = y
    self._mode = mode
    self._output_layer = output_layer
    self._cost = cost
    self._optimizer = optimizer
    self._init = init


  def session(self):
    assert self._graph is not None
    config = tf.ConfigProto(device_count={'GPU': 0}) if CPU_ONLY else tf.ConfigProto()
    config.gpu_options.allow_growth = True  # https://github.com/vijayvee/Recursive-neural-networks-TensorFlow/issues/1
    self._session = tf.Session(graph=self._graph, config=config)
    return self._session


  def fit(self, train):
    assert self._session is not None
    debug('Start training')
    self._session.run(self._init)
    while train.epochs_completed < self._epochs:
      batch_x, batch_y = train.next_batch(self._batch_size)
      _, cost_ = self._session.run([self._optimizer, self._cost],
                                   feed_dict={self._x: batch_x, self._y: batch_y, self._mode: 'train'})
      if train.just_completed and train.epochs_completed % 10 == 0:
        info('Epoch: %2d cost=%.6f' % (train.epochs_completed, cost_))
    debug('Training completed')


  def predict(self, test_x):
    return self._session.run(self._output_layer, feed_dict={self._x: test_x, self._mode: 'test'}).reshape((-1,))


  def save(self, dest_dir):
    path = os.path.join(dest_dir, 'session.data')
    saver = tf.train.Saver()
    saver.save(self._session, path)


  def restore(self, source_dir):
    path = os.path.join(source_dir, 'session.data')
    saver = tf.train.Saver()
    saver.restore(self._session, path)
