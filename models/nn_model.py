#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import tensorflow as tf

from model import Model
from util import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


COST_FUNCTIONS = {
  'l1': lambda output, y: tf.reduce_mean(tf.abs(output - y)),
  'l2': lambda output, y: tf.reduce_mean(tf.pow(output - y, 2.0)),
}

class NeuralNetworkModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)

    self._batch_size = params.get('batch_size', 1024)
    self._epochs = params.get('epochs', 80)
    self._hidden_layer = params.get('hidden_layer', 50)
    self._learning_rate = params.get('learning_rate', 0.001)
    self._init_sigma = params.get('init_sigma', 0.001)
    self._lambda = params.get('lambda', 0.005)
    self._cost_func = COST_FUNCTIONS[params.get('cost_func', 'l2')]
    self._dropout = params.get('dropout', 0.5)

    self._graph = None
    self._session = None
    self._compile()


  def _compile(self):
    with tf.Graph().as_default() as self._graph:
      x = tf.placeholder(tf.float32, shape=[None, self._features], name='x')
      y = tf.placeholder(tf.float32, shape=[None], name='y')
      mode = tf.placeholder(tf.string, name='mode')

      init = lambda shape: tf.random_normal(shape=shape) * self._init_sigma

      W1 = tf.Variable(init([self._features, self._hidden_layer]), name='W1')
      b1 = tf.Variable(init([self._hidden_layer]), name='b1')
      layer1 = tf.matmul(x, W1) + b1
      layer1 = tf.nn.relu(layer1)
      layer1 = dropout(layer1, tf.equal(mode, 'train'), keep_prob=self._dropout)

      W2 = tf.Variable(init([self._hidden_layer, 1]), name='W2')
      b2 = tf.Variable(init([1]), name='b2')
      output_layer = tf.matmul(layer1, W2) + b2

      reg = self._lambda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
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
    config = tf.ConfigProto()
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
    os.makedirs(dest_dir)
    path = os.path.join(dest_dir, 'session.data')
    saver = tf.train.Saver()
    saver.save(self._session, path)
    info('Session saved to %s' % path)


  def restore(self, source_dir):
    path = os.path.join(source_dir, 'session.data')
    saver = tf.train.Saver()
    saver.restore(self._session, path)
    info('Session restored from %s' % path)


def dropout(incoming, is_training, keep_prob):
  if keep_prob is None:
    return incoming
  return tf.cond(is_training, lambda: tf.nn.dropout(incoming, keep_prob), lambda: incoming)
