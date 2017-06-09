#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

import tensorflow as tf

from data_set import DataSet
from model import Model


class NeuralNetworkModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)

    self._batch_size = params.get('batch_size', 1024)
    self._epochs = params.get('epochs', 80)
    self._hidden_layer = params.get('hidden_layer', 50)
    self._learning_rate = params.get('learning_rate', 0.001)
    self._init_sigma = params.get('init_sigma', 0.001)
    self._lambda = params.get('lambda', 0.005)
    self._dropout = params.get('dropout', 0.5)

    self._session = None


  def session(self):
    self._session = tf.Session()
    return self._session


  def _fit(self, train_x, train_y):
    train = DataSet(train_x, train_y)
    _, features = train_x.shape

    x = tf.placeholder('float', shape=[None, features], name='x')
    y = tf.placeholder('float', shape=[None], name='y')
    mode = tf.placeholder(tf.string, name='mode')

    W1 = tf.Variable(tf.random_normal(shape=[features, self._hidden_layer]) * self._init_sigma, name='W1')
    b1 = tf.Variable(tf.random_normal(shape=[self._hidden_layer]) * self._init_sigma, name='b1')
    layer1 = tf.matmul(x, W1) + b1
    layer1 = tf.nn.relu(layer1)
    layer1 = dropout(layer1, tf.equal(mode, 'train'), keep_prob=self._dropout)

    W2 = tf.Variable(tf.random_normal(shape=[self._hidden_layer, 1]) * self._init_sigma, name='W2')
    b2 = tf.Variable(tf.random_normal(shape=[1]) * self._init_sigma, name='b2')
    output_layer = tf.matmul(layer1, W2) + b2

    cost = tf.reduce_mean(tf.abs(output_layer - y)) + self._lambda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
    optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    print 'Start training'
    self._session.run(init)
    while train.epochs_completed < self._epochs:
      batch_x, batch_y = train.next_batch(self._batch_size)
      _, cost_ = self._session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, mode: 'train'})
      if train.just_completed and train.epochs_completed % 10 == 0:
        print 'Epoch: %2d cost=%.6f' % (train.epochs_completed, cost_)
    print 'Training completed'

    self._x = x
    self._mode = mode
    self._output_layer = output_layer

  def predict(self, test_x):
    return self._session.run(self._output_layer, feed_dict={self._x: test_x, self._mode: 'test'}).reshape((-1,))


def dropout(incoming, is_training, keep_prob):
  if keep_prob is None:
    return incoming
  return tf.cond(is_training, lambda: tf.nn.dropout(incoming, keep_prob), lambda: incoming)
