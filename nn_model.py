#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

import tensorflow as tf

from data_set import DataSet
from model import Model


learning_rate = 0.005
batch_size = 100
epochs = 100
hidden_layer = 10


class NeuralNetworkModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)

    self._session = None


  def session(self):
    self._session = tf.Session()
    return self._session


  def _fit(self, train_x, train_y):
    train = DataSet(train_x, train_y)
    _, features = train_x.shape

    x = tf.placeholder('float', shape=[None, features], name='x')
    y = tf.placeholder('float', shape=[None], name='y')

    W1 = tf.Variable(tf.random_normal(shape=[features, hidden_layer]) * 0.01, name='W1')
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer]) * 0.01, name='b1')
    layer1 = tf.matmul(x, W1) + b1
    layer1 = tf.nn.elu(layer1, name='elu-alpha')

    W2 = tf.Variable(tf.random_normal(shape=[hidden_layer, 1]) * 0.01, name='W2')
    b2 = tf.Variable(tf.random_normal(shape=[1]) * 0.01, name='b2')
    output_layer = tf.matmul(layer1, W2) + b2

    cost = tf.reduce_mean(tf.pow(output_layer - y, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    print 'Start training'
    self._session.run(init)

    while train.epochs_completed < epochs:
      batch_x, batch_y = train.next_batch(batch_size)
      self._session.run(optimizer, feed_dict={x: batch_x, y: batch_y})

      if train.just_completed:
        cost_ = self._session.run(cost, feed_dict={x: batch_x, y: batch_y})
        print 'Epoch: %2d  cost=%.6f' % (train.epochs_completed, cost_)

    self._x = x
    self._output_layer = output_layer

  def predict(self, test_x):
    return self._session.run(self._output_layer, feed_dict={self._x: test_x}).reshape((-1,))
