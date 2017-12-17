#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import tensorflow as tf

from model import Model
from nn_ops import COST_FUNCTIONS
from util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
CPU_ONLY = False

class LstmModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)

    self._time_steps = params.get('time_steps', 10)
    self._features = self._features / self._time_steps  # because features are unrolled
    self._batch_size = 128
    self._epochs = params.get('epochs', 80)
    self._hidden_size = 32
    # self._layers = params.get('layers', [])
    self._learning_rate = params.get('learning_rate', 0.001)
    self._lambda = params.get('lambda', 0.005)
    self._cost_func = COST_FUNCTIONS[params.get('cost_func', 'l2')]

    self._graph = None
    self._session = None
    self._compile()

  def _compile(self):
    with tf.Graph().as_default() as self._graph:
      x = tf.placeholder(tf.float32, shape=[None, self._time_steps * self._features], name='x')
      y = tf.placeholder(tf.float32, shape=[None], name='y')
      mode = tf.placeholder(tf.string, name='mode')

      # unroll the features into time-series
      x_series = tf.reshape(x, [-1, self._time_steps, self._features])

      lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._hidden_size)
      multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell])
      outputs, states = tf.nn.dynamic_rnn(multi_cell, x_series, dtype=tf.float32)

      top_layer_h_state = states[-1].h
      output_layer = tf.layers.dense(top_layer_h_state, units=1,
                                     activation=tf.nn.elu,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self._lambda),
                                     name='dense')

      cost = self._cost_func(output_layer, y)
      optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(cost)

      init = tf.global_variables_initializer()

    self._x = x
    self._y = y
    self._mode = mode
    self._output_layer = output_layer
    self._cost = cost
    self._optimizer = optimizer
    self._init = init

  # TODO: reuse with NeuralNetwork
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
