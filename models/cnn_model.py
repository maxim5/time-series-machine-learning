#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = 'maxim'

import tensorflow as tf

from .nn_ops import COST_FUNCTIONS
from .tensorflow_model import TensorflowModel

class ConvModel(TensorflowModel):
  EXPECTS_TIME_PARAM = True

  def __init__(self, **params):
    super().__init__(**params)

    self._time_steps = params.get('time_steps', 10)
    self._features = int(self._features / self._time_steps)  # because features are unrolled
    self._layers = params.get('layers', [])
    self._dropout = params.get('dropout', 0.0)  # drop probability (not keep!)
    self._lambda = params.get('lambda', 0.005)
    self._cost_func = COST_FUNCTIONS[params.get('cost_func', 'l2')]

    self._compile()

  def _compile(self):
    TensorflowModel._compile(self)

    with tf.Graph().as_default() as self._graph:
      x = tf.placeholder(tf.float32, shape=[None, self._time_steps * self._features], name='x')
      y = tf.placeholder(tf.float32, shape=[None], name='y')
      mode = tf.placeholder(tf.string, name='mode')

      # unroll the features into time-series
      layer = tf.reshape(x, [-1, self._time_steps, self._features])

      for filters, kernel_size in self._layers:
        layer = tf.layers.conv1d(layer,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 activation=tf.nn.relu)
      layer = tf.reduce_max(layer, axis=1)

      layer = tf.layers.dropout(layer, rate=self._dropout, training=tf.equal(mode, 'train'))
      output_layer = tf.layers.dense(layer, units=1,
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
    self._output = output_layer
    self._cost = cost
    self._optimizer = optimizer
    self._init = init
