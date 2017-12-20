#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import tensorflow as tf

from nn_ops import COST_FUNCTIONS
from tensorflow_model import TensorflowModel


CELL_TYPES = {
  'lstm': tf.nn.rnn_cell.LSTMCell,
  'gru': tf.nn.rnn_cell.GRUCell
}


class RecurrentModel(TensorflowModel):
  EXPECTS_TIME_PARAM = True

  def __init__(self, **params):
    TensorflowModel.__init__(self, **params)

    self._time_steps = params.get('time_steps', 10)
    self._features = self._features / self._time_steps  # because features are unrolled
    self._layers = params.get('layers', [])
    self._cell_type = params.get('cell_type', 'lstm')
    self._double_state = params.get('double_state', False)
    self._dropout = params.get('dropout', 1.0)
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
      x_series = tf.reshape(x, [-1, self._time_steps, self._features])

      cells = []
      cell_class = CELL_TYPES[self._cell_type]
      for layer_size in self._layers:
        cell = cell_class(num_units=layer_size)
        cells.append(cell)
      multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
      outputs, states = tf.nn.dynamic_rnn(multi_cell, x_series, dtype=tf.float32)

      if self._cell_type == 'lstm':
        top_layer_h_state = states[-1].h
        top_layer_c_state = states[-1].c
        if self._double_state:
          top_layer_state = tf.concat([top_layer_h_state, top_layer_c_state], axis=1)
        else:
          top_layer_state = top_layer_h_state
      elif self._cell_type == 'gru':
        top_layer_state = states[-1]
        # double state is not supported for GRU
      else:
        # unexpected cell type, but let's expect the standard state value
        top_layer_state = states[-1]

      dropout_layer = tf.layers.dropout(top_layer_state, self._dropout, training=tf.equal(mode, 'train'))
      output_layer = tf.layers.dense(dropout_layer, units=1,
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
