#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import tensorflow as tf

from tensorflow_model import TensorflowModel
from nn_ops import ACTIVATIONS, COST_FUNCTIONS, dropout, batch_normalization


class NeuralNetworkModel(TensorflowModel):
  def __init__(self, **params):
    TensorflowModel.__init__(self, **params)

    self._layers = params.get('layers', [])
    self._init_sigma = params.get('init_sigma', 0.001)
    self._lambda = params.get('lambda', 0.005)
    self._cost_func = COST_FUNCTIONS[params.get('cost_func', 'l2')]

    self._compile()


  def _compile(self):
    TensorflowModel._compile(self)

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
    self._output = output_layer
    self._cost = cost
    self._optimizer = optimizer
    self._init = init
