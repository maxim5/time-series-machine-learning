#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import tensorflow as tf


def leaky_relu(x, alpha=0.1):
  x = tf.nn.relu(x)
  m_x = tf.nn.relu(-x)
  x -= alpha * m_x
  return x


def prelu(x):
  shape = x.get_shape()
  alpha = tf.Variable(initial_value=tf.zeros(shape=shape[1:]), name='alpha')
  x = tf.nn.relu(x) + tf.multiply(alpha, (x - tf.abs(x))) * 0.5
  return x


ACTIVATIONS = {'leaky_relu': leaky_relu, 'prelu': prelu}
ACTIVATIONS.update({name: getattr(tf, name) for name in ['tanh']})
ACTIVATIONS.update({name: getattr(tf.nn, name) for name in ['relu', 'elu', 'sigmoid']})


COST_FUNCTIONS = {
  'l1': lambda output, y: tf.reduce_mean(tf.abs(output - y)),
  'l2': lambda output, y: tf.reduce_mean(tf.pow(output - y, 2.0)),
}


def dropout(incoming, is_training, keep_prob):
  if keep_prob is None:
    return incoming
  return tf.cond(is_training, lambda: tf.nn.dropout(incoming, keep_prob), lambda: incoming)
