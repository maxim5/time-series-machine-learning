#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = 'maxim'


import os
# Suppress initial logging, can be modified later
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from .model import Model
from util import *


class TensorflowModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)

    self._cpu_only = params.get('cpu_only', False)
    self._tensorflow_log_level = params.get('tensorflow_log_level', 2)

    self._epochs = params.get('epochs', 80)
    self._batch_size = params.get('batch_size', 1024)
    self._learning_rate = params.get('learning_rate', 0.001)

    self._graph = None
    self._session = None
    self._init = None
    self._x = None
    self._y = None
    self._mode = None
    self._output = None
    self._cost = None
    self._optimizer = None
    self._init = None


  def _compile(self):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self._tensorflow_log_level)


  def session(self):
    assert self._graph is not None
    config = tf.ConfigProto(device_count={'GPU': 0}) if self._cpu_only else tf.ConfigProto()
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
    return self._session.run(self._output, feed_dict={self._x: test_x, self._mode: 'test'}).reshape((-1,))


  def save(self, dest_dir):
    path = os.path.join(dest_dir, 'session.data')
    saver = tf.train.Saver()
    saver.save(self._session, path)


  def restore(self, source_dir):
    path = os.path.join(source_dir, 'session.data')
    saver = tf.train.Saver()
    saver.restore(self._session, path)
