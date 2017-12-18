#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


class Model:
  DATA_WITH_BIAS = False
  EXPECTS_TIME_PARAM = False

  def __init__(self, **params):
    self._features = params['features']

  def session(self):
    class Dummy:
      def __enter__(self): pass
      def __exit__(self, exc_type, exc_val, exc_tb): pass
    return Dummy()

  def fit(self, train_set):
    raise NotImplementedError

  def predict(self, x):
    raise NotImplementedError

  def save(self, dest_dir):
    pass

  def restore(self, source_dir):
    pass
