#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


class Model:
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
