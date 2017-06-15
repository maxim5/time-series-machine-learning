#!/usr/bin/env python
__author__ = 'maxim'


import os
import numpy as np

from model import Model


class LinearModel(Model):
  DATA_WITH_BIAS = True

  def __init__(self, **params):
    Model.__init__(self, **params)
    self._beta = None

  def fit(self, train):
    x, y = train.x, train.y
    self._beta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

  def predict(self, x):
    return x.dot(self._beta)

  def save(self, dest_dir):
    path = os.path.join(dest_dir, 'beta.npy')
    np.save(path, self._beta)

  def restore(self, source_dir):
    path = os.path.join(source_dir, 'beta.npy')
    self._beta = np.load(path)
