#!/usr/bin/env python
__author__ = 'maxim'


import numpy as np

from model import Model


class LinearModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)
    self._beta = None


  def fit(self, train):
    x, y = train.x, train.y
    self._beta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)


  def predict(self, x):
    return x.dot(self._beta)
