#!/usr/bin/env python
from __future__ import absolute_import

__author__ = 'maxim'


import os
import numpy as np

from .model import Model
from util import vlog


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
    vlog('Saving the model to:', path)
    np.save(path, self._beta)

  def restore(self, source_dir):
    path = os.path.join(source_dir, 'beta.npy')
    vlog('Restoring the model from:', path)
    self._beta = np.load(path)
