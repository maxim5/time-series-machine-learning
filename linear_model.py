#!/usr/bin/env python
__author__ = 'maxim'


import numpy as np

from data_util import to_dataset
from train_util import print_residuals


class LinearModel:
  def __init__(self, train_df, test_df, **params):
    self._train_df = train_df
    self._test_df = test_df

    self._beta = None

    self.k = params.get('k')
    self.target_column = params.get('target_column')
    self.residual_fun = params.get('residual_fun')


  def train(self):
    x, y = to_dataset(self._train_df, self.k, target_column=self.target_column)
    self._beta = _linear_solve(x, y)
    print 'Linear solution found'

    prediction = self.predict(x)
    residuals = self.residuals(prediction, y)
    print_residuals(residuals)


  def predict(self, x):
    return _linear_predict(x, self._beta)


  def residuals(self, prediction, truth):
    return self.residual_fun(prediction, truth)


  def test(self):
    x, y = to_dataset(self._test_df, self.k, target_column=self.target_column)
    prediction = self.predict(x)
    residuals = self.residuals(prediction, y)
    print_residuals(residuals)


def _linear_solve(x, y):
  return np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)


def _linear_predict(x, beta):
  return x.dot(beta)
