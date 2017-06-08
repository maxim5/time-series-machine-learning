#!/usr/bin/env python
__author__ = 'maxim'


import numpy as np
import pandas as pd

from data_util import to_dataset


class Model():
  def __init__(self, **params):
    self.k = params.get('k')
    self.target_column = params.get('target_column')
    self.residual_fun = params.get('residual_fun')
    self.cost = None


  def fit(self, train_df):
    x, y = to_dataset(train_df, self.k, target_column=self.target_column)
    self._fit(x, y)

    prediction = self.predict(x)
    residuals, relative = self._residuals(prediction, y)
    _print_residuals(residuals, relative)


  def _fit(self, x, y):
    raise NotImplementedError


  def predict(self, x):
    raise NotImplementedError


  def test(self, test_df):
    x, y = to_dataset(test_df, self.k, target_column=self.target_column)
    prediction = self.predict(x)
    residuals, relative = self._residuals(prediction, y)
    _print_residuals(residuals, relative)
    self.cost = self._cost_function(residuals, relative)


  def _residuals(self, prediction, truth):
    residuals = self.residual_fun(prediction, truth)
    relative = residuals / np.maximum(np.abs(truth), 1e-3)
    return residuals, relative


  def _cost_function(self, residuals, relative):
    stats = pd.Series(relative).describe(percentiles=[0.9])
    return stats['90%']


def _print_residuals(residuals, relative):
  print 'Raw residuals:      %s' % _series_stats(residuals)
  print 'Relative residuals: %s' % _series_stats(relative)


def _series_stats(series):
  stats = pd.Series(series).describe(percentiles=[0.25, 0.5, 0.75, 0.9])
  return 'mean=%.4f std=%.4f percentile=[0%%=%.4f 25%%=%.4f 50%%=%.4f 75%%=%.4f 90%%=%.4f 100%%=%.4f]' % \
         (stats['mean'], stats['std'], stats['min'], stats['25%'], stats['50%'], stats['75%'], stats['90%'], stats['max'])
