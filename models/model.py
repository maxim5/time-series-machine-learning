#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np
import pandas as pd

from util import *


class Model():
  def __init__(self, **params):
    self._k = params.get('k')
    self._target_column = params.get('target_column')
    self._residual_fun = params.get('residual_fun')
    self._cost = None
    self._with_bias = False


  def session(self):
    class Dummy:
      def __enter__(self): pass
      def __exit__(self, exc_type, exc_val, exc_tb): pass
    return Dummy()


  def fit(self, train_df):
    x, y = to_dataset(train_df, self._k, target_column=self._target_column, with_bias=self._with_bias)
    self._fit(x, y)

    prediction = self.predict(x)
    residuals, relative, r2 = self._residuals(prediction, y)
    _print_residuals(residuals, relative, r2)


  def _fit(self, x, y):
    raise NotImplementedError


  def predict(self, x):
    raise NotImplementedError


  def test(self, test_df):
    x, y = to_dataset(test_df, self._k, target_column=self._target_column, with_bias=self._with_bias)
    prediction = self.predict(x)
    residuals, relative, r2 = self._residuals(prediction, y)
    _print_residuals(residuals, relative, r2)
    self._cost = self._cost_function(residuals, relative, r2)


  def _residuals(self, prediction, truth):
    residuals = self._residual_fun(prediction, truth)
    relative = residuals / np.maximum(np.abs(truth), 1e-3)
    r2 = np.mean(np.power(prediction - truth, 2.0))
    return residuals, relative, r2


  @property
  def cost(self):
    return self._cost


  def _cost_function(self, residuals, relative, r2):
    stats = pd.Series(relative).describe()
    return stats['mean'] + stats['max']


def _print_residuals(residuals, relative, r2):
  info('Raw residuals:      %s' % _series_stats(residuals))
  info('Relative residuals: %s' % _series_stats(relative))
  info('R2=%.6f' % r2)


def _series_stats(series):
  stats = pd.Series(series).describe(percentiles=[0.25, 0.5, 0.75, 0.9])
  return 'mean=%.4f std=%.4f percentile=[0%%=%.4f 25%%=%.4f 50%%=%.4f 75%%=%.4f 90%%=%.4f 100%%=%.4f]' % \
         (stats['mean'], stats['std'], stats['min'], stats['25%'], stats['50%'], stats['75%'], stats['90%'], stats['max'])
