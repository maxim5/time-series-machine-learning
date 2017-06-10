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
    self._eval = None
    self._stats = None
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
    stats = self._compute_stats(prediction, y)
    _print_stats(stats)


  def _fit(self, x, y):
    raise NotImplementedError


  def predict(self, x):
    raise NotImplementedError


  def test(self, test_df):
    x, y = to_dataset(test_df, self._k, target_column=self._target_column, with_bias=self._with_bias)
    prediction = self.predict(x)
    self._stats = self._compute_stats(prediction, y)
    _print_stats(self._stats)
    self._eval = self._eval_function(self._stats)


  def _compute_stats(self, prediction, truth):
    raw_residuals = self._residual_fun(prediction, truth)
    rel_residuals = raw_residuals / np.maximum(np.abs(truth), 1e-3)
    r2 = np.mean(np.power(prediction - truth, 2.0))
    r1 = np.power(r2, 0.5)
    return {
      'raw_residuals': raw_residuals,
      'rel_residuals': rel_residuals,
      'r1': r1,
      'r2': r2,
    }


  def eval(self):
    return self._eval


  def stats_str(self):
    return 'Raw residuals:      %s\n' % _series_stats(self._stats['raw_residuals']) + \
           'Relative residuals: %s\n' % _series_stats(self._stats['rel_residuals']) + \
           'R1=%.6f\n' % self._stats['r1'] + \
           'R2=%.6f\n' % self._stats['r2']


  def _eval_function(self, stats):
    rel_residuals = stats['rel_residuals']
    rel_stats = pd.Series(rel_residuals).describe()
    values = np.array([rel_stats['mean'], rel_stats['max'], stats['r1']])
    weights = np.array([1, 1, 8])
    return np.dot(values, weights)


  def save(self, dest_dir):
    pass


def _print_stats(stats):
  info('Raw residuals:      %s' % _series_stats(stats['raw_residuals']))
  info('Relative residuals: %s' % _series_stats(stats['rel_residuals']))
  info('R1=%.6f' % stats['r1'])
  info('R2=%.6f' % stats['r2'])


def _series_stats(series):
  stats = pd.Series(series).describe(percentiles=[0.25, 0.5, 0.75, 0.9])
  return 'mean=%.4f std=%.4f percentile=[0%%=%.4f 25%%=%.4f 50%%=%.4f 75%%=%.4f 90%%=%.4f 100%%=%.4f]' % \
         (stats['mean'], stats['std'], stats['min'], stats['25%'], stats['50%'], stats['75%'], stats['90%'], stats['max'])
