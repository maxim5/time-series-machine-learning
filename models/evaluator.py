#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np
import pandas as pd


class Evaluator:
  def __init__(self, residual_fun):
    self._residual_fun = residual_fun


  def eval(self, model, test_set):
    prediction = model.predict(test_set.x)
    stats = self._compute_stats(prediction, test_set.y)
    return self._evaluate(stats), stats


  def stats_str(self, stats):
    return 'Raw residuals:      %s\n' % _series_stats(stats['raw_residuals']) + \
           'Relative residuals: %s\n' % _series_stats(stats['rel_residuals']) + \
           'R1=%.6f\n' % stats['r1'] + \
           'R2=%.6f\n' % stats['r2']

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

  def _evaluate(self, stats):
    rel_residuals = stats['rel_residuals']
    rel_stats = pd.Series(rel_residuals).describe()
    values = np.array([rel_stats['mean'], rel_stats['max'], stats['r1']])
    weights = np.array([1, 1, 8])
    return np.dot(values, weights)


def _series_stats(series):
  stats = pd.Series(series).describe(percentiles=[0.25, 0.5, 0.75, 0.9])
  return 'mean=%.4f std=%.4f percentile=[0%%=%.4f 25%%=%.4f 50%%=%.4f 75%%=%.4f 90%%=%.4f 100%%=%.4f]' % \
         (stats['mean'], stats['std'], stats['min'], stats['25%'], stats['50%'], stats['75%'], stats['90%'], stats['max'])
