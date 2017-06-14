#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np
import pandas as pd


class Evaluator:
  def __init__(self):
    pass


  def eval(self, model, test_set):
    prediction = model.predict(test_set.x)
    stats = self._compute_stats(prediction, test_set.y)
    return self._evaluate(stats), stats


  def stats_str(self, stats):
    return 'Mean absolute error: %.6f\n' % stats['mae'] + \
           'SD absolute error:   %.6f\n' % stats['stdae'] + \
           'Sign accuracy:       %.6f\n' % stats['sign_accuracy'] + \
           'Mean squared error:  %.6f\n' % stats['mse'] + \
           'Sqrt of MSE:         %.6f\n' % stats['sqrt_mse'] + \
           'Mean error:          %.6f\n' % stats['me'] + \
           'Residuals stats:     %s\n' % _series_stats(stats['raw_residuals']) + \
           'Relative residuals:  %s\n' % _series_stats(stats['rel_residuals'])


  def _compute_stats(self, prediction, truth):
    residuals = np.abs(prediction - truth)
    return {
      'mae': np.mean(residuals),
      'stdae': np.std(residuals),
      'sign_accuracy': np.mean(np.equal(np.sign(prediction), np.sign(truth))),
      'mse': np.mean(np.power(prediction - truth, 2.0)),
      'sqrt_mse': np.mean(np.power(prediction - truth, 2.0)) ** 0.5,
      'me': np.mean(prediction - truth),
      'raw_residuals': residuals,
      'rel_residuals': residuals / np.maximum(np.abs(truth), 1e-3),
    }


  def _evaluate(self, stats):
    risk_factor = 1.0
    return stats['mae'] + risk_factor * stats['stdae']


def _series_stats(series):
  stats = pd.Series(series).describe(percentiles=[0.25, 0.5, 0.75, 0.9])
  return 'mean=%.4f std=%.4f percentile=[0%%=%.4f 25%%=%.4f 50%%=%.4f 75%%=%.4f 90%%=%.4f 100%%=%.4f]' % \
         (stats['mean'], stats['std'], stats['min'], stats['25%'], stats['50%'], stats['75%'], stats['90%'], stats['max'])
