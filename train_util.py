#!/usr/bin/env python
__author__ = 'maxim'

import pandas as pd
import numpy as np


def print_residuals(residuals, truth):
  print 'Raw residuals:      %s' % series_stats(residuals)

  relative = residuals / np.maximum(np.abs(truth), 1e-3)
  print 'Relative residuals: %s' % series_stats(relative)


def series_stats(series):
  stats = pd.Series(series).describe()
  return 'mean=%.4f std=%.4f percentile=[0%%=%.4f 25%%=%.4f 50%%=%.4f 75%%=%.4f 100%%=%.4f]' % \
         (stats['mean'], stats['std'], stats['min'], stats['25%'], stats['50%'], stats['75%'], stats['max'])
