#!/usr/bin/env python
__author__ = 'maxim'

import numpy as np

def print_residuals(residuals):
  print 'residuals: avg=%.8f max=%.8f' % (np.average(residuals), np.max(residuals))
