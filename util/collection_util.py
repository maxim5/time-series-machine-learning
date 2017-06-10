#!/usr/bin/env python
__author__ = 'maxim'


import numpy as np


def smart_str(val):
  if type(val) in [float, np.float32, np.float64] and val:
    return "%.6f" % val if abs(val) > 1e-6 else "%e" % val
  if type(val) == dict:
    return '{%s}' % ', '.join(['%s: %s' % (repr(k), smart_str(val[k])) for k in sorted(val.keys())])
  if type(val) in [list, tuple]:
    return '[%s]' % ', '.join(['%s' % smart_str(i) for i in val])
  return repr(val)


def str_to_obj(s):
  import ast
  return ast.literal_eval(s)
