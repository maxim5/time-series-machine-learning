#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import re
from numpy import np


PESSIMIST = lambda pred, truth: np.maximum(pred - truth, 0)   # favors truth > pred, i.e. lower bound estimator
OPTIMIST  = lambda pred, truth: np.maximum(truth - pred, 0)   # favors pred > truth, i.e. upper bound estimator

class ModelInput:
  def __init__(self, ticker, target, estimator):
    assert target in ['high', 'low']
    assert estimator in ['opt', 'optimist', 'pes', 'pessimist']
    self.ticker = ticker
    self.target = target
    self.estimator = estimator[:3]

  def residual_fun(self):
    if self.estimator == 'opt':
      return OPTIMIST
    elif self.estimator == 'pes':
      return PESSIMIST

  def get_source_name(self):
    return '%s.csv' % self.ticker

  def get_model_dir_name(self, eval_, k):
    return '%s__%s__c=%.4f__k=%d' % (self.target, self.estimator, eval_, k)


def parse_model_infos(directory):
  files = os.listdir(directory)
  return [_parse_model_file(file_name) for file_name in files]


def parse_eval(directory):
  infos = parse_model_infos(directory)
  return [info['eval'] for info in infos if info]


def _parse_model_file(file_name):
  if not '=' in file_name:
    return {}

  match = re.match('([a-z]+)__([a-z]+)__eval=([0-9.]+)__k=([0-9]+)', file_name)
  if not match:
    return {}

  return {
    'target': match.group(1),
    'estimator': match.group(2),
    'eval': float(match.group(3)),
    'k': int(match.group(4)),
  }
