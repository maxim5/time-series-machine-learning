#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import re
import numpy as np


PESSIMIST = lambda pred, truth: np.maximum(pred - truth, 0)   # favors truth > pred, i.e. lower bound estimator
OPTIMIST  = lambda pred, truth: np.maximum(truth - pred, 0)   # favors pred > truth, i.e. upper bound estimator

class JobInfo:
  def __init__(self, data_dir, zoo_dir, ticker, target, estimator):
    assert target in ['high', 'low']
    assert estimator in ['opt', 'optimist', 'optimistic', 'pes', 'pessimist', 'pessimistic']
    self.data_dir = data_dir
    self.zoo_dir = zoo_dir
    self.ticker = ticker
    self.target = target
    self.estimator = estimator[:3]

  def residual_fun(self):
    if self.estimator == 'opt':
      return OPTIMIST
    elif self.estimator == 'pes':
      return PESSIMIST

  def as_run_params(self):
    return {
      'ticker': self.ticker,
      'target': self.target,
      'estimator': self.estimator,
    }

  def get_source_name(self):
    return os.path.join(self.data_dir, '%s.csv' % self.ticker)

  def get_dest_name(self, eval_, k):
    return os.path.join(self.zoo_dir, self.ticker, '%s__%s__eval=%.4f__k=%d' % (self.target, self.estimator, eval_, k))

  def get_current_eval_results(self):
    directory = os.path.join(self.zoo_dir, self.ticker)
    return parse_eval(directory, lambda info: info['target'] == self.target and info['estimator'] == self.estimator)


def parse_model_infos(directory):
  if os.path.exists(directory):
    files = os.listdir(directory)
    return [_parse_model_file(file_name) for file_name in files]
  return []


def parse_eval(directory, accept):
  infos = parse_model_infos(directory)
  return [info['eval'] for info in infos if info if accept(info)]


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
