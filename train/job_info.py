#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import re


class JobInfo:
  def __init__(self, data_dir, zoo_dir, ticker, target):
    self.data_dir = data_dir
    self.zoo_dir = zoo_dir
    self.ticker = ticker
    self.target = target

  def as_run_params(self):
    return {
      'ticker': self.ticker,
      'target': self.target,
    }

  def get_source_name(self):
    return os.path.join(self.data_dir, '%s.csv' % self.ticker)

  def get_dest_name(self, eval_, k):
    return os.path.join(self.zoo_dir, self.ticker, '%s_eval=%.4f_k=%d' % (self.target, eval_, k))

  def get_current_eval_results(self):
    directory = os.path.join(self.zoo_dir, self.ticker)
    return parse_eval(directory, lambda info: info['target'] == self.target)


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

  match = re.match('([a-z]+)_eval=([0-9.]+)_k=([0-9]+)', file_name)
  if not match:
    return {}

  return {
    'target': match.group(1),
    'eval': float(match.group(2)),
    'k': int(match.group(3)),
  }
