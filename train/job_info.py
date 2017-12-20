#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import re


class JobInfo(object):
  def __init__(self, data_dir, zoo_dir, name, target):
    self.data_dir = data_dir
    self.zoo_dir = zoo_dir
    self.name = name
    self.ticker, self.period = _split_name(name)
    self.target = target

  def as_run_params(self):
    return {
      'name': self.name,
      'ticker': self.ticker,
      'period': self.period,
      'target': self.target,
    }

  def get_source_name(self):
    return os.path.join(self.data_dir, '%s.csv' % self.name)

  def get_dest_name(self, eval_, k, id_=None):
    suffix = '_' + id_ if id_ else ''
    return os.path.join(self.zoo_dir, self.name, '%s_eval=%.4f_k=%d%s' % (self.target, eval_, k, suffix))

  def get_current_eval_results(self):
    directory = os.path.join(self.zoo_dir, self.name)
    return parse_eval(directory, lambda info: info['target'] == self.target)


def _split_name(name):
  idx = name.rindex('_')
  return name[:idx], name[idx+1:]


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

  match = re.match('([a-z]+)_eval=([0-9.]+)_k=([0-9]+)(_\w*)?', file_name)
  if not match:
    return {}

  return {
    'target': match.group(1),
    'eval': float(match.group(2)),
    'k': int(match.group(3)),
  }
