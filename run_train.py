#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np

from models import *
from train import JobRunner, JobInfo
import util


def iterate_neural(job_info, job_runner, iterations=10, k_lim=21):
  if NeuralNetworkModel is None:
    return

  job_runner.iterate(iterations, params_fun=lambda: {
    'target': job_info.target,
    'k': np.random.randint(1, k_lim),
    'model_class': NeuralNetworkModel,
    'model_params': {
      'batch_size': np.random.choice([500, 1000, 2000, 4000]),
      'epochs': 100,
      'learning_rate': 10 ** np.random.uniform(-10, -2),
      'init_sigma': 10 ** np.random.uniform(-10, -3),
      'layers': _random_layers(np.random.randint(1, 4)),
      'cost_func': np.random.choice(['l1', 'l2']),
      'lambda': 10 ** np.random.uniform(-8, -2),
    }
  })


def _random_layers(num):
  return [{
    'size': np.random.randint(50, 200),
    'batchnorm': np.random.choice([True, False]),
    'activation_func': np.random.choice(['relu', 'elu', 'sigmoid', 'leaky_relu', 'prelu']),
    'dropout': np.random.uniform(0.1, 0.95),
  } for _ in xrange(num)]


def iterate_rnn(job_info, job_runner, iterations=10):
  if RecurrentModel is None:
    return

  job_runner.iterate(iterations, params_fun=lambda: {
    'target': job_info.target,
    'k': np.random.choice([24, 32, 48, 64, 96]),
    'model_class': RecurrentModel,
    'model_params': {
      'batch_size': np.random.choice([1000, 2000, 4000]),
      'epochs': 100,
      'learning_rate': 10 ** np.random.uniform(-4, -2),
      'layers': [np.random.choice([32, 64, 96]) for _ in xrange(np.random.randint(1, 4))],
      'cell_type': np.random.choice(['lstm', 'gru']),
      'double_state': np.random.choice([True, False]),
      'dropout': np.random.uniform(0.0, 1.0),
      'cost_func': np.random.choice(['l1', 'l2']),
      'lambda': 10 ** np.random.uniform(-10, -6),
    }
  })


def iterate_linear(job_info, job_runner, k_lim=25):
  if LinearModel is None:
    return

  for k in xrange(1, k_lim):
    job_runner.single_run(**{
      'target': job_info.target,
      'k': k,
      'model_class': LinearModel,
      'model_params': {}
    })


def iterate_xgb(job_info, job_runner, iterations=10, k_lim=21):
  if XgbModel is None:
    return

  job_runner.iterate(iterations, params_fun=lambda: {
    'target': job_info.target,
    'k': np.random.randint(1, k_lim),
    'model_class': XgbModel,
    'model_params': {
      'max_depth': np.random.randint(3, 8),
      'n_estimators': np.random.randint(100, 300),
      'learning_rate': 10 ** np.random.uniform(-2, -0.5),
      'gamma': np.random.uniform(0, 0.1),
      'subsample': np.random.uniform(0.5, 1),
    }
  })


def main():
  tickers, periods, targets = util.parse_command_line(default_periods=['day'],
                                                      default_targets=['high'])
  while True:
    for ticker in tickers:
      for period in periods:
        for target in targets:
          job_info = JobInfo('_data', '_zoo', name='%s_%s' % (ticker, period), target=target)
          job_runner = JobRunner(job_info, limit=np.median)
          iterate_linear(job_info, job_runner)
          iterate_neural(job_info, job_runner)
          iterate_xgb(job_info, job_runner)
          iterate_rnn(job_info, job_runner)
          job_runner.print_result()


if __name__ == '__main__':
  main()
