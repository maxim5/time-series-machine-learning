#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np

from models import *
from train import JobRunner, JobInfo


def iterate_neural(job_info, job_runner, iterations=10, k_lim=21):
  job_runner.iterate(iterations, params_fun=lambda: {
    'target': job_info.target,
    'k': np.random.randint(1, k_lim),
    'model_class': NeuralNetworkModel,
    'model_params': {
      'batch_size': np.random.choice([500, 1000, 2000, 4000]),
      'epochs': 100,
      'learning_rate': 10 ** np.random.uniform(-10, -2),
      'init_sigma': 10 ** np.random.uniform(-10, -3),
      'layers': random_layers(np.random.randint(1, 4)),
      'cost_func': np.random.choice(['l1', 'l2']),
      'lambda': 10 ** np.random.uniform(-8, -2),
    }
  })


def random_layers(num):
  return [{
    'size': np.random.randint(50, 200),
    'batchnorm': np.random.choice([True, False]),
    'activation_func': np.random.choice(['relu', 'elu', 'sigmoid', 'leaky_relu', 'prelu']),
    'dropout': np.random.uniform(0.1, 0.95),
  } for _ in xrange(num)]


def iterate_lstm(job_info, job_runner, iterations=10):
  job_runner.iterate(iterations, params_fun=lambda: {
    'target': job_info.target,
    'k': np.random.choice([16, 32, 64]),
    'model_class': RecurrentModel,
    'model_params': {
      'batch_size': np.random.choice([500, 1000, 2000, 4000]),
      'epochs': 100,
      'learning_rate': 10 ** np.random.uniform(-10, -2),
      # 'layers': random_layers(np.random.randint(1, 4)),
      'cost_func': np.random.choice(['l1', 'l2']),
      'lambda': 10 ** np.random.uniform(-8, -2),
    }
  })


def iterate_linear(job_info, job_runner, k_lim=25):
  for k in xrange(1, k_lim):
    job_runner.single_run(**{
      'target': job_info.target,
      'k': k,
      'model_class': LinearModel,
      'model_params': {}
    })


def iterate_xgb(job_info, job_runner, iterations=10, k_lim=21):
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
  while True:
    for ticker in ['BTC_ETH', 'BTC_DGB']:
      for period in ['4h', 'day']:
        for target in ['high']:
          job_info = JobInfo('_data', '_zoo', name='%s_%s' % (ticker, period), target=target)
          job_runner = JobRunner(job_info, limit=np.mean)
          iterate_linear(job_info, job_runner)
          iterate_neural(job_info, job_runner)
          iterate_xgb(job_info, job_runner)
          iterate_lstm(job_info, job_runner)
          job_runner.print_result()


if __name__ == '__main__':
  main()
