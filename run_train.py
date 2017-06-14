#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np

from models import *
from train import JobRunner, JobInfo


def main():
  for ticker in ['BTC_ETH_2h', 'BTC_DGB_2h']:
    for target in ['high']:
      job_info = JobInfo('_data', '_zoo', ticker=ticker, target=target)
      job_runner = JobRunner(job_info, limit='auto')
      job_runner.iterate(iterations=10, params_fun=lambda : {
        'target': job_info.target,
        'k': np.random.choice(range(1, 9)),
        'model_class': NeuralNetworkModel,
        'model_params': {
          'batch_size': np.random.choice([1000, 2000, 4000]),
          'epochs': 100,
          'hidden_layer': np.random.randint(50, 400),
          'learning_rate': 10 ** np.random.uniform(-2.0, -0.0),
          'init_sigma': 10 ** np.random.uniform(-8, -2),
          'cost_func': np.random.choice(['l1', 'l2']),
          'lambda': 10 ** np.random.uniform(-12, -6),
          'dropout': np.random.uniform(0.1, 0.9),
        }
      })
      job_runner.print_result()


if __name__ == '__main__':
  main()
