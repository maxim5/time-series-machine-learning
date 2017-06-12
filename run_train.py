#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np

from models import *
from train import JobRunner, JobInfo


def main():
  for ticker in ['BTC_ETH_2h', 'BTC_DGB_2h', 'BTC_STR_2h']:
    for target in ['high', 'low']:
        for estimator in ['optimistic', 'pessimistic']:
          job_info = JobInfo('_data', '_zoo', ticker=ticker, target=target, estimator=estimator)
          job_runner = JobRunner(job_info, limit='auto')
          job_runner.iterate(iterations=10, params_fun=lambda : {
            'target': job_info.target,
            'residual_fun': job_info.residual_fun(),
            'k': np.random.choice([1, 2, 3, 4]),
            'model_class': NeuralNetworkModel,
            'model_params': {
              'batch_size': np.random.choice([100, 200, 500, 1000]),
              'epochs': 50,
              'hidden_layer': np.random.randint(20, 80),
              'learning_rate': 10 ** np.random.uniform(-0.5, 0.5),
              'init_sigma': 10 ** np.random.uniform(-7, -2),
              'lambda': 10 ** np.random.uniform(-11, -6),
              'dropout': np.random.uniform(0.1, 0.9),
            }
          })
          job_runner.print_result()


if __name__ == '__main__':
  main()
