#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np

from models import *
from util.data_util import read_df, to_changes, split_train_test


def select_best_model(source, params_fun, iterations):
  raw = read_df(source)
  changes = to_changes(raw)
  train_df, test_df = split_train_test(changes)

  min_cost = 1e100
  min_params = None

  for i in xrange(iterations):
    print '\nIteration #%d' % (i+1)
    params = params_fun()
    cost = run_model(train_df, test_df, **params)
    if cost < min_cost:
      min_cost = cost
      min_params = params
    if cost < 1.0:
      print 'Promising!'

  print '\n***\nBest result:'
  print 'Cost=%.5f' % min_cost
  print 'Params=%s' % str(min_params)


def run_model(train_df, test_df, **params):
  print 'Params=%s' % str(params)
  model_class = params['model_class']
  model = model_class(**params)
  with model.session():
    model.fit(train_df)
    model.test(test_df)
    print 'Cost=%.6f' % model.cost
  return model.cost


def simple_run(source, model_class, ks):
  raw = read_df(source)
  changes = to_changes(raw)
  train_df, test_df = split_train_test(changes)
  for k in ks:
    run_model(train_df, test_df,
              model_class=model_class, k=k, target_column='high', residual_fun=lambda pred, truth: np.maximum(pred - truth, 0))


def main():
  # simple_run('data/BTC_ETH_30m.csv', NeuralNetworkModel, [1, 2, 3, 4, 5])

  select_best_model(source='data/BTC_ETH_2h.csv',
                    params_fun=lambda : {
                      'target_column': 'high',
                      'residual_fun': lambda pred, truth: np.maximum(pred - truth, 0),

                      'k': np.random.randint(1, 4),
                      'model_class': NeuralNetworkModel,
                      'batch_size': np.random.choice([2000, 4000]),
                      'epochs': 100,
                      'hidden_layer': np.random.randint(40, 80),
                      'learning_rate': 10**np.random.uniform(-5.0, -2.0),
                      'init_sigma': 10**np.random.uniform(-7, -5),
                      'lambda': 10**np.random.uniform(-10, -7),
                      'dropout': np.random.uniform(0.1, 0.5),
                    },
                    iterations=100)

if __name__ == '__main__':
  main()
