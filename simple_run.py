#!/usr/bin/env python
__author__ = 'maxim'


import numpy as np

from data_util import read_df, to_changes, split_train_test
from linear_model import LinearModel
from xgboost_model import XgbModel


def select_best_model(source, iterations):
  raw = read_df(source)
  changes = to_changes(raw)
  train_df, test_df = split_train_test(changes)

  min_cost = 1e100
  min_params = None

  for _ in xrange(iterations):
    k = np.random.randint(1, 10)
    model_class = np.random.choice([LinearModel, XgbModel])

    cost = run_model(model_class, k, train_df, test_df)
    if cost < min_cost:
      min_cost = cost
      min_params = (model_class, k)

  print '\n***\nBest result:'
  print 'Cost=%.5f' % min_cost
  print 'Params=%s' % str(min_params)


def run_model(model_class, k, train_df, test_df):
  params = {
    'k': k,
    'target_column': 'high',
    'residual_fun': lambda pred, truth: np.maximum(pred - truth, 0),
  }

  print '\nModel=%s k=%d' % (model_class.__name__, k)
  model = model_class(**params)
  model.fit(train_df)
  model.test(test_df)
  print 'Cost=%.5f' % model.cost

  return model.cost


def simple_run(source, model_class, ks):
  raw = read_df(source)
  changes = to_changes(raw)
  train_df, test_df = split_train_test(changes)

  for k in ks:
    run_model(model_class, k, train_df, test_df)


def main():
  # simple_run('data/BTC_ETH_30m.csv', XgbModel, [1, 2, 3, 4, 5])
  select_best_model('data/BTC_ETH_30m.csv', 10)


if __name__ == '__main__':
  main()
