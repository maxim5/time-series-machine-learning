#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import numpy as np

from models import *
from util import *


class Processor:
  def __init__(self, dest, start_limit):
    self._dest = dest
    self._min_eval = start_limit
    self._min_params = None

  def process(self, model, params):
    eval = model.eval()
    is_record = eval < self._min_eval
    marker = '!!!' if is_record else '   '
    info('%s Eval=%.6f' % (marker, eval))
    if is_record:
      self._min_eval = eval
      self._min_params = params

      dest_dir = self._dest % (params['target_column'], model.eval(), params['k'])
      model.save(dest_dir)
      self.save_stats(dest_dir, model)
      self.save_params(dest_dir, **params)

  def save_stats(self, dest_dir, model):
    stats_file_name = os.path.join(dest_dir, 'stats.txt')
    with open(stats_file_name, 'w') as file_:
      file_.write(model.stats_str())
      info('Stats   saved to %s' % stats_file_name)

  def save_params(self, dest_dir, **params):
    params_file_name = os.path.join(dest_dir, 'params.txt')
    del params['residual_fun']
    del params['model_class']
    with open(params_file_name, 'w') as file_:
      file_.write(smart_str(params))
      info('Params  saved to %s' % params_file_name)

  def print_result(self):
    info('***')
    info('Best result:')
    info('Eval=%.5f' % self._min_eval)
    info('Params=%s' % str(self._min_params))


def select_best_model(source, dest, params_fun, iterations):
  raw = read_df(source)
  changes = to_changes(raw)
  train_df, test_df = split_train_test(changes)

  processor = Processor(dest=dest, start_limit=1.0)
  for i in xrange(iterations):
    info('Iteration #%d' % (i+1))
    params = params_fun()
    run_model(train_df, test_df, processor=processor, **params)
  processor.print_result()


def run_model(train_df, test_df, processor=None, **params):
  info('Params=%s' % str(params))
  model_class = params['model_class']
  model = model_class(**params)
  with model.session():
    model.fit(train_df)
    model.test(test_df)
    if processor:
      processor.process(model, params)


def simple_run(source, model_class, ks):
  raw = read_df(source)
  changes = to_changes(raw)
  train_df, test_df = split_train_test(changes)
  for k in ks:
    run_model(train_df, test_df,
              model_class=model_class, k=k, target_column='high', residual_fun=lambda pred, truth: np.maximum(pred - truth, 0))


def main():
  # simple_run('data/BTC_ETH_30m.csv', NeuralNetworkModel, [1, 2, 3, 4, 5])

  for ticker in ['BTC_ETH_2h', 'BTC_DGB_2h']:
    for target_column in ['high', 'low']:
      select_best_model(source='data/%s.csv' % ticker,
                        dest='_zoo/%s' % ticker + '__%s__c=%.6f__k=%d',
                        params_fun=lambda : {
                          'target_column': target_column,
                          'residual_fun': lambda pred, truth: np.maximum(pred - truth, 0),

                          'k': np.random.randint(1, 4),
                          'model_class': NeuralNetworkModel,
                          'batch_size': np.random.choice([100, 200, 500, 1000]),
                          'epochs': 50,
                          'hidden_layer': np.random.randint(20, 80),
                          'learning_rate': 10**np.random.uniform(-1.5, 0.2),
                          'init_sigma': 10**np.random.uniform(-7, -2),
                          'lambda': 10**np.random.uniform(-11, -6),
                          'dropout': np.random.uniform(0.1, 0.9),
                        },
                        iterations=20)

if __name__ == '__main__':
  main()
