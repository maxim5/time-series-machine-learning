#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import numpy as np
import os

from models import Evaluator
from util import *


class JobRunner:
  def __init__(self, job_info, limit):
    raw_df = read_df(job_info.get_source_name())
    changes_df = to_changes(raw_df)

    self._job_info = job_info
    self._changes_df = changes_df
    self._min_eval = _resolve_auto(job_info) if limit == 'auto' else limit
    self._min_params = None


  def single_run(self, **params):
    info('Params=%s' % smart_str(params))

    data_set = to_dataset(self._changes_df, params['k'], params['target_column'], params.get('with_bias', False))
    train, test = split_dataset(data_set)

    model_class = params['model_class']
    model = model_class(**params['model_params'])
    evaluator = Evaluator(params['residual_fun'])

    with model.session():
      model.fit(train)
      train_eval, train_stats = evaluator.eval(model, train)
      info('Train result:\n%sEval=%.6f' % (evaluator.stats_str(train_stats), train_eval))

      model.predict(test.x)
      test_eval, test_stats = evaluator.eval(model, test)
      is_record = test_eval < self._min_eval
      marker = ' !!!' if is_record else ''
      info('Test result:\n%sEval=%.6f%s\n' % (evaluator.stats_str(test_stats), test_eval, marker))

      if is_record:
        self._min_eval = test_eval
        self._min_params = params

        dest_dir = self._job_info.get_dest_name(test_eval, params['k'])
        model.save(dest_dir)
        self.save_stats(dest_dir, evaluator.stats_str(test_stats))
        self.save_params(dest_dir, smart_str(params['model_params']))


  def iterate(self, iterations, params_fun):
    for i in xrange(iterations):
      info('Iteration #%d' % (i + 1))
      params = params_fun()
      self.single_run(**params)


  def save_stats(self, dest_dir, stats):
    stats_file_name = os.path.join(dest_dir, 'stats.txt')
    with open(stats_file_name, 'w') as file_:
      file_.write(stats)
      info('Stats   saved to %s' % stats_file_name)


  def save_params(self, dest_dir, params):
    params_file_name = os.path.join(dest_dir, 'model_params.txt')
    with open(params_file_name, 'w') as file_:
      file_.write(params)
      info('Params  saved to %s' % params_file_name)


  def print_result(self):
    if self._min_params is None:
      warn('Nothing found!!!\n')
    else:
      info('*** Best result: ***\n' + \
           'Eval=%.5f\n' % self._min_eval + \
           'Params=%s\n' % str(self._min_params))


def _resolve_auto(job_info):
  results = job_info.get_current_eval_results()
  info('Auto-detected current results for %s: %s' % (job_info.ticker, results))
  if results:
    mean = np.mean(results)
    info('Using the limit=%.5f' % mean)
    return mean
  info('Using the default limit=1.5')
  return 1.5
