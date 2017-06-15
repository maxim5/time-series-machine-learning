#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os

from models import Evaluator
from util import *


class JobRunner:
  def __init__(self, job_info, limit):
    raw_df = read_df(job_info.get_source_name())
    changes_df = to_changes(raw_df)

    self._job_info = job_info
    self._changes_df = changes_df
    self._min_eval = _resolve_limit(limit, job_info)
    self._min_params = None


  def single_run(self, **params):
    info('Params=%s' % smart_str(params))

    model_class = params['model_class']
    with_bias = model_class.DATA_WITH_BIAS

    data_set = to_dataset(self._changes_df, k=params['k'], target_column=params['target'], with_bias=with_bias)
    train, test = split_dataset(data_set)

    model_params = params['model_params']
    model_params['features'] = train.x.shape[1]

    run_params = {key: params[key] for key in ['k', 'target']}
    run_params.update(self._job_info.as_run_params())

    model = model_class(**model_params)
    evaluator = Evaluator()

    with model.session():
      model.fit(train)
      train_eval, train_stats = evaluator.eval(model, train)
      info('Train result:\n%sEval=%.6f' % (evaluator.stats_str(train_stats), train_eval))

      test_eval, test_stats = evaluator.eval(model, test)
      is_record = test_eval < self._min_eval
      marker = ' !!!' if is_record else ''
      info('Test result:\n%sEval=%.6f%s\n' % (evaluator.stats_str(test_stats), test_eval, marker))

      if is_record:
        self._min_eval = test_eval
        self._min_params = params

        dest_dir = self._job_info.get_dest_name(test_eval, params['k'])
        model.save(dest_dir)
        _save_to(dest_dir, 'stats.txt', evaluator.stats_str(test_stats))
        _save_to(dest_dir, 'model-params.txt', smart_str(model_params))
        _save_to(dest_dir, 'run-params.txt', smart_str(run_params))


  def iterate(self, iterations, params_fun):
    for i in xrange(iterations):
      info('Iteration %s#%d' % (self._job_info.name, i + 1))
      params = params_fun()
      self.single_run(**params)


  def print_result(self):
    if self._min_params is None:
      warn('Nothing found!!!\n')
    else:
      info('*** Best result: ***\n' + \
           'Eval=%.5f\n' % self._min_eval + \
           'Params=%s\n' % str(self._min_params))


def _save_to(dest_dir, name, data):
  path = os.path.join(dest_dir, name)
  with open(path, 'w') as file_:
    file_.write(data)
    info('Data saved: %s' % path)


def _resolve_limit(limit, job_info):
  if not callable(limit):
    return limit

  results = job_info.get_current_eval_results()
  if results:
    info('Auto-detected current results for %s: %s' % (job_info.name, results))
    value = limit(results)
    info('Using the limit=%.5f' % value)
    return value

  info('Using the default limit=1.0')
  return 1.0
