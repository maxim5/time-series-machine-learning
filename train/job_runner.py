#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os

from evaluator import Evaluator
from util import *


class JobRunner(object):
  def __init__(self, job_info, limit):
    raw_df = read_df(job_info.get_source_name())
    changes_df = to_changes(raw_df)

    self._job_info = job_info
    self._changes_df = changes_df
    self._min_eval = _resolve_limit(limit, job_info)
    self._min_params = None


  def single_run(self, **params):
    model_class = params['model_class']
    with_bias = model_class.DATA_WITH_BIAS
    expects_k = model_class.EXPECTS_TIME_PARAM

    data_set = to_dataset(self._changes_df, k=params['k'], target_column=params['target'], with_bias=with_bias)
    train, test = split_dataset(data_set)

    adapted = params.copy()
    adapted['model_class'] = params['model_class'].__name__
    info('Params=%s' % smart_str(adapted))

    model_params = params['model_params']
    model_params['features'] = int(train.x.shape[1])
    if expects_k:
      model_params['time_steps'] = params['k']

    run_params = {key: adapted[key] for key in ['k', 'model_class']}
    run_params.update(self._job_info.as_run_params())

    model = model_class(**model_params)
    evaluator = Evaluator(*params.get('eval_params', {}))

    with model.session():
      model.fit(train)
      train_eval, train_stats = evaluator.eval(model, train)
      train_stats_str = evaluator.stats_str(train_stats)
      debug('Train results:\n', train_stats_str)

      test_eval, test_stats = evaluator.eval(model, test)
      test_stats_str = evaluator.stats_str(test_stats)
      is_record = test_eval < self._min_eval
      marker = ' !!!' if is_record else ''
      info('Test results:\n%sEval=%.6f%s\n' % (test_stats_str, test_eval, marker))

      if is_record:
        self._min_eval = test_eval
        self._min_params = params

        dest_dir = self._job_info.get_dest_name(test_eval, params['k'])
        while os.path.exists(dest_dir):
          dest_dir = self._job_info.get_dest_name(test_eval, params['k'], random_id(4))
        os.makedirs(dest_dir)

        model.save(dest_dir)
        _save_to(dest_dir, 'stats.txt', '\n'.join(['# Train results:', train_stats_str, '# Test results:', test_stats_str]))
        _save_to(dest_dir, 'model-params.txt', smart_str(model_params))
        _save_to(dest_dir, 'run-params.txt', smart_str(run_params))
        debug('Model saved to %s' % dest_dir)


  def iterate(self, iterations, params_fun):
    for i in xrange(iterations):
      info('Iteration %s: %s #%d' % (self._job_info.name, self._job_info.target, i + 1))
      params = params_fun()
      self.single_run(**params)


  def print_result(self):
    if self._min_params is None:
      info('Nothing found...\n')
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
    info('Using the hard limit=%.5f' % limit)
    return limit

  results = job_info.get_current_eval_results()
  if results:
    info('Auto-detected current model results for %s: %s' % (job_info.name, results))
    value = limit(results)
    info('Using the limit=%.5f (computed by "%s" function)' % (value, limit.__name__))
    return value

  info('Using the default limit=1.0')
  return 1.0


def random_id(size):
  import string, random
  chars = string.ascii_letters + string.digits
  return ''.join(random.choice(chars) for _ in xrange(size))
