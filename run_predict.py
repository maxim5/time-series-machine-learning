#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import numpy as np

import poloniex
from predict import *
from train import *
from train.evaluator import Evaluator
from util import *


def try_model(path, data_dir='_data', zoo_dir='_zoo'):
  model_info = get_model_info(path)
  run_params = model_info.run_params
  job = JobInfo(data_dir, zoo_dir, run_params['name'], run_params['target'])
  raw_df = read_df(job.get_source_name())
  changes_df = to_changes(raw_df)
  data_set = to_dataset(changes_df, run_params['k'], run_params['target'], model_info.model_class.DATA_WITH_BIAS)

  model = model_info.model_class(**model_info.model_params)
  evaluator = Evaluator()

  with model.session():
    model.restore(model_info.path)
    test_eval, test_stats = evaluator.eval(model, data_set)
    info('Result:\n%sEval=%.6f\n' % (evaluator.stats_str(test_stats), test_eval))


def predict_model(changes_df, path):
  model_info = get_model_info(path)
  run_params = model_info.run_params
  model = model_info.model_class(**model_info.model_params)
  x = to_dataset_for_prediction(changes_df[:-1], run_params['k'], model_info.model_class.DATA_WITH_BIAS)
  x = x[-1:]

  with model.session():
    model.restore(model_info.path)
    predicted = float(model.predict(x))
    info('Predicted change=%.5f' % predicted)
    return predicted


def predict_all_models(changes_df, name, accept):
  home_dir = '_zoo/%s' % name
  models = [dir for dir in os.listdir(home_dir) if accept(dir)]
  if not models:
    info('No models found for %s' % name)
    return

  predictions = []
  for model in models:
    try:
      value = predict_model(changes_df, os.path.join(home_dir, model))
      predictions.append(value)
    except ModelNotAvailable as e:
      warn('Cannot use model from \"%s\": class \"%s\" is not available not this system' % (model, e.model_class))
      warn('Most probable reason is that model dependencies are not met')
  info()
  info('Mean predicted value for %s: %.5f' % (name, np.mean(predictions)))
  info()


def main():
  tickers, periods, targets = parse_command_line(default_tickers=[],
                                                 default_periods=['day'],
                                                 default_targets=['high'])

  for ticker in tickers:
    for period in periods:
      for target in targets:
        raw_df = poloniex.get_latest_data(ticker, period=period, depth=100)
        changes_df = to_changes(raw_df)
        predict_all_models(changes_df, '%s_%s' % (ticker, period), lambda name: name.startswith('%s_' % target))


if __name__ == '__main__':
  main()
