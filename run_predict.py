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
  data_set = to_dataset(changes_df, run_params['k'], run_params['target'], model_info.model_class.DATA_WITH_BIAS)
  data_set.x = data_set.x[-1:]

  with model.session():
    model.restore(model_info.path)
    predicted = float(model.predict(data_set.x))
    info('Predicted change=%.5f' % predicted)
    return predicted


def predict_all_models(changes_df, name, accept):
  home_dir = '_zoo/%s' % name
  models = [dir for dir in os.listdir(home_dir) if accept(dir)]
  predictions = []
  for model in models:
    value = predict_model(changes_df, os.path.join(home_dir, model))
    predictions.append(value)
  info()
  info('Mean predicted value: %.5f' % np.mean(predictions))


def main():
  period = '2h'
  for ticker in ['BTC_ETH', 'BTC_DGB']:
    raw_df = poloniex.get_latest_data(ticker, period=period, depth=10)
    changes_df = to_changes(raw_df)
    predict_all_models(changes_df, '%s_%s' % (ticker, period), lambda name: name.startswith('high_'))


if __name__ == '__main__':
  main()
