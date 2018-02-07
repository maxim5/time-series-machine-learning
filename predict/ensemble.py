#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

try:
  from itertools import izip as zip
except ImportError:  # python 3.x
  pass
from itertools import count
import os

import numpy as np
import pandas as pd

from train.job_info import parse_model_infos
from util import *
from .model_io import get_model_info


class Ensemble(object):
  def __init__(self, models):
    self._models = models

  def predict_aggregated(self, df, last_rows=None, reducer=np.mean):
    debug('Models=%d last_rows=%d' % (len(self._models), last_rows))
    changes = [Ensemble.predict_changes_for_model(model_info, df, last_rows) for model_info in self._models]
    changes = np.array(changes)
    vlog('Predicted changes:', changes.shape)
    vlog2('Predicted values:\n', changes[:, :6])
    return reducer(changes, axis=0)

  @staticmethod
  def predict_changes_for_model(model_info, df, last_rows=None):
    run_params = model_info.run_params
    model = model_info.model_class(**model_info.model_params)
    x = to_dataset_for_prediction(df, run_params['k'], model_info.model_class.DATA_WITH_BIAS)

    if last_rows is None:
      assert run_params['k'] <= 100, 'One of the models is using k=%d. Set last rows manually' % run_params['k']
      last_rows = df.shape[0] - 100   # 100 is max k
    assert last_rows <= x.shape[0], 'Last rows is too large. Actual rows: %d' % x.shape[0]
    x = x[-last_rows:]                # take only the last `last_rows` rows
    vlog('Input for prediction:', x.shape)

    with model.session():
      model.restore(model_info.path)
      predicted_changes = model.predict(x)
      vlog('Predicted:', predicted_changes.shape, ' for model:', model_info.path)
      vlog2('Predicted values:', predicted_changes[:20])
      return predicted_changes

  @staticmethod
  def ensemble_top_models(job_info, top_n=5):
    home_dir = os.path.join(job_info.zoo_dir, '%s_%s' % (job_info.ticker, job_info.period))
    models = parse_model_infos(home_dir)
    models.sort(key=lambda d: d['eval'])

    model_paths = [os.path.join(home_dir, d['name']) for d in models]
    models = [get_model_info(path, strict=False) for path in model_paths]
    top_models = [model for model in models if model.is_available()][:top_n]
    return Ensemble(top_models)


def predict_multiple(job_info, raw_df, rows_to_predict, top_models_num=5):
  debug('Predicting %s target=%s' % (job_info.name, job_info.target))

  raw_targets = raw_df[job_info.target][-(rows_to_predict + 1):].reset_index(drop=True)
  changes_df = to_changes(raw_df)
  target_changes = changes_df[job_info.target][-rows_to_predict:].reset_index(drop=True)
  dates = changes_df.date[-rows_to_predict:].reset_index(drop=True)

  df = changes_df[:-1]  # the data for models is shifted by one: the target for the last row is unknown

  ensemble = Ensemble.ensemble_top_models(job_info, top_n=top_models_num)
  predictions = ensemble.predict_aggregated(df, last_rows=rows_to_predict)

  result = []
  for idx, date, prediction_change, target_change in zip(count(), dates, predictions, target_changes):
    debug('%%-change on %s: predict=%+.5f target=%+.5f' % (date, prediction_change, target_change))

    # target_change is approx. raw_targets[idx + 1] / raw_targets[idx] - 1.0
    raw_target = raw_targets[idx + 1]
    raw_predicted = (1 + prediction_change) * raw_targets[idx]
    debug('   value on %s: predict= %.5f target= %.5f' % (date, raw_predicted, raw_target))

    result.append({'Time': date, 'Prediction': raw_predicted, 'True': raw_target})

  result_df = pd.DataFrame(result)
  result_df.set_index('Time', inplace=True)
  return result_df
