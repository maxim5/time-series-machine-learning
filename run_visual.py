#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

from itertools import izip, count

import matplotlib.pyplot as plt
import pandas as pd

from predict import *
from train import *
from util import *

plt.style.use('ggplot')


def predict_multiple(job_info, last_rows):
  debug('Predicting %s target=%s' % (job_info.name, job_info.target))

  raw_df = read_df(job_info.get_source_name())
  raw_targets = raw_df[job_info.target][-(last_rows + 1):].reset_index(drop=True)

  changes_df = to_changes(raw_df)
  target_changes = changes_df[job_info.target][-last_rows:].reset_index(drop=True)
  dates = changes_df.date[-last_rows:].reset_index(drop=True)

  df = changes_df[:-1]  # the data for models is shifted by one: the target for the last row is unknown

  ensemble = Ensemble.ensemble_top_models(job_info)
  predictions = ensemble.predict_aggregated(df, last_rows=last_rows)

  result = []
  for idx, date, prediction_change, target_change in izip(count(), dates, predictions, target_changes):
    debug('%%-change on %s: predict=%+.5f target=%+.5f' % (date, prediction_change, target_change))

    # target_change is approx. raw_targets[idx + 1] / raw_targets[idx] - 1.0
    raw_target = raw_targets[idx + 1]
    raw_predicted = (1 + prediction_change) * raw_targets[idx]
    debug('   value on %s: predict= %.5f target= %.5f' % (date, raw_predicted, raw_target))

    result.append({'Time': date, 'Prediction': raw_predicted, 'True': raw_target})

  result_df = pd.DataFrame(result)
  result_df.set_index('Time', inplace=True)
  result_df.index.names = ['']
  return result_df


def main():
  train_date = None
  tickers, periods, targets = parse_command_line(default_tickers=['BTC_ETH', 'BTC_LTC'],
                                                 default_periods=['day'],
                                                 default_targets=['high'])

  for ticker in tickers:
    for period in periods:
      for target in targets:
        job = JobInfo('_data', '_zoo', name='%s_%s' % (ticker, period), target=target)
        result_df = predict_multiple(job, last_rows=120)
        result_df.plot(title=job.name)

        if train_date is not None:
          x = train_date
          y = result_df['True'].min()
          plt.axvline(x, color='k', linestyle='--')
          plt.annotate('Training stop', xy=(x, y), xytext=(result_df.index.min(), y), color='k',
                       arrowprops={'arrowstyle': "->", 'connectionstyle': 'arc3', 'color': 'k'})

  plt.show()


if __name__ == '__main__':
  main()
