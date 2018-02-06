#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

import matplotlib.pyplot as plt

from predict import predict_multiple
from train import JobInfo
from util import parse_command_line, read_df

plt.style.use('ggplot')

def main():
  train_date = None
  tickers, periods, targets = parse_command_line(default_tickers=['BTC_ETH', 'BTC_LTC'],
                                                 default_periods=['day'],
                                                 default_targets=['high'])

  for ticker in tickers:
    for period in periods:
      for target in targets:
        job = JobInfo('_data', '_zoo', name='%s_%s' % (ticker, period), target=target)
        result_df = predict_multiple(job, raw_df=read_df(job.get_source_name()), rows_to_predict=120)
        result_df.index.names = ['']
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
