#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

import poloniex
from predict import *
from train import *
from util import *

def main():
  tickers, periods, targets = parse_command_line(default_tickers=['BTC_ETH'],
                                                 default_periods=['day'],
                                                 default_targets=['high'])

  for ticker in tickers:
    for period in periods:
      for target in targets:
        job = JobInfo('_data', '_zoo', name='%s_%s' % (ticker, period), target=target)
        raw_df = poloniex.get_latest_data(ticker, period=period, depth=100)
        result_df = predict_multiple(job, raw_df=raw_df, rows_to_predict=1)

        raw_df.set_index('date', inplace=True)
        result_df.rename(columns={'True': 'Current-Truth'}, inplace=True)

        info('Latest chart info:', raw_df.tail(2), '', sep='\n')
        info('Prediction for "%s":' % target, result_df, '', sep='\n')

if __name__ == '__main__':
  main()
