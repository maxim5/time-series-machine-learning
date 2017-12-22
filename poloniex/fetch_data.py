#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import time
import pandas as pd

import api
from util import *


COLUMNS = ['date', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']


def get_all_tickers_list():
  df = api.get_24h_volume()
  return [ticker for ticker in df.columns if not ticker.startswith('total')]


def update_ticker(ticker, period, dest_dir):
  path = os.path.join(dest_dir, '%s_%s.csv' % (ticker, api.period_to_human(period)))
  if os.path.exists(path):
    existing_df = pd.read_csv(path)
    start_time = existing_df.tail(2).date.min()
    debug('Selected start_time=%d' % start_time)
  else:
    existing_df = None
    start_time = 0
  end_time = 2 ** 32

  new_df = api.get_chart_data(ticker, start_time, end_time, period)
  if new_df.date.iloc[-1] == 0:
    warn('Error. No data for %s.' % ticker)
    return

  if existing_df is not None:
    assert start_time > 0
    df = pd.concat([existing_df[existing_df.date < start_time], new_df])
    df.to_csv(path, index=False, columns=COLUMNS)
    info("Data frame updated: %s" % path)
  else:
    new_df.to_csv(path, index=False, columns=COLUMNS)
    info("Data frame saved to %s" % path)


def update_selected(tickers, periods=api.AVAILABLE_PERIODS, data_dir='_data', sleep=0.5):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  info('Fetching the tickers: %s' % tickers)
  info('Periods: %s' % [api.period_to_human(period) for period in periods])

  for ticker in tickers:
    for period in periods:
      update_ticker(ticker, period, dest_dir=data_dir)
      time.sleep(sleep)


def update_all(**kwargs):
  tickers = get_all_tickers_list()
  kwargs['tickers'] = tickers
  update_selected(**kwargs)


def get_latest_data(ticker, period, depth):
  now = time.time()
  now_seconds = int(now)
  start_time = now_seconds - (depth + 2) * api.period_to_seconds(period)
  end_time = 2 ** 32

  df = api.get_chart_data(ticker, start_time, end_time, period)
  if df.date.iloc[-1] == 0:
    warn('Error. No data for %s.' % ticker)
    return

  return df
