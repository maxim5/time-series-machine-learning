#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os
import time
import pandas as pd

import api
from util import *


COLUMNS = ['date', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']


def get_all_pairs_list():
  df = api.get_24h_volume()
  return [pair for pair in df.columns if not pair.startswith('total')]


def update_pair(pair, period, dest_dir):
  path = os.path.join(dest_dir, '%s_%s.csv' % (pair, api.period_to_human(period)))
  if os.path.exists(path):
    existing_df = pd.read_csv(path)
    start_time = existing_df.tail(2).date.min()
    debug('Selected start_time=%d' % start_time)
  else:
    existing_df = None
    start_time = 0
  end_time = 2 ** 32

  new_df = api.get_chart_data(pair, start_time, end_time, period)
  if new_df.date.iloc[-1] == 0:
    warn('Error. No data for %s.' % pair)
    return

  if existing_df is not None:
    assert start_time > 0
    df = pd.concat([existing_df[existing_df.date < start_time], new_df])
    df.to_csv(path, index=False, columns=COLUMNS)
    info("Data frame updated: %s" % path)
  else:
    new_df.to_csv(path, index=False, columns=COLUMNS)
    info("Data frame saved to %s" % path)


def update_selected(pairs, periods=api.AVAILABLE_PERIODS, data_dir='_data', sleep=1):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  info('Fetching the pairs: %s' % pairs)
  info('Periods: %s' % [api.period_to_human(period) for period in periods])

  for pair in pairs:
    for period in periods:
      update_pair(pair, period, dest_dir=data_dir)
      time.sleep(sleep)


def update_all(**kwargs):
  pairs = get_all_pairs_list()
  kwargs['pairs'] = pairs
  update_selected(**kwargs)


def get_latest_data(pair, period, depth):
  now = time.time()
  now_seconds = int(now)
  start_time = now_seconds - (depth + 2) * api.period_to_seconds(period)
  end_time = 2 ** 32

  df = api.get_chart_data(pair, start_time, end_time, period)
  if df.date.iloc[-1] == 0:
    warn('Error. No data for %s.' % pair)
    return

  return df
