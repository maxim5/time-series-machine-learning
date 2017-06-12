#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import datetime
import numpy as np
import pandas as pd

from data_set import DataSet
from logging import info, debug


pd.set_option('display.expand_frame_repr', False)


def read_df(file_name):
  df = pd.read_csv(file_name)
  df.date = pd.to_datetime(df.date * 1000, unit='ms')
  return df


def to_changes(raw):
  if raw.date.dtype == np.int64:
    raw.date = pd.to_datetime(raw.date * 1000, unit='ms')
  return pd.DataFrame({
    'date': raw.date,
    'time': raw.date.astype(datetime.datetime).apply(lambda val: seconds(val) / (60*60*24)),
    'high': raw.high.pct_change(),
    'low': raw.low.pct_change(),
    'open': raw.open.pct_change(),
    'close': raw.close.pct_change(),
    'vol': raw.volume.replace({0: 1e-5}).pct_change(),
    'avg': raw.weightedAverage.pct_change(),
  }, columns=['date', 'time', 'high', 'low', 'open', 'close', 'vol', 'avg'])


def seconds(datetime_):
  return (datetime_ - datetime_.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()


def to_dataset(df, k, target_column, with_bias):
  df = df[1:]
  df = df.drop(['date'], axis=1)
  target = df[target_column]

  n, cols = df.shape
  windows_num = n - k  # effective window size, including the label, is k + 1

  x = np.empty([windows_num, k * cols + int(with_bias)])
  y = np.empty([windows_num])

  for i in xrange(windows_num):
    window = df[i:i+k]
    row = window.as_matrix().reshape((-1,))
    if with_bias:
      row = np.insert(row, 0, 1)
    x[i] = row
    y[i] = target[i+k]

  debug('data set: x=%s y=%s' % (x.shape, y.shape))
  return DataSet(x, y)


def split_dataset(dataset, ratio=None):
  size = dataset.size
  if ratio is None:
    ratio = _choose_optimal_train_ratio(size)

  mask = np.zeros(size, dtype=np.bool_)
  train_size = int(size * ratio)
  mask[:train_size] = True
  np.random.shuffle(mask)

  train_x = dataset.x[mask, :]
  train_y = dataset.y[mask]

  mask = np.invert(mask)
  test_x = dataset.x[mask, :]
  test_y = dataset.y[mask]

  return DataSet(train_x, train_y), DataSet(test_x, test_y)


def _choose_optimal_train_ratio(size):
  if size > 100000: return 0.95
  if size > 50000:  return 0.9
  if size > 20000:  return 0.875
  if size > 10000:  return 0.85
  if size > 7500:   return 0.825
  if size > 5000:   return 0.8
  if size > 3000:   return 0.775
  if size > 2000:   return 0.75
  if size > 1000:   return 0.7
  return 0.7
