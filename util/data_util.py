#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import datetime
import numpy as np
import pandas as pd


# pd.set_option('display.expand_frame_repr', False)


def read_df(file_name):
  df = pd.read_csv(file_name)
  df.date = pd.to_datetime(df.date * 1000, unit='ms')
  return df


def to_changes(raw):
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


def split_train_test(changes, ratio=0.8):
  total_rows = changes.shape[0] - 1
  train_size = int(total_rows * ratio)
  test_size = total_rows - train_size
  print 'total_rows=%d train_size=%d train_size=%d' % (total_rows, train_size, test_size)

  train_df = changes[1:train_size].reset_index(drop=True)
  test_df = changes[train_size:].reset_index(drop=True)
  return train_df, test_df


def to_dataset(df, k, target_column, with_bias):
  df = df.drop(['date'], axis=1)
  target = df[target_column]

  n, cols = df.shape
  windows_num = n - k  # exclude last row, used for the label

  x = np.empty([windows_num, k * cols + int(with_bias)])
  y = np.empty([windows_num])

  for i in xrange(windows_num):
    window = df[i:i+k]
    row = window.as_matrix().reshape((-1,))
    if with_bias:
      row = np.insert(row, 0, 1)
    x[i] = row
    y[i] = target[i+k]

  print 'data set: x=%s y=%s' % (x.shape, y.shape)
  return x, y
