#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# See the full API here:
# https://poloniex.com/support/api/

__author__ = 'maxim'


import pandas as pd

from util import *


AVAILABLE_PERIODS = [300, 900, 1800, 7200, 14400, 86400]


def get_chart_data(pair, start_time, end_time, period):
  url = 'https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=%d' % \
        (pair, start_time, end_time, period_to_seconds(period))
  info('Fetching %s: %s' % (pair, url))
  df = pd.read_json(url, convert_dates=False)
  info('Fetched %s (%s)' % (pair, period_to_human(period)))
  return df


def get_24h_volume():
  url = 'https://poloniex.com/public?command=return24hVolume'
  info('Fetching %s' % url)
  return pd.read_json(url)


def period_to_human(period):
  if isinstance(period, basestring):
    return period
  if period == 300:
    return '5m'
  if period == 900:
    return '15m'
  if period == 1800:
    return '30m'
  if period == 7200:
    return '2h'
  if period == 14400:
    return '4h'
  if period == 86400:
    return 'day'
  return str(period)


def period_to_seconds(period):
  if isinstance(period, int):
    return period
  if period == '5m':
    return 300
  if period == '15m':
    return 900
  if period == '30m':
    return 1800
  if period == '2h':
    return 7200
  if period == '4h':
    return 14400
  if period == 'day':
    return 86400
  return int(period)
