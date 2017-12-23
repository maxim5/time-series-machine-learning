#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import re
import sys
from logging import info, warn


DEFAULT_TICKERS = ['BTC_ETH', 'BTC_LTC', 'BTC_XRP', 'BTC_DGB', 'BTC_STR', 'BTC_ZEC']


def get_tickers(default=DEFAULT_TICKERS):
  args = sys.argv[1:]
  if not args:
    if not default:
      warn('No tickers provided. Example usage: ./runner.py BTC_ETH')
      return default
    info('Hint: you can provide the tickers in the arguments, like ./runner.py BTC_ETH BTC_LTC')
    info('Using default tickers: %s\n' % ', '.join(['"%s"' % ticker for ticker in default]))
    return default
  for arg in args:
    if not re.match(r'[A-Z]{2,5}_[A-Z]{2,5}', arg):
      warn('Warning: the argument "%s" doesn\'t look like a ticker. Example: BTC_ETH' % arg)
  return args
