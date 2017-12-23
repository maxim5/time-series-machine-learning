#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import re
import sys

from logging import info, warn


DEFAULT_TICKERS = ['BTC_ETH', 'BTC_LTC', 'BTC_XRP', 'BTC_DGB', 'BTC_STR', 'BTC_ZEC']
DEFAULT_PERIODS = ['4h', 'day']
DEFAULT_TARGETS = ['high']


def parse_command_line(default_tickers=DEFAULT_TICKERS,
                       default_periods=DEFAULT_PERIODS,
                       default_targets=DEFAULT_TARGETS):
  args = sys.argv[1:]
  options = [arg[2:].split('=') for arg in args if arg.startswith('--')]
  args = [arg for arg in args if not arg.startswith('--')]

  # local import to avoid cyclic dependencies
  from poloniex.fetch_data import COLUMNS, PERIODS
  tickers = get_tickers(args, default_tickers)
  periods = parse_option('period', options, PERIODS, default_periods)
  targets = parse_option('target', options, COLUMNS, default_targets)
  return tickers, periods, targets


def get_tickers(args, default=DEFAULT_TICKERS):
  if not args:
    if not default:
      warn('No tickers provided. Example usage: ./runner.py BTC_ETH')
      return default
    info('Hint: you can provide the tickers in the arguments, like ./runner.py BTC_ETH BTC_LTC')
    info('Using default tickers: %s\n' % pretty_list(default))
    return default
  for arg in args:
    if not re.match(r'[A-Z]{2,5}_[A-Z]{2,5}', arg):
      warn('Warning: the argument "%s" doesn\'t look like a ticker. Example: BTC_ETH' % arg)
  info('Using tickers: %s\n' % pretty_list(args))
  return args


def parse_option(name, options, valid, default):
  for option in options:
    if len(option) == 2 and option[0] == name:
      values = [value for value in option[1].split(',') if value in valid]
      if values:
        info('Using %s: %s' % (name, pretty_list(values)))
        return values
      else:
        warn('No valid %s specified. Using default: %s' % (name, pretty_list(default)))
  return default


def pretty_list(values):
  return ', '.join(['"%s"' % value for value in values])
