#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import poloniex
import util


def main():
  tickers, periods, _ = util.parse_command_line(default_periods=poloniex.AVAILABLE_PERIODS,
                                                default_targets=[])
  poloniex.update_selected(tickers, periods)


if __name__ == '__main__':
  main()
