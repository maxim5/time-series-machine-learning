#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import poloniex
import util


def main():
  tickers = util.get_tickers()
  poloniex.update_selected(tickers)


if __name__ == '__main__':
  main()
