#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import poloniex


def main():
  poloniex.update_selected(pairs=['BTC_ETH', 'BTC_DGB'])


if __name__ == '__main__':
  main()
