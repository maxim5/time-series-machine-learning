#!/usr/bin/env python
__author__ = 'maxim'

import numpy as np

from data_util import read_df, to_changes, split_train_test
from linear_model import LinearModel


def main():
  raw = read_df('data/BTC_ETH_30m.csv')
  changes = to_changes(raw)

  train_df, test_df = split_train_test(changes)

  for k in [1, 2, 3, 4, 5]:
    print
    print 'k=%d' % k
    model = LinearModel(train_df, test_df, k=k, target_column='high', residual_fun=lambda pred, truth: np.maximum(pred - truth, 0))
    model.train()
    model.test()


if __name__ == '__main__':
  main()
