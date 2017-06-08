#!/usr/bin/env python
__author__ = 'maxim'

import numpy as np

from data_util import read_df, to_changes, split_train_test
from linear_model import LinearModel
from xgboost_model import XgbModel


def main():
  raw = read_df('data/BTC_ETH_30m.csv')
  changes = to_changes(raw)

  train_df, test_df = split_train_test(changes)

  for k in [1, 2, 3, 4, 5]:
    params = {
      'k': k,
      'target_column': 'high',
      'residual_fun': lambda pred, truth: np.maximum(pred - truth, 0),
    }
    model = XgbModel(**params)

    print '\nModel=%s k=%d' % (model.__class__, k)

    model.fit(train_df)
    model.test(test_df)


if __name__ == '__main__':
  main()
