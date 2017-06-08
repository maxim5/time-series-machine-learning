#!/usr/bin/env python
__author__ = 'maxim'


from data_util import to_dataset
from train_util import print_residuals


class Model():
  def __init__(self, **params):
    self.k = params.get('k')
    self.target_column = params.get('target_column')
    self.residual_fun = params.get('residual_fun')


  def fit(self, train_df):
    x, y = to_dataset(train_df, self.k, target_column=self.target_column)
    self._fit(x, y)

    prediction = self.predict(x)
    residuals = self.residuals(prediction, y)
    print_residuals(residuals, y)


  def _fit(self, x, y):
    raise NotImplementedError


  def predict(self, x):
    raise NotImplementedError


  def residuals(self, prediction, truth):
    return self.residual_fun(prediction, truth)


  def test(self, test_df):
    x, y = to_dataset(test_df, self.k, target_column=self.target_column)
    prediction = self.predict(x)
    residuals = self.residuals(prediction, y)
    print_residuals(residuals, y)
