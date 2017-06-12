#!/usr/bin/env python
__author__ = 'maxim'


from xgboost import XGBRegressor

from model import Model


class XgbModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)
    self._model = XGBRegressor()


  def fit(self, train):
    self._model.fit(train.x, train.y)


  def predict(self, x):
    return self._model.predict(x)
