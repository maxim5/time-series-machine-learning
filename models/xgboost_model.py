#!/usr/bin/env python
__author__ = 'maxim'


from xgboost import XGBRegressor

from model import Model


class XgbModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)
    self._model = XGBRegressor()


  def _fit(self, x, y):
    self._model.fit(x, y)


  def predict(self, x):
    return self._model.predict(x)
