#!/usr/bin/env python
__author__ = 'maxim'


import os
from xgboost import XGBRegressor
from sklearn.externals import joblib

from model import Model
from util import vlog


class XgbModel(Model):
  def __init__(self, **params):
    Model.__init__(self, **params)
    del params['features']
    self._model = XGBRegressor(**params)

  def fit(self, train):
    self._model.fit(train.x, train.y)

  def predict(self, x):
    return self._model.predict(x)

  def save(self, dest_dir):
    path = os.path.join(dest_dir, 'model.joblib.dat')
    vlog('Saving the model to:', path)
    joblib.dump(self._model, path)

  def restore(self, source_dir):
    path = os.path.join(source_dir, 'model.joblib.dat')
    vlog('Restoring the model from:', path)
    self._model = joblib.load(path)
