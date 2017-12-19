#!/usr/bin/env python
__author__ = 'maxim'

try:
  from linear_model import LinearModel
except ImportError:
  LinearModel = None

try:
  from rnn_model import RecurrentModel
except ImportError:
  RecurrentModel = None

try:
  from nn_model import NeuralNetworkModel
except ImportError:
  NeuralNetworkModel = None

try:
  from xgboost_model import XgbModel
except ImportError:
  XgbModel = None
