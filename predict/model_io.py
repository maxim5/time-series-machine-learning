#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os

# noinspection PyUnresolvedReferences
from models import *
from util import *


class ModelInfo(object):
  def __init__(self, path, model_class, model_params, run_params):
    self.path = path
    self.model_class = model_class
    self.model_params = model_params
    self.run_params = run_params

  def is_available(self):
    return self.model_class is not None

  def __repr__(self):
    return repr({'path': self.path, 'class': self.model_class})


def get_model_info(path, strict=True):
  model_params = _read_dict(os.path.join(path, 'model-params.txt'))
  run_params = _read_dict(os.path.join(path, 'run-params.txt'))
  model_class = run_params['model_class']
  resolved_class = globals()[model_class]
  if strict and resolved_class is None:
    raise ModelNotAvailable(model_class)
  return ModelInfo(path, resolved_class, model_params, run_params)


def _read_dict(path):
  with open(path, 'r') as file_:
    content = file_.read()
    return str_to_obj(content)


class ModelNotAvailable(BaseException):
  def __init__(self, model_class, *args):
    super(ModelNotAvailable, self).__init__(*args)
    self.model_class = model_class
