#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


import os

from models import *
from util import *


class ModelInfo:
  def __init__(self, path, model_class, model_params, run_params):
    self.path = path
    self.model_class = model_class
    self.model_params = model_params
    self.run_params = run_params


def get_model_info(path):
  model_params = _read_dict(os.path.join(path, 'model-params.txt'))
  run_params = _read_dict(os.path.join(path, 'run-params.txt'))
  return ModelInfo(path, NeuralNetworkModel, model_params, run_params)


def _read_dict(path):
  with open(path, 'r') as file_:
    content = file_.read()
    return str_to_obj(content)
