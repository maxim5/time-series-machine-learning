#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'


from models import Evaluator
from predict import *
from train import *
from util import *


def eval(path, data_dir='_data', zoo_dir='_zoo'):
  model_info = get_model_info(path)

  run_params = model_info.run_params
  job = JobInfo(data_dir, zoo_dir, run_params['ticker'], run_params['target'], run_params['estimator'])
  raw_df = read_df(job.get_source_name())
  changes_df = to_changes(raw_df)
  data_set = to_dataset(changes_df, run_params['k'], run_params['target'], False)

  model = model_info.model_class(**model_info.model_params)
  evaluator = Evaluator(job.residual_fun())

  with model.session():
    model.restore(model_info.path)
    test_eval, test_stats = evaluator.eval(model, data_set)
    info('Result:\n%sEval=%.6f\n' % (evaluator.stats_str(test_stats), test_eval))


def main():
  eval(path='_zoo/BTC_ETH_2h/high__opt__eval=1.2445__k=3')
  eval(path='_zoo/BTC_ETH_2h/high__opt__eval=1.2900__k=2')
  eval(path='_zoo/BTC_ETH_2h/high__pes__eval=0.9856__k=4')


if __name__ == '__main__':
  main()
