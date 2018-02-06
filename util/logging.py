#!/usr/bin/env python

from __future__ import print_function

__author__ = 'maxim'


LOG_LEVEL = 1

def log(*msg, **kwargs):
  import datetime
  sep = kwargs.get('sep', ' ')
  print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sep.join([str(it) for it in msg]))

def set_silence():
  global LOG_LEVEL
  LOG_LEVEL = 10

def set_verbose(level=1):
  global LOG_LEVEL
  LOG_LEVEL = -level

def is_debug_logged():
  return LOG_LEVEL <= 0

def is_info_logged():
  return LOG_LEVEL <= 1

def debug(*msg, **kwargs):
  log_at_level(0, *msg, **kwargs)

def info(*msg, **kwargs):
  log_at_level(1, *msg, **kwargs)

def warn(*msg, **kwargs):
  log_at_level(2, *msg, **kwargs)

def vlog(*msg, **kwargs):
  log_at_level(-1, *msg, **kwargs)

def vlog2(*msg, **kwargs):
  log_at_level(-2, *msg, **kwargs)

def vlog3(*msg, **kwargs):
  log_at_level(-3, *msg, **kwargs)

def log_at_level(level, *msg, **kwargs):
  if level >= LOG_LEVEL:
    log(*msg, **kwargs)
