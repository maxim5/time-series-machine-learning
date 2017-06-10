#!/usr/bin/env python
__author__ = 'maxim'


LOG_LEVEL = 1

def log(*msg):
  import datetime
  print '[%s]' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' '.join([str(it) for it in msg])

def set_silence():
  global LOG_LEVEL
  LOG_LEVEL = 10

def set_verbose(level=1):
  global LOG_LEVEL
  LOG_LEVEL = -level

def debug(*msg):
  log_at_level(0, *msg)

def info(*msg):
  log_at_level(1, *msg)

def warn(*msg):
  log_at_level(2, *msg)

def vlog(*msg):
  log_at_level(-1, *msg)

def vlog2(*msg):
  log_at_level(-2, *msg)

def vlog3(*msg):
  log_at_level(-3, *msg)

def log_at_level(level, *msg):
  if level >= LOG_LEVEL:
    log(*msg)
