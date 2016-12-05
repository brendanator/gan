import tensorflow as tf
import os
import shutil

class RunDirectories():
  def __init__(self, run_dir, copy_source=True):
    self._run_dir = run_dir

    if os.path.isdir(run_dir):
      previous_runs = [dir for dir in os.listdir(run_dir) if 'run-' in dir]
    else:
      previous_runs = []

    if previous_runs:
      self._last_run = max([int(run[4:]) for run in previous_runs])
      self._this_run = self._last_run + 1
    else:
      self._this_run = 1

    print('Run directories will be created at runs/run-%d' % self._this_run)

    if copy_source:
      self.copy_source()

  def copy_source(self):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(src_dir, self._make_run_dir('src') + os.path.relpath(src_dir))

  def summaries(self):
    return self._make_run_dir('summaries')

  def images(self):
    return self._make_run_dir('images')

  def checkpoints(self):
    return self._make_run_dir('checkpoints')

  def latest_checkpoints(self):
    return self._run_dir + '/run-%d/checkpoints' % self._last_run

  def _make_run_dir(self, child):
    dir = self._run_dir  + '/run-%d/%s/' % (self._this_run, child)
    if not os.path.isdir(dir):
      os.makedirs(dir)
    return dir
