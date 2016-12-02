import os
import shutil

class RunDirectories():
  def __init__(self, copy_source=True):
    if os.path.isdir('runs'):
      previous_runs = [dir for dir in os.listdir('runs') if 'run-' in dir]
    else:
      previous_runs = []

    if previous_runs:
      last_run = max([int(run[4:]) for run in previous_runs])
      self._this_run = last_run + 1
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

  def _make_run_dir(self, child):
    dir = 'runs/run-%d/%s/' % (self._this_run, child)
    if not os.path.isdir(dir):
      os.makedirs(dir)
    return dir
