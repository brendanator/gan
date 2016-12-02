import os

class RunDirectories():
  def __init__(self):
    previous_runs = [dir for dir in os.listdir('logs') if 'run' in dir]
    if previous_runs:
      last_run = max([int(run[4:]) for run in previous_runs])
      self._this_run = last_run + 1
    else:
      self._this_run = 1

    print('Run directories will be creates at run-%d' % self._this_run)

  def logs(self):
    return self._make_run_dir('logs')

  def images(self):
    return self._make_run_dir('images')

  def checkpoints(self):
    return self._make_run_dir('checkpoints')

  def _make_run_dir(self, parent):
    dir = '%s/run-%d/' % (parent, self._this_run)
    if not os.path.isdir(dir):
      os.mkdir(dir)
    return dir
