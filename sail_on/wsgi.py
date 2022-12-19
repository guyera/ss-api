from .api.server import main,set_provider,app
from .api.file_provider import FileProvider
import logging
import os
import sys

class Args:

   def __init__(self, *args,**kwargs):
      self.results_directory = kwargs['results_directory'] if 'results_directory' in  kwargs else './RESULTS'
      self.data_directory = kwargs['data_directory'] if 'data_directory' in  kwargs else './TEST'
      self.log_file = f'{os.getpid()}_wsgi.log'
      self.log_level = logging.INFO

def set_up(args):
    set_provider(
        FileProvider(
            os.path.abspath(args.data_directory),
            os.path.abspath(args.results_directory),
        )
    )
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        filename=args.log_file, filemode="w", level=args.log_level, format=log_format
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Api server starting with provider set to FileProvider")

def create_app(**kwargs):
   set_up(Args(**kwargs))
   return app

