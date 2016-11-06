import os
import sys

def nb_setup():
    # Add tools to pythonpath
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)