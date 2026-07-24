import os
import sys

# Insert root directory into sys.path to allow pytest to run successfully without manual PYTHONPATH setting
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
