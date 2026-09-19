import os
import sys

# Safely inject the project root directory into python's search path matrix
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
