# run_analysis.py

import sys
import os

# Get the absolute path to the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now you can import particle_analysis
from particle_analysis.__main__ import main

if __name__ == "__main__":
    main()
