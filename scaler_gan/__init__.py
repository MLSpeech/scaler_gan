"""
.. include:: ../README.md
"""

import sys
import git
wt_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
try:
    sys.path.index(wt_dir)
except ValueError:
    sys.path.append(wt_dir)
sys.path.append(wt_dir)