"""
Run a script or module in Python isolated mode with only trusted import paths (backend/ + user site-packages).
Use this for anything that reads the downloaded datasets in external_data/ (untrusted files):

  python3 -I scripts/run_isolated.py -m:app.real_data_v7 --stage all
  python3 -I scripts/run_isolated.py path/to/script.py

-I keeps the current directory and the script's directory off sys.path, and ignores PYTHON* env vars.
"""
import os
import runpy
import site
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [BACKEND, site.getusersitepackages()]
target, sys.argv = sys.argv[1], sys.argv[1:]
if target.startswith("-m:"):
    runpy.run_module(target[3:], run_name="__main__", alter_sys=True)
else:
    runpy.run_path(target, run_name="__main__")
