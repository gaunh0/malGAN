import ctypes
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlretrieve
from shutil import copyfile

WINDOWS_PATH = Path(os.environ["SystemRoot"])  # Raises error if not running on Windows
DUMMY_FILE = Path(tempfile.mkstemp()[1])


# Generated script goes below
