import os
import inspect
from pathlib import Path

### Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
BASE_DIR   = os.path.join(*Path(SCRIPT_DIR).parts[:-2])
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'models')
SIM_DIR    = os.path.join(BASE_DIR, 'simulations')

if __name__ == "__main__":
    print(BASE_DIR)
    print(MODEL_DIR)
