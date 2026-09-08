import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

__all__ = ("models", "methods", "Layers")

from netjet.models import * 
from netjet.methods import * 
from netjet.Layers import *

# global statics and variables
no_local_storage = False
NAME = 'netjet'
LOCAL_STORAGE_PATH = f"{NAME}/local_storage"

if __name__ == NAME:
    message = 'NetJet v1.0.0 is successfully working.'
    print(message)