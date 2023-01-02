import sys
import os
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.getcwd())

from utils.test_module import foo

# from test.test_module import foo
from test_module2 import foo2

print(foo(7))
foo2()

