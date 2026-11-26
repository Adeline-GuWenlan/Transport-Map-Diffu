import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "src"), os.path.join(HERE, "..", "..", "src")):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(4)
