import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
# Few threads: tests may run next to a training job on the same 8-CPU node. Override with TMAP_TEST_THREADS.
torch.set_num_threads(int(os.environ.get("TMAP_TEST_THREADS", "4")))


@pytest.fixture
def root():
    return ROOT
