# flake8: noqa
import os.path as osp
from basicsr.test import test_pipeline

import sys
sys.path.append('.')
sys.path.append('..')
from DiffIR import archs
from DiffIR import data
from DiffIR import models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
