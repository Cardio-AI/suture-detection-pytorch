from __future__ import absolute_import, division, print_function

import os
from options import TestingOptions
from tester import SegTester, SegTesterWithoutMask

if __name__ == "__main__":

    test_options = TestingOptions()
    opts = test_options.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.device_num)

    if opts.no_gt: tester = SegTesterWithoutMask(opts)
    else: tester = SegTester(opts)
    tester.predict()