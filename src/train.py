from __future__ import absolute_import, division, print_function

import os
import time
import torch
import numpy as np
import random

from utils import get_hrs_min_sec
from options import TrainingOptions
from trainer import SegmentationTrainer


if __name__ == "__main__":

    seed = 10

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prev_state = torch.random.get_rng_state()
    train_options = TrainingOptions()
    opts = train_options.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.device_num)
    trainer = SegmentationTrainer(opts)

    before_train_time = time.time()
    trainer.train()
    train_duration = time.time() - before_train_time

    hrs, min, sec = get_hrs_min_sec(train_duration)
    print('Total train duration: {} hrs {} min {} sec'.format(hrs, min, sec))
    torch.random.set_rng_state(prev_state)
