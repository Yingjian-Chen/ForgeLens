import random
from models.trainer_stage2 import Trainer_stage2
from options.options import Options
from util import *
import util
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from tensorboardX import SummaryWriter
from models.trainer_stage1 import Trainer_stage1
import torch.nn as nn


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Options
    options = Options()
    opt = options.parse()

    seed_torch(opt.seed)

    log_path = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_path, exist_ok=True)

    train_writer_stage1 = SummaryWriter(os.path.join(log_path, 'train_stage_1'))
    train_writer_stage2 = SummaryWriter(os.path.join(log_path, 'train_stage_2'))

    if opt.training_stage == 1:
        Logger(os.path.join(log_path, 'train_stage_1', 'train_stage1.log'))
        options.print_options()
        # Data Load
        train_dataset = get_dataset(opt.train_data_root, opt.train_classes)
        sampler = get_bal_sampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.stage1_batch_size,
                                                   shuffle=False,
                                                   sampler=sampler,
                                                   drop_last=True,
                                                   num_workers=opt.num_workers)

        val_dataset = get_dataset_test(opt.val_data_root, opt.val_classes)
        sampler = get_bal_sampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.stage1_batch_size,
                                                 shuffle=False,
                                                 sampler=sampler,
                                                 drop_last=True,
                                                 num_workers=opt.num_workers)
        model = Trainer_stage1(opt)
        model.train(train_loader, val_loader, nn.BCEWithLogitsLoss(), opt.stage1_epochs,
                    os.path.join(log_path, 'train_stage_1', 'model'), train_writer_stage1)
    else:
        Logger(os.path.join(log_path, 'train_stage_2', 'train_stage2.log'))
        options.print_options()
        # Data Load
        train_dataset = get_dataset(opt.train_data_root, opt.train_classes)
        sampler = get_bal_sampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.stage2_batch_size,
                                                   shuffle=False,
                                                   sampler=sampler,
                                                   drop_last=True,
                                                   num_workers=opt.num_workers)

        val_dataset = get_dataset_test(opt.val_data_root, opt.val_classes)
        sampler = get_bal_sampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.stage2_batch_size,
                                                 shuffle=False,
                                                 sampler=sampler,
                                                 drop_last=True,
                                                 num_workers=opt.num_workers)
        model = Trainer_stage2(opt)
        model.train(train_loader, val_loader, nn.BCEWithLogitsLoss(), opt.stage2_epochs,
                    os.path.join(log_path, 'train_stage_2', 'model'), train_writer_stage2)



