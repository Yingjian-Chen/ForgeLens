import os
import torch
import numpy as np
import random

from tqdm import tqdm

from models.network.net_stage1 import net_stage1
from models.network.net_stage2 import net_stage2
from options.options import Options
from util import Logger, read_yaml
from util import  get_dataset_test
from util import get_bal_sampler
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score, average_precision_score
from torch.cuda.amp import autocast, GradScaler

vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake', 'seeingdark', 'san', 'crn', 'imle', 'guided', 'ldm_200', 'ldm_200_cfg', 'ldm_100', 'glide_100_27', 'glide_50_27', 'glide_100_10', 'dalle']
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# vals = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']
# multiclass = [0, 0, 0, 0, 0, 0, 0, 0]

def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    seed_torch(3407)

    # Options
    options = Options()
    opt = options.parse()
    log_dir = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    Logger(os.path.join(log_dir, 'evaluation_log.log'))

    classes = ''

    # loading
    if opt.eval_stage == 1:
        model = net_stage1()
    else:
        model = net_stage2(opt, train=False)
    model_load = torch.load(opt.weights)
    model.load_state_dict(model_load['model_state_dict'])
    params = []
    for name, p in model.named_parameters():
        if name == "fc.weight" or name == "fc.bias":
            params.append(p)
        else:
            p.requires_grad = False
    model.cuda()
    model.eval()
    print(f"LOAD {opt.weights}!!!!!!")

    scaler = GradScaler()

    accs = []
    aps = []

    # test
    for val_id, val in enumerate(vals):
        sub_test_data_root = '{}/{}'.format(opt.eval_data_root, val)
        if multiclass[val_id] == 1:
            classes = os.listdir(sub_test_data_root)
        else:
            classes = ['']

        val_dataset = get_dataset_test(sub_test_data_root, classes)
        sampler = get_bal_sampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.batch_size,
                                                 shuffle=False,
                                                 drop_last=True,
                                                 sampler=sampler,
                                                 num_workers=opt.num_workers)

        val_accs = []
        val_aps = []
        all_targets = []
        all_pre_probs = []

        for data, target in tqdm(val_loader):
            data, target = data.cuda(), target.cuda()
            with autocast():

                with torch.no_grad():
                    if opt.eval_stage == 1:
                        pre, _ = model(data)
                    else:
                        pre = model(data)

                    pre_prob = torch.sigmoid(pre).cpu()
                    target = target.cpu()

                    acc = accuracy_score(target.numpy(), pre_prob.numpy() > 0.5)
                    ap = average_precision_score(target.numpy(), pre_prob.numpy())

                    val_accs.append(acc)
                    val_aps.append(ap)

                    all_targets.extend(target.numpy())
                    all_pre_probs.extend(pre_prob.numpy())

        val_mean_acc = np.mean(val_accs)
        val_mean_ap = np.mean(val_aps)
        print(
            "({} {:10}) acc: {:.2f}; ap: {:.2f}".format(val_id + 1, val, val_mean_acc * 100, val_mean_ap * 100))
        accs.append(val_mean_acc)
        aps.append(val_mean_ap)

    mean_acc = np.mean(accs) * 100
    mean_ap = np.mean(aps) * 100
    print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format('*', 'Mean', mean_acc, mean_ap))
    print('*'*60)
