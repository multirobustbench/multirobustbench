import argparse
import torch
import torch.nn as nn
import numpy as np
import models
from torchvision import datasets, transforms
import utils
import attacks
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/cifar10')
    parser.add_argument('--at_ckpts_dir', type=str, default='at_models')
    parser.add_argument('--out_file', type=str, default='advtrain_accs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
    parser.add_argument('--normalize', action='store_true', help='whether data is normalized before passing into model or not')
    args = parser.parse_args()

    # load data
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.ToTensor()
    test_data = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform, download=True)
    num_classes = 10
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # set up model
    if args.arch in models.__dict__:
        model = models.__dict__[args.arch](num_classes)
        if args.normalize:
            model = models.apply_normalization(model, mean, std)
    else:
        raise ValueError('unsupported architecture')
    model = nn.DataParallel(model).cuda()

    # run overall evals and save to file
    at_accs_all = {}
    for filename in os.listdir(args.at_ckpts_dir):
        print('Evaluating ', filename)
        comps = filename.split("_")
        eps = float(comps[2])
        atk_name = comps[1]
        ckpt = torch.load(os.path.join(args.at_ckpts_dir, filename))
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        if atk_name == 'NoAttack':
            atk = attacks.__dict__[atk_name](model)
            at_accs_all[atk_name] = utils.get_acc_single(test_loader, model, atk)
        else:
            if atk_name == 'LinfAttack' or atk_name == 'L2Attack' or atk_name == 'L1Attack':
                atk_name = 'Auto' + atk_name
            elif atk_name == 'FastLagrangePerceptualAttack':
                atk_name = 'LPIPSAttack'
            if atk_name not in at_accs_all:
                at_accs_all[atk_name] = {}
            if atk_name == 'LPIPSAttack':
                atk = attacks.__dict__[atk_name](model, bound=eps, lpips_model='alexnet_cifar')
            else:
                try:
                    atk = attacks.__dict__[atk_name](model, bound=eps, dataset_name='cifar')
                except:
                    atk = attacks.__dict__[atk_name](model, bound=eps)
            at_accs_all[atk_name][round(eps, 5)] = utils.get_acc_single(test_loader, model, atk)
        utils.save(at_accs_all, args.out_file)


if __name__== '__main__':
    main()
