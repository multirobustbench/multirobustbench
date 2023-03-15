# code for running full evaluation
import argparse
import torch
import torch.nn as nn
import numpy as np
import models
from torchvision import datasets, transforms
import utils
import attacks
import metrics
import time

EVAL_CONFIG_FILE = 'cifar10_eval_config_lpips.yml'

def evaluate(model, out_file, data_dir='data', batch_size=100):
    # load data
    transform = transforms.ToTensor()
    test_data = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # run overall evals and save to file
    ranges = utils.load(EVAL_CONFIG_FILE)
    clean_acc = utils.get_acc_single(test_loader, model, attacks.NoAttack())
    accs = {'NoAttack': clean_acc}
    print('NoAttack', clean_acc)
    #accs = {'NoAttack': utils.get_acc_single(test_loader, model, attacks.NoAttack())}
    for attack_name in ranges:
        print(attack_name)
        atks = []
        eps = np.arange(ranges[attack_name]['start'], ranges[attack_name]['end'] + ranges[attack_name]['step'], ranges[attack_name]['step'])[1:]
        for i in eps:
            if attack_name == "LagrangePerceptualAttack":
                atk = attacks.__dict__[attack_name](model, bound=i, lpips_model='alexnet_cifar')
            else:
                try:
                    atk = attacks.__dict__[attack_name](model, bound=i)
                except: 
                    atk = attacks.__dict__[attack_name](model, bound=i, dataset_name='cifar')
            atks.append(atk)
        accs[attack_name] = utils.get_acc(test_loader, model, atks, eps)
        utils.save(accs, out_file + '_accs')
    return accs

def evaluate_bin(model, out_file, data_dir='data', batch_size=100):
    # load data
    transform = transforms.ToTensor()
    test_data = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # run overall evals and save to file
    ranges = utils.load(EVAL_CONFIG_FILE)
    clean_acc = utils.get_acc_single(test_loader, model, attacks.NoAttack())
    accs = {'NoAttack': clean_acc}
    print('NoAttack:', clean_acc)
    for attack_name in ranges:
        print(attack_name)
        atks = []
        eps = np.arange(ranges[attack_name]['start'], ranges[attack_name]['end'] + ranges[attack_name]['step'], ranges[attack_name]['step'])[1:]
        for i in eps:
            if attack_name == "LagrangePerceptualAttack":
                atk = attacks.__dict__[attack_name](model, bound=i, lpips_model='alexnet_cifar')
            else:
                try:
                    atk = attacks.__dict__[attack_name](model, bound=i)
                except: 
                    atk = attacks.__dict__[attack_name](model, bound=i, dataset_name='cifar')
            atks.append(atk)
        accs[attack_name] = utils.get_acc_bin_search(test_loader, model, atks, eps)
        utils.save(accs, out_file + '_accs') 
    return accs
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--arch', type=str, help='model architecture')
    parser.add_argument('--normalize', action='store_true', help='whether data is normalized before passing into model or not')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--bin', action='store_true', help='run binary variant')
    args = parser.parse_args()

    # load data
    #std = (0.2471, 0.2435, 0.2616)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    num_classes = 10

    # set up model
    if args.arch in models.__dict__:
        model = models.__dict__[args.arch](num_classes=num_classes)
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(utils.filter_state_dict(ckpt))
        if args.normalize:
            model = models.apply_normalization(model, mean, std)
        model = nn.DataParallel(model).cuda()
        model.eval()
    else:
        raise ValueError('unsupported architecture')
    s = time.time()
    if args.bin:
        evaluate_bin(model, args.out_file, data_dir='data', batch_size=args.batch_size)
    else:
        evaluate(model, args.out_file, data_dir='data', batch_size=args.batch_size)
    t = time.time()
    print('Time elapsed', t - s)
if __name__== '__main__':
    main()
