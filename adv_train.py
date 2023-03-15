# script for adversarial training for obtaining scores for plotting ideal curve
import argparse
import torch
import torch.nn as nn
import numpy as np
import models
from torchvision import datasets, transforms
import utils
import attacks
import logging
import os
import time

def lr_schedule(t, epochs, max_lr):
    if t / epochs < 0.5:
        return max_lr
    elif t / epochs < 0.75:
        return max_lr / 10.
    else:
        return max_lr / 100.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--metrics', nargs='+')
    parser.add_argument('--arch', type=str, help='model architecture', default='resnet18')
    parser.add_argument('--normalize', action='store_true', help='whether data is normalized before passing into model or not')
    parser.add_argument('--attack', type=str, default='NoAttack')
    parser.add_argument('--eps', type=float)
    parser.add_argument('--model_dir', type=str, help='path to directory of ckpts', default='adv_train_resnet')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr_max', type=float, default=0.1)
    parser.add_argument('--resume', type=int, help='epoch to resume training from if corresponding ckpt file exists')
    parser.add_argument('--num_iters', type=int, default=20)
    parser.add_argument('--chkpt_iters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--use_syn', action='store_true')
    parser.add_argument('--batch_size_syn', type=int, default=350)
    parser.add_argument('--syn_data_path', type=str, default='data/cifar10_ddpm.npz')
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # load data
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor()])
    train_data = datasets.CIFAR10(root=args.data_dir, train=True, transform=transform, download=True)
    num_classes = 10
    if args.use_syn:
        from data import SynData, combine_dataloaders
        syn_data = SynData(args.syn_data_path, transform=transforms.ToTensor())
        syn_loader = torch.utils.data.DataLoader(dataset=syn_data, batch_size=args.batch_size_syn, shuffle=True, num_workers=4)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_loader = combine_dataloaders(train_loader, syn_loader)
    test_transform = transforms.ToTensor()
    test_data = datasets.CIFAR10(root=args.data_dir, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # set up model
    if args.arch in models.__dict__:
        model = models.__dict__[args.arch](num_classes)
        if args.normalize:
            model = models.apply_normalization(model, mean, std)
        model = nn.DataParallel(model).cuda()
    else:
        raise ValueError('unsupported architecture')

    # initialize adversary
    if args.attack == 'NoAttack':
        atk = attacks.__dict__[args.attack]()
        test_atk = attacks.__dict__[args.attack]()
        args.eps = 0
    else:
        if args.attack == 'FastLagrangePerceptualAttack':
            atk = attacks.__dict__[args.attack](model, bound=args.eps, lpips_model='alexnet_cifar', num_iterations=args.num_iters)
            test_atk = attacks.__dict__['LPIPSAttack'](model, bound=args.eps, lpips_model='alexnet_cifar', num_iterations=args.num_iters)
        else:
            try:
                atk = attacks.__dict__[args.attack](model, dataset_name='cifar',bound=args.eps, num_iterations=args.num_iters)
            except:
                atk = attacks.__dict__[args.attack](model, bound=args.eps, num_iterations=args.num_iters)
            test_atk = atk

    # run training
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, 'at_{}_{}_output.log'.format(args.attack, args.eps))),
            logging.StreamHandler()
        ])

    logger.info(args)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    best_test_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.model_dir, f'{args.arch}_{args.attack}_{args.eps}_model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.model_dir, f'{args.arch}_{args.attack}_{args.eps}_opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        if os.path.exists(os.path.join(args.model_dir, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(args.model_dir, f'{args.arch}_{args.attack}_{args.eps}_model_best.pth'))['test_robust_acc']
    else:
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            model.eval()
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i + 1) / len(train_loader), args.epochs, max_lr=args.lr_max)
            opt.param_groups[0].update(lr=lr)
            X_adv = atk(X, y)

            model.train()
            robust_output = model(X_adv)
            robust_loss = criterion(robust_output, y)

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            output = model(X)
            loss = criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            # Random initialization
            X_adv = test_atk(X, y)

            robust_output = model(X_adv)
            robust_loss = criterion(robust_output, y)

            output = model(X)
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()
        logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
            epoch, train_time - start_time, test_time - train_time, lr,
            train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
            test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)

        # save checkpoint
        if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'{args.arch}_{args.attack}_{args.eps}_model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(args.model_dir, f'{args.arch}_{args.attack}_{args.eps}_opt_{epoch}.pth'))

        # save best
        if test_robust_acc/test_n > best_test_robust_acc:
            torch.save({
                    'state_dict':model.state_dict(),
                    'test_robust_acc':test_robust_acc/test_n,
                    'test_robust_loss':test_robust_loss/test_n,
                    'test_loss':test_loss/test_n,
                    'test_acc':test_acc/test_n,
                }, os.path.join(args.model_dir, f'{args.arch}_{args.attack}_{args.eps}_model_best.pth'))
            best_test_robust_acc = test_robust_acc/test_n
    

if __name__== '__main__':
    main()
