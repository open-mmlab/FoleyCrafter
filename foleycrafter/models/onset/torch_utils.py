# Copied from https://github.com/XYPB/CondFoleyGen/blob/main/specvqgan/onset_baseline/utils/torch_utils.py
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ... import data


# ---------------------------------------------------- #
def load_model(cp_path, net, device=None, strict=True):
    if not device:
        device = torch.device("cpu")
    if os.path.isfile(cp_path):
        print("=> loading checkpoint '{}'".format(cp_path))
        checkpoint = torch.load(cp_path, map_location=device)

        # check if there is module
        if list(checkpoint["state_dict"].keys())[0][:7] == "module.":
            state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:]
                state_dict[name] = v
        else:
            state_dict = checkpoint["state_dict"]
        net.load_state_dict(state_dict, strict=strict)

        print("=> loaded checkpoint '{}' (epoch {})".format(cp_path, checkpoint["epoch"]))
        start_epoch = checkpoint["epoch"]
    else:
        print("=> no checkpoint found at '{}'".format(cp_path))
        start_epoch = 0
        sys.exit()

    return net, start_epoch


# ---------------------------------------------------- #
def binary_acc(pred, target, threshold):
    pred = pred > threshold
    acc = np.sum(pred == target) / target.shape[0]
    return acc


def calc_acc(prob, labels, k):
    pred = torch.argsort(prob, dim=-1, descending=True)[..., :k]
    top_k_acc = torch.sum(pred == labels.view(-1, 1)).float() / labels.size(0)
    return top_k_acc


# ---------------------------------------------------- #


def get_dataloader(args, pr, split="train", shuffle=False, drop_last=False, batch_size=None):
    data_loader = getattr(data, pr.dataloader)
    if split == "train":
        read_list = pr.list_train
    elif split == "val":
        read_list = pr.list_val
    elif split == "test":
        read_list = pr.list_test
    dataset = data_loader(args, pr, read_list, split=split)
    batch_size = batch_size if batch_size else args.batch_size
    dataset.getitem_test(1)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return dataset, loader


# ---------------------------------------------------- #
def make_optimizer(model, args):
    """
    Args:
        model: NN to train
    Returns:
        optimizer: pytorch optmizer for updating the given model parameters.
    """
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False,
        )
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return optimizer


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.schedule == "cos":  # cosine lr schedule
        lr *= 0.5 * (1.0 + np.cos(np.pi * epoch / args.epochs))
    elif args.schedule == "none":  # no lr schedule
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
