#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List
from yacs.config import CfgNode as CN


class Config(object):
    r"""
    Configuration class for MPRNet training on HIDE dataset.
    """

    def __init__(self, config_yaml: str = "", config_override: List[Any] = []):
        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = True

        # Model parameters
        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.SESSION = 'ps128_bs2'

        # Optimizer parameters
        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 2          # adjust depending on GPU memory
        self._C.OPTIM.NUM_EPOCHS = 200        # total epochs
        self._C.OPTIM.NEPOCH_DECAY = [150]    # epoch to start learning rate decay
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 1e-6
        self._C.OPTIM.BETA1 = 0.5

        # Training dataset parameters
        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.SAVE_IMAGES = False

        # Paths to HIDE dataset
        self._C.TRAINING.TRAIN_DIR = '/content/HIDE_split/train'
        self._C.TRAINING.VAL_DIR = '/content/HIDE_split/val'
        self._C.TRAINING.SAVE_DIR = '/content/checkpoints_HIDE'

        # Patch size
        self._C.TRAINING.TRAIN_PS = 64
        self._C.TRAINING.VAL_PS = 64

        # Optional: pretrained GoPro weights (uncomment if needed)
        # self._C.MODEL.PRETRAINED = '/content/checkpoints_gopro/latest.pth'

        # Override parameter values from YAML file first, then from override list
        if config_yaml:
            self._C.merge_from_file(config_yaml)
        if config_override:
            self._C.merge_from_list(config_override)

        # Make immutable
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path."""
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
