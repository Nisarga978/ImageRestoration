import os
from torch.utils.data import DataLoader
from dataset_RGB import DeblurDataset

def get_training_data(img_options):
    dataset = DeblurDataset(img_options, split='train')
    return DataLoader(dataset,
                      batch_size=img_options.BATCH_SIZE,
                      shuffle=True,
                      num_workers=4,
                      pin_memory=True,
                      drop_last=True)

def get_validation_data(img_options):
    dataset = DeblurDataset(img_options, split='val')
    return DataLoader(dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=4,
                      pin_memory=True,
                      drop_last=False)

def get_test_data(img_options):
    dataset = DeblurDataset(img_options, split='val')
    return DataLoader(dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=4,
                      pin_memory=True,
                      drop_last=False)
