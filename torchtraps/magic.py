#!/usr/bin/env python
# coding: utf-8

# magic - random helper functions and classes

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from coco_camera_traps_loader import load_image
import json
import os
from barbar import Bar
from random import sample
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def location_splits(json_file):
    """ Get location sets based on train/val split file
    :param json_file: train/val split json file
    :return: train_locations, val_locations
    """
    with open(json_file) as f:
        splits = json.load(f)
    train_locations = set(splits['splits']['train'])
    val_locations = set(splits['splits']['val'])
    return train_locations, val_locations


def exist_on_disk(args, images):
    """
    :param args: config params
    :param images: list of COCO image dicts
    :return: list of COCO image dicts that exist
    """
    if os.path.exists(f'lila/{args.dataset}/images_exist.p'):
        images_exist = pickle.load(open(f'lila/{args.dataset}/images_exist.p', "rb"))
    else:
        images_exist = [i for i in images if os.path.exists(f'{args.data_dir}/{i.get("file_name")}')]
        pickle.dump(images_exist, open( f'lila/{args.dataset}/images_exist.p', "wb" ) )
    return images_exist


def compute_stats(args, images, sample_size=10000):
    """Compute Mean and Std. Dev. of sample of dataset"""
    sample_images = sample(images, sample_size)
    sample_data = StatsDataset(sample_images, args.data_dir, args.resize_dim)
    sample_loader = DataLoader(sample_data, batch_size=32, shuffle=True, num_workers=8)

    mean = 0.0
    std = 0.0

    for idx, data in enumerate(Bar(sample_loader)):
        batch = data.get('image')
        batch_samples = batch.size(0)
        batch = batch.view(batch_samples, batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    mean /= len(sample_loader.dataset)
    std /= len(sample_loader.dataset)

    return mean, std


def save_as_tensors(args, images, mean, std):
    tensor_saver = TensorSaverDataset(images, args.data_dir, mean, std, args.resize_dim)
    saver_dataloader = DataLoader(tensor_saver, batch_size=32, shuffle=False, num_workers=8)
    for idx, data in enumerate(Bar(saver_dataloader)):
        batch = data.get('image')


def wrangler(args):
    """ clean data """
    with open(args.dataset_json) as f:
        dataset = json.load(f)

    images = dataset['images']
    annotations = dataset['annotations']
#     images = exist_on_disk(args, images)

    if args.presave:
        mean, std = compute_stats(args, images, sample_size=10000)
        save_as_tensors(args, images, mean, std)

#     # remove annotations belonging to images that do not exits
#     image_id_set = set([i.get('id') for i in images])
#     anno_exist = [i for i in annotations if i.get('image_id') in image_id_set]

#     # remove images with more than one annotation
#     image_mult_anno = set()
#     for i in anno_exist:
#         if i.get('image_id') in image_id_set:
#             image_id_set.remove(i.get('image_id'))
#         else:
#             image_mult_anno.add(i.get('image_id'))
#     annotations = [i for i in anno_exist if i.get('image_id') not in image_mult_anno]

    # split train/val
    train_locations, val_locations = location_splits(args.splits_json)
    train_images = [i for i in images if i['location'] in train_locations]
    #train_annotations = [i for i in annotations if i['location'] in train_locations]
    val_images = [i for i in images if i['location'] in val_locations]
    #val_annotations = [i for i in annotations if i['location'] in val_locations]

    # Save Training Set
    train_data = dataset
    train_data['images'] = train_images
    #train_data['annotations'] = train_annotations
    assert(len(train_data['images']) == len(train_images))

    with open('lila/SS_S1/SS_S1_train.json', 'w') as outfile:
        json.dump(train_data, outfile)

    # Save Validation Set
    val_data = dataset
    val_data['images'] = val_images
    #val_data['annotations'] = val_annotations
    assert(len(val_data['images']) == len(val_images))

    with open('lila/SS_S1/SS_S1_val.json', 'w') as outfile:
        json.dump(val_data, outfile)


def configure_report(args, run):
    """Configure training report"""
    if args.load_weights is None:
        weights = "scratch"
    else:
        weights = args.load_weights
    experiment_name = f'{args.dataset}_{args.level}_{args.task}_{weights}_{args.arch}_0{run}'
    f = f'reports/{args.dataset}/{experiment_name}.tsv'
    print('epoch\ttraining_loss\ttraining_acc\tvalid_loss\tvalid_acc', file=open(f, "a"))
    return experiment_name, f


# def training_report(args, run):


class TensorSaverDataset(Dataset):
    def __init__(self, datalist, root_dir, mean, std, resize_dim):
        self.datalist = datalist
        self.root_dir = root_dir
        self.mean = mean
        self.std = std
        self.resize_dim = resize_dim

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        imagepath = f'{self.root_dir}/{self.datalist[idx].get("file_name")}'
        image = load_image(imagepath)
        # normalize based on calculated mean and std of image channels
        image = TF.normalize(image, self.mean, self.std)
        torch.save(image, f'{imagepath[:-4]}.pt')

        # as dict
        sample = {'image': image}
        return sample


class StatsDataset(Dataset):
    def __init__(self, datalist, root_dir, resize_dim):
        self.datalist = datalist
        self.root_dir = root_dir
        self.resize_dim = resize_dim

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        imagepath = f'{self.root_dir}/{self.datalist[idx].get("file_name")}'
        image = load_image(imagepath)

        # as dict
        sample = {'image': image}
        return sample

