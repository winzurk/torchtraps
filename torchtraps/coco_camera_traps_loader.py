import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import jpeg4py as jpeg
import numpy as np
import os


class CocoCameraTrapsDetection(Dataset):
    """`Coco Camera Traps Detection Dataset. """

    def __init__(self, root, annFile, transform=None, target_transform=None, img_size=(300, 500)):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids, self.anns = get_ids_and_anns(self.root, self.coco)
        self.id_dict = get_id_dict(self.coco, self.ids, self.anns) # get dict to convert ids
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        self.cats = self.coco.cats

    def __getitem__(self, idx):
        """Get pair of image and target"""
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = self.id_dict[img_id]
        #ann_ids = coco.getAnnIds(imgIds=img_id)
        
        # get first annotation only for now
        anns = coco.loadAnns(ann_ids)[0]
        
        target = {}
        # get bounding box and rescale to image size
        bbox = anns['bbox']
        og_width = self.coco.imgs[img_id]['width']
        og_height = self.coco.imgs[img_id]['height']
        w_percent = self.img_size[1] / og_width
        h_percent = self.img_size[0] / og_height
        bbox = [bbox[0] * w_percent, bbox[1] * h_percent, bbox[2] * w_percent, bbox[3] * h_percent]
        target['boxes'] =  torch.unsqueeze(torch.as_tensor(bbox, dtype=torch.float32), 0)
        target['image_id'] = torch.tensor([idx])
        target['labels'] = torch.tensor(anns['category_id'])

        # load image
        path = self.coco.imgs[img_id]['file_name']
        path = f'{self.root}/{path}'
        image = load_image(path, self.img_size)
        
        #if self.transform is not None:
        #    img = self.transform(img)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        return {'image': image, 'target': target}


    def __len__(self):
        return len(self.ids)


class CocoCameraTrapsDataset(Dataset):
    """COCO Camera Traps Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, transform=None, pre_saved_tensors=False, level='species', task='classification',
                 img_size=(300, 500)):
        """Set the path for images and labels.

        Args:
            root: image directory.
            json: coco camera traps annotation file path.
            transform: image transformations.
            pre_saved_tensors: denotes whether tensors have been pre-saved to disk
            level: animal or species
            task: classification or detection
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.cats = self.coco.cats
        self.transform = transform
        self.pre_saved_tensors = pre_saved_tensors
        self.level = level
        self.task = task
        self.img_size = img_size

    def __getitem__(self, index):
        """Returns one data pair (image and labels)."""
        ann_id = self.ids[index]
        cat_id = self.coco.anns[ann_id]['category_id']
        if self.level == 'animal' and cat_id != 0: 
            cat_id == 1
        img_id = self.coco.anns[ann_id]['image_id']
        if self.task == 'detection':
            bbox = self.coco.anns[ann_id]['bbox']
            og_width = self.coco.imgs[img_id]['width']
            og_height = self.coco.imgs[img_id]['height']
            w_percent = self.img_size[1] / og_width
            h_percent = self.img_size[0] / og_height
            bbox = [bbox[0] * w_percent, bbox[1] * h_percent, bbox[2] * w_percent, bbox[3] * h_percent]
        class_label = self.cats[cat_id]['name']
        path = self.coco.imgs[img_id]['file_name']
        path = f'{self.root}/{path}'

        # load image from path
        try:
            if self.pre_saved_tensors:
                image = torch.load(f'{path[:-4]}.pt')
            else:
                image = load_image(path, self.img_size)
        # load zeros if image corrupted
        except:
            image = torch.zeros(3, self.img_size[0], self.img_size[1])

        # data augmentations
        if self.transform is not None:
            image = self.transform(image)

        # return item dictionary
        if self.task == 'detection':
            return {'image': image, 'target': cat_id, 'bbox': bbox, 'class': class_label}
        else:
            return {'image': image, 'target': cat_id, 'class': class_label}

    def __len__(self):
        return len(self.ids)


def load_train_val(args):
    # Load Dataset
    dataset_name = f'lila/{args.dataset}/{args.dataset}'
    if args.task == 'detection':
        train_data = CocoCameraTrapsDetection(args.data_dir, f'{dataset_name}_train.json', transform=None)
        val_data = CocoCameraTrapsDetection(args.data_dir, f'{dataset_name}_val.json', transform=None)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    else:
        train_data = CocoCameraTrapsDataset(args.data_dir, f'{dataset_name}_train.json', transform=None)
        val_data = CocoCameraTrapsDataset(args.data_dir, f'{dataset_name}_val.json', transform=None)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    if args.level == 'species':
        num_classes = len(train_data.cats)
    else:
        num_classes = 2
    # Return Dataset Dict
    return {'train_data': train_data,
            'train_dataloader': train_dataloader,
            'val_data': val_data,
            'val_dataloader': val_dataloader,
            'num_classes': num_classes}


def load_test(args):
    # Load Dataset
    dataset_name = f'lila/{args.dataset}/{args.dataset}'
    test_data = CocoCameraTrapsDataset(args.data_dir, f'{dataset_name}_train.json', transform=None)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # Return Dataset Dict
    return {'test_data': test_data,
            'test_dataloader': test_dataloader,
            'num_classes': len(test_data.cats)}


def load_image(path, resize_dim):
    image = jpeg.JPEG(path).decode()
    image = Image.fromarray(image).convert('RGB')
    #image = torch.from_numpy(x)
    image = TF.resize(image, resize_dim)
    image = TF.to_tensor(image)
    return image


def get_id_dict(coco, ids, anns):
    id_dict = dict.fromkeys(ids)
    id_set = set(ids)
    for i in anns:
        img_id = coco.anns[i]['image_id']
        if img_id in id_set:
            value = id_dict[img_id]
            if value is not None:
                value.append(i)
            else:
                value = []
                value.append(i)
            pair = {img_id: value}
            id_dict.update(pair)
    return id_dict


def get_ids_and_anns(root, coco):
    imgs = list(coco.imgs.keys())
    anns = list(coco.anns.keys())
    
    # remove annotations not for images
    imgs_set = set(imgs)
    anns = [i for i in anns if coco.anns[i]['image_id'] in imgs_set]
    
    # remove images with no annotations
    imgs = list(set([coco.anns[i]['image_id'] for i in anns]))
    
    # remove images not in season 1
    imgs = [i for i in imgs if coco.imgs[i]['season'] == 'S1']
    
    # check images exist on disk
    imgs = [i for i in imgs if os.path.exists(f'{root}/{coco.imgs[i].get("file_name")}')]
    
    # remove annotations not for images
    imgs_set = set(imgs)
    anns = [i for i in anns if coco.anns[i]['image_id'] in imgs_set]
    return imgs, anns

