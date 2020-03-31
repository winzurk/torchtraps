
import os
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as TF
from PIL import Image
import json
from torch.nn import Sigmoid


def load_label_dict(path_to_label_file):
    with open('imagenet-simple-labels.json') as f:
        labels = json.load(f)
    return labels


def load_one_image(path_to_image, resize_dim=(300, 500)):
    """Loads one image from path, resize, convert to Torch Tensor"""
    image = Image.open(path_to_image).convert('RGB')
    image = TF.resize(image, resize_dim)
    image = TF.to_tensor(image)
    image = torch.unsqueeze(image, 0)
    return image


def get_images_from_dir(path_to_dir):
    """returns list of images from directory (folder) path"""
    valid_file_set = ['.jpeg', '.JPG', 'JPEG']
    file_list = os.listdir(path_to_dir)
    file_list = [i for i in file_list if i[-4:] in valid_file_set]
    return file_list


def predict(image_list, rel_path, labels):
    """returns predictions on a list of input images"""
    preds_list = []
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    for i in image_list:
        image_path = f'{rel_path}/{i}'
        image = load_one_image(image_path)
        output = model(image)
        _, p = torch.max(output, 1)
        pred_class = labels[p.item()]
        # conf = Sigmoid(output[p])
        preds_list.append([i, pred_class, 0.99])
    return preds_list


def export_to_csv(preds, name='output.csv'):
    """Export predictions to csv file"""
    df = pd.DataFrame(preds, columns=['image_name', 'prediction', 'confidence_score'])
    df.to_csv(name, index=False)


def kachow(path_to_dir, path_to_labels='imagenet-simple-labels.json', output_name='output.csv'):
    labels = load_label_dict(path_to_labels)
    images = get_images_from_dir(path_to_dir)
    predictions = predict(images, path_to_dir, labels)
    export_to_csv(predictions, name=output_name)


# test functionality
if __name__ == "__main__":

    path = '../../NJP'
    kachow(path, output_name='../../NJP_test_output.csv')

