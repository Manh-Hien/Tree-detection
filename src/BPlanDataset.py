from __future__ import annotations
from cProfile import label
from email.mime import image
import os
import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import torch
import random
import math
from tools.model import get_model_object_detection
from tools.utils import collate_fn
from pycocotools.coco import COCO
from tools.engine import train_one_epoch, evaluate
from tools.helper import get_transform

root_dir = '/opt/data/team/hien/data/raw'

class BPlanDataset(Dataset):
    def __init__(self, root, transforms=None):

        self.root = root

        json_file = open(
            self.root + "/big_bbox.json")
        self.obj = json.load(json_file)
        self.coco = COCO(self.root + "/BPlan_Berlin_Planzeichen_resized_new.json")
        self.transforms = transforms
        self.category_list = self.get_category_list()
        #print(self.category_list)
        #print(len(self.category_list))
        self.category_id_list = sorted(self.map_cats_to_cat_id().values())
        #print(self.category_id_list)
        #print(len(self.category_id_list))
        self.image_ids = list(sorted(self.get_img_ids_by_cat()))
        #print(self.image_ids)
        #print(len(self.image_ids))
        self.cat_map = self.map_cat_to_internal_cat_ids()
        #print(self.cat_map)

    def get_category_list(self):
        category_list = []
        for cat in self.obj["categories"]:
            category_list.append(cat["name"])
        return category_list

    def map_cats_to_cat_id(self):
        cats_to_cat_id = {}
        for cat_id in self.coco.cats.keys():
            cat_name = self.coco.cats[cat_id]['name']
            cat_id = self.coco.cats[cat_id]['id']
            if cat_name in self.category_list:
                cats_to_cat_id[cat_name] = cat_id
        return cats_to_cat_id

    def get_img_ids_by_cat(self):
        ids = []
        for img_id in self.coco.imgs.keys():
            cats = set(self.coco.imgs[img_id]['category_ids'])
            common_cats = cats.intersection(set(self.category_id_list))
            if len(common_cats) != 0:
                ids.append(img_id)
        return ids 

    def map_cat_to_internal_cat_ids(self):
        map_cat_to_ids = {}
        for i in range(1, len(self.category_id_list) + 1):
            map_cat_to_ids[self.category_id_list[i - 1]] = i
        return map_cat_to_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.image_ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.category_id_list)
        anns = coco.loadAnns(ann_ids)
        root = self.root
        
        img_filename = coco.loadImgs(img_id)[0]['file_name']
        image_path = root + "/files/resized_dataset/" + img_filename

        img = Image.open(os.path.join(self.root, image_path)).convert('RGB')

        num_objs = len(anns)

        boxes = []
        areas = []
        labels = []
        for i in range(num_objs):
            x_min = anns[i]['bbox'][0]
            y_min = anns[i]['bbox'][1]
            x_max = x_min + anns[i]['bbox'][2]
            y_max = y_min + anns[i]['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])
            areas.append(anns[i]['area'])
            cat = anns[i]['category_id']
            labels.append(self.cat_map[cat])
       
    

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([img_id])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd


        if self.transforms is not None:
            img, target = self.transforms(img, target)
        

        return img, target

    
def train_test_split(dataset, test_set_proportion):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_index = math.floor(len(indices) * test_set_proportion)

    train = torch.utils.data.Subset(dataset, indices[:-split_index])
    test = torch.utils.data.Subset(dataset, indices[-split_index:])

    return train, test

    
