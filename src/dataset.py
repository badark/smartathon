import os
from typing import Dict, Optional, Tuple
import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F, transforms as T
import pandas as pd
from PIL import Image
import json

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target

class SmartathonImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, training=True, horiz_flip=False):
      self.training = training
      self.horiz_flip = horiz_flip
      if self.training:
        self.img_labels = dict(json.load(open(annotations_file, 'r')))
        self.img_keys = list(self.img_labels.keys())
      else:
        self.img_labels = pd.read_csv(annotations_file)
      self.img_dir = img_dir
      self.transform = transform
      self.target_transform = target_transform
      self.image_target_transform = None
      if self.horiz_flip:
        self.image_target_transform = RandomHorizontalFlip(0.5)

    def __len__(self):
      if self.training:
        return len(self.img_keys)
      else:
        return len(self.img_labels.index)

    def __getitem__(self, idx):
      if self.training:
        img_key = self.img_keys[idx]
        img_labels = self.img_labels[img_key] 
      else:
        img_key = self.img_labels.iloc[idx]['image_path']
      
      img_path = os.path.join(self.img_dir, img_key)
      image = Image.open(img_path)
      
      if not  self.training:
        if self.transform:
          image = self.transform(image)
        return image, dict(img_keys=img_key)
      
      # class 0 is reserved for background according to the docs, we have to modify to deal with removing BAD_STREETLIGHT
      label = [int(label_dict['class']) for label_dict in img_labels]
      for i,l in enumerate(label):
        if l < 6:
          label[i]+=1

      # extract the bounding box coordinates
      boxes = list()
      areas = list()
      for label_dict in img_labels:
        xmax, xmin, ymax, ymin = [label_dict.get(key) for key in ['xmax', 'xmin', 'ymax', 'ymin']]
        width, height = image.size
        def adjust_bb(c, lim):
          if c < 0: return 0
          if c >= lim: return lim-1
          return c
        xmax, xmin, ymax, ymin = adjust_bb(xmax, width+1), adjust_bb(xmin, width), adjust_bb(ymax, height+1), adjust_bb(ymin, height)
        if xmin == xmax:
          xmax = xmin+1
        if ymin == ymax:
          ymax = ymin+1
        boxes.append([xmin, ymin, xmax, ymax])
        areas.append((ymax-ymin)*(xmax-xmin))
      boxes = torch.as_tensor(boxes, dtype=torch.float32)
      areas = torch.as_tensor(areas, dtype=torch.float32)
      
      if self.transform:
        image = self.transform(image)
      if self.target_transform:
        label = self.target_transform(label)
      label = torch.as_tensor(label, dtype=torch.int64)
      target = dict(labels=label, boxes=boxes, areas=areas, img_keys=img_key)
      if self.image_target_transform:
        image, target = self.image_target_transform(image, target)
      return image, target
