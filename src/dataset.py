import os
import pandas as pd
from PIL import Image
import torch

class SmartathonImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = Image.open(img_path)
        # class 0 is reserved for background according to the docs
        label = int(self.img_labels.iloc[idx, 0])+1
        # extract the bounding box coordinates
        xmax, xmin, ymax, ymin = list(map(int, self.img_labels.iloc[idx, 3:7]))
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
        coords = [xmin, ymin, xmax, ymax]
        area = (ymax-ymin)*(xmax-xmin)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        target = dict(labels=label, boxes=torch.tensor(coords), areas=area)
        return image, target