import os
import torch
import torchvision
from torchvision import models, transforms, datasets
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import torch.utils.data as data
import copy, tqdm
from ignite.engine import Engine, Events
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pandas as pd

from argparse import ArgumentParser

ROOT_DIR='Datasets/'

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 12  # 11 classes + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# freeze the entire model
for parameter in model.parameters():
  parameter.requires_grad = False

# unfreeze the predictor
for parameter in model.roi_heads.box_predictor.parameters():
  parameter.requires_grad = True

transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

IMG_ROOT = ROOT_DIR+'dataset/resized_images/'

class SmartathonImageDataset(data.Dataset):
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

train_data = SmartathonImageDataset(ROOT_DIR+'dataset/train_split.csv', IMG_ROOT, transform=transforms)

val_data = SmartathonImageDataset(ROOT_DIR+'dataset/val_split.csv', IMG_ROOT, transform=transforms)

test_data = SmartathonImageDataset(ROOT_DIR+'dataset/test_split.csv', IMG_ROOT, transform=transforms)

test2_data = SmartathonImageDataset(ROOT_DIR+'dataset/test2_split.csv', IMG_ROOT, transform=transforms)

BATCH_SIZE = 8

train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(val_data,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
meanAP = MeanAveragePrecision(iou_type="bbox")

cpu_device = torch.device('cpu')
device = torch.device('cuda')
model.to(device)

def train_step(engine, batch):
    model.train()
    images, targets = batch
    images = images.to(device)
    targets = {k: v.to(device) for k,v in targets.items()}
    images = list(image for image in images)
    labels = list(label.unsqueeze(0) for label in targets['labels'])
    boxes = [box.unsqueeze(0) for box in  targets['boxes']]
    targets = [dict(labels=label, boxes=box) for label, box in zip(labels, boxes)]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    lr_scheduler.step()
    return losses.item()

trainer = Engine(train_step)

def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        images, targets = batch
        labels = list(label.unsqueeze(0) for label in targets['labels'])
        boxes = [box.unsqueeze(0) for box in  targets['boxes']]
        targets = [dict(labels=label, boxes=box) for label, box in zip(labels, boxes)]
        images = images.to(device)
        images = list(image for image in images)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        meanAP.update(outputs, targets)

evaluator = Engine(validation_step)

@trainer.on(Events.ITERATION_COMPLETED(every=25))
def log_training_loss(trainer):
    print(f"Epoch[{trainer.state.epoch}] Iteration[{trainer.state.iteration}] Loss: {trainer.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    meanAP.reset()
    evaluator.run(valid_iterator)
    meanAP_metrics = meanAP.compute()
    print(f"Validation Results - Epoch[{trainer.state.epoch}]  meanAP: {meanAP_metrics['map']:.2f} - Time({trainer.state.times[Events.EPOCH_COMPLETED]:.2f}s)")

trainer.run(train_iterator, max_epochs=10)