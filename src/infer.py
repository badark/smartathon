import os
import torch
import torch.utils.data as data
from ignite.engine import Engine, Events
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from argparse import ArgumentParser

from dataset import SmartathonImageDataset
from model import init_model

def load_from_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main(args):
    model, transforms = init_model(args)
    model = load_from_checkpoint(args.checkpoint_file, model)
    img_dir = args.data_dir+'/resized_images/'
    test_data = SmartathonImageDataset(args.data_dir+'/test_split.csv', img_dir, transform=transforms)
    test_iterator = data.DataLoader(test_data, batch_size=args.batch_size)

    meanAP = MeanAveragePrecision(iou_type="bbox")

    cpu_device = torch.device('cpu')
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    for batch in test_iterator:
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

    meanAP_metrics = meanAP.compute()
    print(f"Test Results - meanAP: {meanAP_metrics['map']:.2f}")

    #write code to create submission file


if __name__ == "__main__":
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model_type', dest='model_type', 
        help='type of model to train, see model.py for supported model types')
    parser.add_argument('-c', '--checkpoint_file', dest='checkpoint_file',
        help='path to checkpoint file')
    parser.add_argument('-d', '--data_dir', dest='data_dir', 
        help='root directory of the dataset')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=8, type=int,
        help="batch size during training")
    parser.add_argument('-o', '--output_prefix', dest='output_prefix',
        help="output path for model checkpoints")
    args = parser.parse_args()
    main(args)
