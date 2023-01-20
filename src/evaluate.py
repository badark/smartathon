import os
import torch
import torch.utils.data as data
import pandas as pd
from ignite.engine import Engine, Events
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from argparse import ArgumentParser

from dataset import SmartathonImageDataset
from utils import init_model, collate_fn, dict_to_string, write_string_csv

def load_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main(args):
    model, transforms = init_model(args)
    model = load_checkpoint(args.checkpoint_file, model)
    
    if os.name == 'nt':
        img_dir = args.data_dir+'\\resized_images\\'
        test_data = SmartathonImageDataset(args.data_dir+'\\test_split.json', img_dir, transform=transforms)
    else:
        img_dir = args.data_dir+'/resized_images/'
        test_data = SmartathonImageDataset(args.data_dir+'/test_split.json', img_dir, transform=transforms)
    
    test_iterator = data.DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn)


    meanAP = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    header = ",".join(['class','image_path', 'name', 'xmax','xmin','ymax','ymin','scores'])
    out_csv_list = []
    for batch in test_iterator:
        with torch.no_grad():
            images, targets = batch
            images = list(image.to(device) for image in images)
            img_keys = [t['img_keys'] for t in targets]
            targets = [{k: v for k, v in t.items() if k != 'img_keys'} for t in targets]
            # - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
            # ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            # - labels (``Int64Tensor[N]``): the predicted labels for each detection
            # - scores (``Tensor[N]``): the scores of each detection
            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            meanAP.update(outputs, targets)
            out_csv_list.extend(dict_to_string(img_keys,outputs))

    meanAP_metrics = meanAP.compute()
    print(f"Test Results - meanAP: {meanAP_metrics['map']:.2f}")
    print("\tMetrics:" + ",".join([f"{k}: ({v})" for k, v in meanAP_metrics.items()]))

    #write code to create submission file
    # format cld_ind, filename, cls_name, xmax, xmin, ymax, ymin 
    write_string_csv(out_csv_list,header, os.path.join(args.output_prefix, 'submission.csv'))

if __name__ == "__main__":
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model_type', dest='model_type', 
        help='type of model to train, see model.py for supported model types')
    parser.add_argument('--old_classes', dest='old_classes', default=False, action='store_true',
        help="to run with the old number of classes")
    parser.add_argument('-c', '--checkpoint_file', dest='checkpoint_file',
        help='path to checkpoint file')
    parser.add_argument('-d', '--data_dir', dest='data_dir', 
        help='root directory of the dataset')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=8, type=int,
        help="batch size during inference")
    parser.add_argument('-o', '--output_prefix', dest='output_prefix',
        help="output path for model predictions")
    args = parser.parse_args()
    os.makedirs(args.output_prefix, exist_ok = True) 
    main(args)
