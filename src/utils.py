import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from collections import defaultdict
import pandas as pd
import csv



def init_model(args):
    num_classes = 12  # 11 classes + background
    model_type = args.model_type
    if model_type == 'resnet50':
        # load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
    elif model_type == 'resnet50_v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
    else:
        raise NotImplementedError(f"model type not recognized {model_type}")

    return model, transforms

def init_optimizer(model, args):
    optim_type = args.optim_type
    params = [p for p in model.parameters() if p.requires_grad]
    if optim_type == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.learning_rate,
            momentum=0.9, weight_decay=0.0005)
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(params, lr=args.learning_rate,
            weight_decay=0.0005)
    else:
        raise NotImplementedError(f"optimizer type not recognized {optim_type}")
        
    warmup_factor = 1.0 / 1000
    warmup_iters = 1000
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
        start_factor=warmup_factor, 
        total_iters=warmup_iters)
    return optimizer, lr_scheduler

def csv_to_dict(csv_data):
    dictData = defaultdict(list)
    #index through each of the items in the csv file and add to the dictionary 
    for index, row in csv_data.iterrows():
        dictData[row['image_path']].append(row.to_dict())
    return dictData

def collate_fn(batch):
    return tuple(zip(*batch))

def dict_to_string(images, coord_data_class):

    full_data_set  = pd.DataFrame(columns = ['labels','image_path', 'xmax','xmin','ymax','ymin', 'scores'])

    for i in images:
        for j in coord_data_class:
            boxes = j.get('boxes')
            labels = j.get('labels')
            scores = j.get('scores')

            my_df = pd.DataFrame(boxes.cpu().numpy(), columns = ['xmax','xmin','ymax','ymin'])
            my_df['labels'] = labels.cpu().numpy()
            my_df['scores'] = scores.cpu().numpy()
            my_df['image_path'] = i
    
            full_data_set = pd.concat([full_data_set, my_df])
            #full_data_set.to_csv('file1.csv',index=False) #save to file for testing
            #stringF = full_data_set.to_string(index=False,header=False)
            #print(stringF)
            #break;
               
    return(full_data_set.to_string(index=False,header=False))
    
    
def write_string_csv(string_obj, headername, filename):
        #only thing missing is the class name
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)

                # write the header
                writer.writerow(headername)

                # write the data
                writer.writerow(string_obj)