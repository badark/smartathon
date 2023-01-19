import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from collections import defaultdict

CLASS_MAPPING=["GRAFFITI", "FADED_SIGNAGE", "POTHOLES", "GARBAGE", "CONSTRUCTION_ROAD", 
"BROKEN_SIGNAGE", "BAD_STREETLIGHT", "BAD_BILLBOARD", "SAND_ON_ROAD", "CLUTTER_SIDEWALK", "UNKEPT_FACADE"]

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

def dict_to_string(img_keys, output_data):
    rows = []
    for i, img_key in enumerate(img_keys):
        data_dict = output_data[i]
        boxes = data_dict.get('boxes').numpy() # stored as xmin, ymin, xmax, ymax - remember to reorder
        cls_inds = data_dict.get('labels').numpy() - 1 # offset by one to remove the background class
        scores = data_dict.get('scores').numpy()
        names = [CLASS_MAPPING[ind] for ind in cls_inds]

        for cls_ind, name, box, score in zip(cls_inds, names, boxes, scores):
            rows.append(",".join(map(str, [cls_ind, img_key, name, box[2], box[0], box[3], box[1]])))

    return(rows)
    
    
def write_string_csv(string_list, header, filename):
    #only thing missing is the class name
    data_string = '\n'.join([header]+string_list)
    with open(filename, 'w') as f:
        f.write(data_string)