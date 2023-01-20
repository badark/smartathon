import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from collections import defaultdict
from IPython.display import display
import cv2
import os
from PIL import Image

CLASS_MAPPING=["GRAFFITI", "FADED_SIGNAGE", "POTHOLES", "GARBAGE", "CONSTRUCTION_ROAD", 
"BROKEN_SIGNAGE", "BAD_BILLBOARD", "SAND_ON_ROAD", "CLUTTER_SIDEWALK", "UNKEPT_FACADE"]

def init_model(args):
    if args.old_classes:
        num_classes = 12
    else:
        num_classes = 11  # 11 classes + background
    model_type = args.model_type
    if model_type == 'resnet50':
        # load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", box_detections_per_img=5)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
    elif model_type == 'resnet50_v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", box_detections_per_img=5)
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
    
    lr_schedule_type = args.lr_schedule_type
    if lr_schedule_type == 'linear':
        warmup_factor = 1.0 / 1000
        warmup_iters = 1000
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
            start_factor=warmup_factor, 
            total_iters=warmup_iters)
    elif lr_schedule_type == 'one_cycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
            max_lr=args.learning_rate, total_steps=30000)
        pass
    else:
        raise NotImplementedError(f"lr scheduler type not recognized {lr_schedule_type}")
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
        cls_inds = data_dict.get('labels').numpy() - 1
            
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

def plot_one_box(x, img, color=None, label=None, line_thickness=None, Inverted=False):

  #since the images are resized resize the bounding boxes coordinates
  x = list(map(lambda x: x/2, x))

  # Plots one bounding box on image img
  tl = line_thickness or 2 # line thickness
  c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
  cv2.rectangle(img, c1, c2, color, thickness=tl)
  if label:
    tf = 1 # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=1)[0]
  if Inverted == True:
    c1 = c2
    c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
  else:
    c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
  cv2.rectangle(img, c1, c2, color, -1) # filled
  cv2.putText(img, label, (c1[0], c1[1] + t_size[1]), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,)


def show_images(ground_truth_dict, final_dict_data):
    #pass in dictionary of images to show
    imagePath= '../data/resized_images/'
    show_dict = final_dict_data

    for key in show_dict.keys():
        
        all_coords = []
        labels_img = []
        ground_all_coords = []
        ground_label_img = []

        for label_dict in show_dict[key]:
            #ground truth boxes
            xmax, xmin, ymax, ymin = [label_dict.get(key) for key in ['xmax', 'xmin', 'ymax', 'ymin']]     
            all_coords.append([xmin, ymin, xmax, ymax])
            label_img = [label_dict.get(key) for key in ['name']]
            labels_img.append([label_img])
            #print(xmax, xmin, ymax, ymin)

        for label_dict in ground_truth_dict[key]:
            #ground truth boxes
            xmax, xmin, ymax, ymin = [label_dict.get(key) for key in ['xmax', 'xmin', 'ymax', 'ymin']]     
            ground_all_coords.append([xmin, ymin, xmax, ymax])
            label_img = [label_dict.get(key) for key in ['name']]
            ground_label_img.append([label_img])
            #print(xmax, xmin, ymax, ymin)

        img = cv2.imread(os.path.join(imagePath, key))

        for item, coord in enumerate(all_coords):
            #print(item)
            #print(labels_img[item][0][0])
            plot_one_box(coord,img, color=(0, 255, 0), label=labels_img[item][0][0], line_thickness=2)
            plot_one_box(ground_all_coords[item],img, color=(255, 0, 0), label=ground_label_img[item][0][0], line_thickness=2)

        im_pil = Image.fromarray(img)

        #add in code to process ground truth image

        #put the two images side-by-side
        #Image.fromarray(np.hstack((np.array(im_pil),np.array(im_pil)))).show()

        #show single image
        im_pil.show()