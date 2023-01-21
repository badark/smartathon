
from utils import csv_to_dict, show_images, show_images_single
import pandas as pd
import numpy as np
import json
from collections import defaultdict



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def calc_centroid(xMin, xMax, yMin, yMax):
    #calculate average across x and y coordinates
    xCentroid = (xMin + xMax)/2
    yCentroid = (yMin + yMax)/2

    # create pair for x and y centroid coordinates
    centroid = {xCentroid, yCentroid}
    
    return centroid


if __name__ == '__main__':
    #x max, x min, ymax, ymin
    # this code: xmin, y min, xmax, ymax
    boxA = [701, 211, 797, 262]
    boxB = [700, 215, 800, 250]

# Load CSV - ground truth and model output

# Process CSV file and convert to dictionary (key by image path) and will gather bounding boxes
train_csv_path = '../data/test_split.json'
#csv_data = pd.read_csv(train_csv_path)
#dict_data = csv_to_dict(csv_data=csv_data)
with open(train_csv_path, 'r') as file:
    dict_data = json.load(file)

# Process CSV file and convert to dictionary (key by image path) and will gather bounding boxes
result_csv_path = '../data/evaluation.csv'
csv_data = pd.read_csv(result_csv_path)
result_dict_data = csv_to_dict(csv_data=csv_data)

images_to_print = defaultdict(list)
groundTruth_mage = defaultdict(list)
mislabeled, bad_iou, unpaired, total_bb = 0,0,0,0
all_problematic_bb = []
accurate_predictions = defaultdict(list)
accurate_ground_truth = defaultdict(list)
#show_images_single(result_dict_data)

for key in dict_data.keys():
    ground_truth_bb = []
    ground_truth_centroids = []
    model_output_bb = []
    model_output_centroids = []
    problematic_bb = []
  


    for label_dict in dict_data[key]:
        #ground truth boxes
        xmax, xmin, ymax, ymin = [label_dict.get(key) for key in ['xmax', 'xmin', 'ymax', 'ymin']]
        total_bb+=1
        ground_truth_bb.append([xmin, ymin, xmax, ymax])
        #print(xmax, xmin, ymax, ymin)

    
    for label_dict in result_dict_data[key]:
        #model result boxes
        xmax, xmin, ymax, ymin = [label_dict.get(key) for key in ['xmax', 'xmin', 'ymax', 'ymin']]     
        model_output_bb.append([xmin, ymin, xmax, ymax])

    num_gt_bbox = len(ground_truth_bb)
    num_mo_bbox = len(model_output_bb)
    iou_table = np.zeros((num_gt_bbox, num_mo_bbox))

    for i, bb_ground_truth in enumerate(ground_truth_bb):
        for j, bb_model in enumerate(model_output_bb):
            iou_table[i,j] = bb_intersection_over_union(bb_ground_truth, bb_model)
    
    #print(ground_truth_bb)
    #print(model_output_bb)

    
    # algorithm to pair the model output bboxes to ground truth bboxes
    # while we have an unassigned gt bbox or there are no more mo bboxes
    #   let u,v be the location of the maximum value in iou_table
    #   create the gt-mo pair (u,v)
    #   zero out the u-th row of iou_table and the v-th column of iou_table
    paired_bbs = [] # ~ a list of tuples of paired bounding boxes
    problematic_bb = []
    gt_bb_inds = np.ones(num_gt_bbox)
    mo_bb_inds = np.ones(num_mo_bbox)
    while np.sum(gt_bb_inds) and np.sum(mo_bb_inds) and iou_table[:].sum():
        u, v = np.unravel_index(iou_table.argmax(), iou_table.shape)
        #print(iou_table)
        #print(u,v)
        #print(gt_bb_inds)
        #print(mo_bb_inds)

        # [u].name != [v].name?
        #Then we found one that has different class names so lets just continue and not add
        #that to the list of paired_bbs
        if dict_data.get(key)[u].get('name') != result_dict_data.get(key)[v].get('name'):
            #good IOU, labels don't match
            mislabeled+=1
            problematic_bb.append((u,v))
            all_problematic_bb.append((u,v))
            gt_bb_inds[u] = 0
            mo_bb_inds[v] = 0
            iou_table[u,:] = 0.0
            iou_table[:,v] = 0.0

            #Need to capture the ground truth image once
            images_to_print[key].append(result_dict_data.get(key)[v])
            groundTruth_mage[key].append(dict_data.get(key)[u])

            continue

        if dict_data.get(key)[u].get('name') == result_dict_data.get(key)[v].get('name') and iou_table[u,v] < 0.5:
            #bad IOU, labels match
            bad_iou+=1
            problematic_bb.append((u,v))
            all_problematic_bb.append((u,v))
            gt_bb_inds[u] = 0
            mo_bb_inds[v] = 0
            iou_table[u,:] = 0.0
            iou_table[:,v] = 0.0

            #Need to capture the ground truth image once
            images_to_print[key].append(result_dict_data.get(key)[v])
            groundTruth_mage[key].append(dict_data.get(key)[u])

            continue

       
        gt_bb_inds[u] = 0
        mo_bb_inds[v] = 0
        iou_table[u,:] = 0.0
        iou_table[:,v] = 0.0
        #print(iou_table)
        accurate_predictions[key].append(result_dict_data.get(key)[v])
        accurate_ground_truth[key].append(dict_data.get(key)[u])
        #we might not even need to store this here as 
        # paired_bbs.append((u,v))

    for i in gt_bb_inds:
        if i == 1:
            unpaired+=1
            problematic_bb.append(i)
            all_problematic_bb.append((u,v))
    for j in mo_bb_inds:
        if j == 1:
            unpaired+=1
            problematic_bb.append(j)  
            all_problematic_bb.append((u,v))

    # print("problematic_bb", problematic_bb)
    # print(paired_bbs)


#print(groundTruth_mage)
#df = pd.DataFrame.from_dict(groundTruth_mage)
#print(df.to_string())

#df = pd.DataFrame.from_dict(images_to_print)
#print(df.to_string())
#print(images_to_print)

print(dict_data)

print("Statistics: ")
print("Number of total bb's in ground truth set: ", total_bb)
print("Number of problematic bb's: ", len(all_problematic_bb))
print("/*%* of problematic bb's: ", ((len(all_problematic_bb))/(total_bb)*100))
print("Number of mislabeled: ", mislabeled)
print("Correctly labeled but low IOU: ", bad_iou)
print("Unpaired boxes: ", unpaired)

show_images(groundTruth_mage,images_to_print) #test to show images from results we will need to move this to the right spot
#show_images(accurate_ground_truth,accurate_predictions) #test to show images from results we will need to move this to the right spot

# for extra, label as "bad"; for fewer label 'recall'