<<<<<<< Updated upstream
import numpy as np
np.__version__
def get_iou(boxGround, boxPred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(boxGround[0], boxPred[0])
    iy1 = np.maximum(boxGround[1], boxPred[1])
    ix2 = np.minimum(boxGround[2], boxPred[2])
    iy2 = np.minimum(boxGround[3], boxPred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 , np.array(0.))
    i_width = np.maximum(ix2 - ix1 , np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = boxGround[3] - boxGround[1] 
    gt_width = boxGround[2] - boxGround[0] 
     
    # Prediction dimensions.
    pd_height = boxPred[3] - boxPred[1] 
    pd_width = boxPred[2] - boxPred[0] 
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    print("Area Ground")
    print(gt_height * gt_width)
    print("Area Predict")
    print(pd_height * pd_width)
    print("Area Union")
    print(area_of_union)
    print("Area Intersect")
    print(area_of_intersection)
    iou = area_of_intersection / area_of_union
     
=======

from utils import csv_to_dict
import pandas as pd


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
>>>>>>> Stashed changes
    return iou

def calc_centroid(xMin, xMax, yMin, yMax):
    #calculate average across x and y coordinates
    xCentroid = (xMin + xMax)/2
    yCentroid = (yMin + yMax)/2

    # create pair for x and y centroid coordinates
    centroid = {xCentroid, yCentroid}
    
    return centroid


if __name__ == '__main__':
    # Pointing out a wrong IoU implementation in https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
<<<<<<< Updated upstream
    boxTruth = [1202, 123, 1650, 868]
    boxPrediction = [1,1,1,1]
=======
    #x max, x min, ymax, ymin
    # this code: xmin, y min, xmax, ymax
    boxA = [701, 211, 797, 262]
    boxB = [700, 215, 800, 250]
>>>>>>> Stashed changes

    correct = get_iou(boxTruth, boxPrediction)

# Load CSV - ground truth and model output

# Process CSV file and convert to dictionary (key by image path) and will gather bounding boxes
train_csv_path = '../data/train.csv'
csv_data = pd.read_csv(train_csv_path)
dict_data = csv_to_dict(csv_data=csv_data)

# Process CSV file and convert to dictionary (key by image path) and will gather bounding boxes
result_csv_path = '../data/train.csv'
csv_data = pd.read_csv(result_csv_path)
result_dict_data = csv_to_dict(csv_data=csv_data)

# Pair n boxes to k boxes, calculate centroid of each box and measure distance, closest ones to each other will be paired
for key in dict_data.keys():
    ground_truth_bb = []
    model_output_bb = []
    problematic_bb = []

    for label_dict in dict_data[key]:
        #ground truth boxes
        xmax, xmin, ymax, ymin = [label_dict.get(key) for key in ['xmax', 'xmin', 'ymax', 'ymin']]     
        ground_truth_bb.append([xmin, ymin, xmax, ymax])
        print(xmax, xmin, ymax, ymin)

    
    for label_dict in result_dict_data[key]:
        #ground truth boxes
        xmax, xmin, ymax, ymin = [label_dict.get(key) for key in ['xmax', 'xmin', 'ymax', 'ymin']]     
        model_output_bb.append([xmin, ymin, xmax, ymax])
        print(xmax, xmin, ymax, ymin)

    #match model box to ground truth box
    model_output_bb.append([xmin, ymin, xmax, ymax])
    break



#calculate centroid given 2 coordinates
#(x min + xmax)/2; (ymin+ymax)/2

#calc_centroid()

# for extra, label as "bad"; for fewer label 'recall'
    

<<<<<<< Updated upstream
    print('Correct solution - also analytical: {}\n'
=======
print('Correct solution - also analytical: {0}\n'
>>>>>>> Stashed changes
          .format(correct))